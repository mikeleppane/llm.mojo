# Architecture

This document explains how the codebase is put together and why it is shaped
this way. [README.md](README.md) says what the project is; [AGENTS.md](AGENTS.md)
lists the working rules; this document covers the design: the system's parts,
the boundaries between them, and the reasoning behind each decision, with
examples from the actual code.

Parts I–XIX have landed: the full GPT-2 (124M) model, training, generation, the
KV cache (incremental decode), the SIMD + threading performance work, loading
OpenAI's real released weights, and the validation gauntlet all exist and are
tested. [PROGRESS.md](PROGRESS.md) tracks what exists today per part.

## The goal, and the three constraints that shape everything

The end state is a decoder-only Transformer that loads OpenAI's released
GPT-2 (124M) weights into our own structs and matches a reference
implementation's logits on identical prompts. Working backward from that goal,
three constraints decide almost every design question:

1. **This is teaching code.** The repo is the companion to a written guide, so
   the code is read more often than it is run. When a clearer version and a
   faster version compete, the clearer one wins until a benchmark says
   otherwise.
2. **Correctness must be checkable, not asserted.** Every claim the guide
   makes has a test that would fail if the claim were false. Independent
   oracles (NumPy, OpenAI's reference tokenizer, hand computation) referee the
   Mojo implementation.
3. **GPT-2 fidelity is a hard constraint.** Byte-level BPE with exactly 50,257
   tokens, learned positional embeddings, pre-LayerNorm blocks, GELU, weight
   tying, `d_model=768`, 12 layers, 12 heads, context 1024. Architectural
   drift anywhere would break weight loading and the logit-parity test, so the
   tests pin these numbers.

The reference configuration:

| Parameter | Value |
|---|---|
| vocabulary | 50,257 (GPT-2 BPE) |
| context length | 1024 |
| d_model | 768 |
| layers | 12 |
| heads | 12 |
| parameters | 124,439,808 (asserted by test) |

## System overview

Two data paths run through the system. Training turns text into weight
updates; generation turns a prompt into new text. They share the tokenizer
and the model forward pass.

```mermaid
flowchart TD
    subgraph shared["Shared"]
        TOK["tokenizer<br/>text to ids and back"]
        MODEL["model forward<br/>ids [B, T] to logits [B, T, V]"]
    end

    subgraph train["Training path"]
        CORPUS["corpus (tiny Shakespeare)"] --> TOK
        TOK --> DS["TokenDataset<br/>train/val split"]
        DS --> BL["BatchLoader<br/>seeded windows"]
        BL --> BATCH["TokenBatch<br/>inputs + targets [B, T]"]
        BATCH --> MODEL
        MODEL --> LOSS["cross-entropy loss"]
        LOSS --> GRAD["hand-written backward"]
        GRAD --> OPT["optimizer step"]
        OPT --> MODEL
    end

    subgraph gen["Generation path"]
        PROMPT["prompt"] --> TOK
        TOK --> MODEL
        MODEL --> SAMPLE["sampler<br/>greedy / temperature / top-k / top-p"]
        SAMPLE --> NEXT["next token id"]
        NEXT --> MODEL
        NEXT --> TOK
        TOK --> TEXT["generated text"]
    end
```

The shapes in the diagram are part of the design, not decoration. Token ids
are integers with shape `[B, T]` (batch, time). The model maps them to float
logits `[B, T, V]`. Keeping the integer world (tokenizer, data) separate from
the float world (tensors, model) is one of the load-bearing boundaries below.

## Packages and the one-direction rule

```text
src/llm/
  config.mojo       model and training configuration
  vocab.mojo        toy whitespace vocabulary (an early chapter's example)
  tokenizer/        char tokenizer, byte-level BPE, GPT-2 tokenizer
  data/             corpus loading, splits, windowing, batching
  tensor/           Tensor2D/3D, matmul, softmax, cross-entropy, init
  nn/               parameter, linear, embedding, norms, activations, MLP, AdamW math
  transformer/      masks, attention, positional encoding, blocks, the GPT model,
                    the released-weight loader
  models/           small self-contained models (bigram)
  training/         loss, optimizer, LR schedule, trainer, checkpoints
  generation/       sampling and the autoregressive generation loop
  lab/              Part-XII encoder-decoder lab (cross-attention), quarantined
  utils/            seeded RNG, timing helpers
```

Tensors currently go up to rank 3 (`Tensor2D`, `Tensor3D`); the four-dimensional
attention shapes below are realized by indexing and looping over those, not by a
dedicated `Tensor4D`. The KV cache (Part XVII) lives in
`transformer/kv_cache.mojo`, consumed by `generate_cached` in `generation/`.

Dependencies flow in one direction. A package may import from packages to its
left in the graph below, never from the right. The **authoritative** version of
this graph — with every edge and the reasoning — is the "Dependency layering"
section of [AGENTS.md](AGENTS.md#dependency-layering--one-direction-only); this
diagram mirrors it:

```mermaid
flowchart LR
    utils --> tensor --> nn --> transformer
    utils --> data
    tensor --> models
    data --> models
    config --> transformer
    config --> training
    data --> training
    models --> training
    transformer --> training
    transformer --> generation
    transformer --> lab
    training --> lab
```

`vocab` and `tokenizer` import nothing internal and are not yet imported by any
`src/` package (each is exercised only by its own test/example today), so they
are omitted from the edges above. `lab` is a quarantined Part-XII leaf that sits
above `training`; nothing imports it.

The rule exists because a cycle makes code impossible to test in isolation: if
`utils` imported `tensor`, you could not test the RNG without compiling the
tensor library. This is not hypothetical. The guide's own draft chapter placed
`xavier_2d` (weight initialization, which needs `Tensor2D`) next to the RNG in
`utils/`. That would have pointed an import from `utils` up to `tensor`, so
the function lives in `src/llm/tensor/init_weights.mojo` instead: the RNG
knows nothing about tensors, and tensor initialization knows about both.

Each package's `__init__.mojo` re-exports its public surface and contains no
other code:

```mojo
# src/llm/data/__init__.mojo
from .corpus import load_text
from .dataset import TokenDataset, TrainValSplit, train_val_split
from .batch import TokenBatch
from .loader import BatchLoader, overfit_batch
```

Callers write `from llm.data import BatchLoader` and never depend on file
names inside the package, so files can move without breaking anything.

## Shape discipline

Most LLM implementation bugs are shape bugs, so shapes are a documented
contract rather than something the reader reconstructs from loop bounds. Six
letters are used consistently everywhere:

| Letter | Meaning |
|---|---|
| B | batch size |
| T | sequence length |
| C | model dimension (`d_model`) |
| H | number of heads |
| D | head dimension (`C / H`) |
| V | vocabulary size |

Every public tensor function documents four facts: shapes in and out, whether
it mutates its inputs, whether it allocates, and whether it can raise.

```text
def softmax_rows(scores: Tensor2D) -> Tensor2D:
    """Row-wise numerically stable softmax.

    Args:
        scores: Input scores, shape [rows, cols].

    Returns:
        probs [rows, cols], each row summing to ~1. Allocates a new tensor; does
        not mutate the input; does not raise.
    """
    ...
```

The convention earns its keep in attention, where one forward pass moves
through four shapes:

```mermaid
flowchart LR
    X["x [B, T, C]"] --> P["Q, K, V projections"]
    P --> Q["q, k, v [B, H, T, D]"]
    Q --> S["scores [B, H, T, T]"]
    S --> M["masked softmax"]
    M --> O["per-head output [B, H, T, D]"]
    O --> R["merged [B, T, C]"]
```

Tests pin these shapes explicitly, and the split-heads/merge-heads round trip
(`merge(split(x)) == x`) is a standing test, because a transposed stride in
that path produces plausible-looking garbage rather than a crash.

## Core abstractions

### Tensors: flat storage, explicit offsets

There is no framework tensor to lean on, and the standard library does not
provide one. The project defines its own, and the first version is
deliberately simple: a struct with dimensions and a flat `List[Float64]` in
row-major order.

```mojo
struct Tensor2D(Copyable, Movable, Writable):
    var rows: Int
    var cols: Int
    var data: List[Float64]  # flat row-major [rows, cols]

    def offset(self, row: Int, col: Int) -> Int:
        return row * self.cols + col

    # Fast, unchecked access for hot loops. Returns a reference into the buffer,
    # so one method serves read, write, and += (no separate setter).
    def __getitem__(ref self, row: Int, col: Int) -> ref[self.data] Float64:
        return self.data[self.offset(row, col)]

    # Checked access for tests and debugging; forces a raises context.
    def at(self, row: Int, col: Int) raises -> Float64:
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise Error("Tensor2D index out of range")
        return self.data[self.offset(row, col)]
```

Two access paths is a deliberate choice: `t[i, j]` is fast by default, `at()`
is checked on demand. The offset arithmetic is public and tested directly
(`offset(1, 2, 3) == 23` for a `[2, 3, 4]` tensor), because pinning the
memory layout in a test catches stride bugs before they become model bugs.
Higher ranks repeat the same pattern: `Tensor3D` uses
`(i * d1 + j) * d2 + k`. There is no `Tensor4D` — attention's four-dimensional
`[B, H, T, D]` shapes are handled by indexing and looping over `Tensor2D`/
`Tensor3D`, which keeps the storage story simple; a dedicated rank-4 tensor can
arrive later if batching needs it. The `List`-backed storage survived the
performance work: Part XVIII vectorized and threaded the kernels through
`List.unsafe_ptr()` without a storage redesign, so the `List` interface stayed
put and the existing tests stayed the safety net.

### Integer data stays integer

Token ids never live in float tensors. The data pipeline produces
`TokenBatch`, a flat row-major `[B, T]` pair of integer lists (inputs and
targets), and the shift-by-one relationship between them is built into
window construction: a window of `T + 1` tokens becomes inputs
`ids[s : s+T]` and targets `ids[s+1 : s+T+1]`. The property
`target[b, t] == input[b, t+1]` is then verified by test anyway. Storing ids
as floats would lose exactness for large vocabularies and blur the type
boundary that keeps `data/` independent from `tensor/`.

### The tokenizer family

Three tokenizers, in increasing order of realism:

| Tokenizer | What it is | Why it exists |
|---|---|---|
| `CharTokenizer` | one id per codepoint, vocabulary built from the corpus | smallest possible vocabulary (65 for tiny Shakespeare); ideal for the first trained models |
| `BPETokenizer` | byte-level BPE core: ids over raw bytes, merge loop, trainer | the algorithm GPT-2 actually uses, in a form small enough to read |
| `GPT2Tokenizer` | BPE core + GPT-2's pre-tokenizer and vocabulary files | exact parity with the real GPT-2, proven against OpenAI's reference encoder |

The BPE core works on integer token ids over raw bytes rather than on
GPT-2's unicode-remapped strings. Both formulations produce identical output
(the remapping is a bijection); the integer form keeps the merge loop free of
UTF-8 string handling, and the byte-to-unicode table appears only in the file
loader, where GPT-2's `vocab.json` and `merges.txt` formats require it.

### One RNG, owned by the project

All randomness (weight init, shuffling, dropout, sampling) flows through one
seeded linear congruential generator in `utils/random.mojo`, using Knuth's
MMIX constants. Owning the RNG buys two things: identical results on every
machine (no dependence on a standard library RNG's version-to-version
behavior), and testability (the first outputs for a given seed are frozen as
golden values, computed independently in Python). Uniform doubles come from
the top 53 bits; normals come from Box-Muller; both are additions on top of
the same integer stream.

## The correctness architecture

Testing is not a phase here; it is most of the architecture. The pyramid:

| Layer | Count | Examples |
|---|---|---|
| unit | many | offset layout, matmul against hand-computed values, softmax rows sum to 1, RNG goldens, config validation |
| component | some | attention forward shapes, causal-mask leakage, block forward |
| integration | few | overfit-one-batch, tokenizer parity with the reference encoder, checkpoint round trip, cached-vs-uncached generation equality |

Five patterns recur, and knowing them explains most test files:

**Round trips.** Any operation with an inverse gets the inverse test:
`decode(encode(text)) == text`, `transpose(transpose(a)) == a`,
`merge(split(x)) == x`, `load(save(t)) == t`. These are cheap to write and
brutal to fool.

**Hand-computed tiny cases.** A 2x3 matmul with values checkable on paper, a
five-token corpus whose bigram counts you can tally by eye. Small inputs make
failures readable.

**Invariants.** Softmax rows sum to 1. Cross-entropy gradients sum to 0. A
uniform model's loss equals `log(V)` exactly. Masked attention rows
renormalize to 1.

**Oracles.** An independent implementation referees ours. The GPT-2 tokenizer
must match OpenAI's reference encoder token for token on a fixed sample set.
Analytic gradients must match central finite differences:

```text
df/dx ~ (f(x + h) - f(x - h)) / 2h      with h ~ 1e-5 for Float64
```

The step size is chosen where truncation error (wants small h) and
floating-point cancellation (wants large h) balance, near sqrt(machine
epsilon). Every backward pass in the project must pass this check before it
is trusted.

**Overfit-one-batch.** The highest-value integration test: a correct model
with a correct training loop drives the loss on one small batch to
approximately its theoretical floor. For most models that floor is near zero.
The bigram model is the instructive exception: its floor on real text is the
batch's conditional entropy, and the test asserts convergence to the
count-model optimum instead. When this test fails, the bug is in the loss,
the gradients, or the optimizer, in that order of likelihood.

Mechanics: one test file per unit under `tests/`, discovered by `TestSuite`
and run with `mojo run` (there is no `mojo test` subcommand). The whole suite
runs offline; every reference file the tests need is committed.

## Numerics policy

Reference math is `Float64`. `Float32` arrives late, as a measured
performance decision, with tolerances retuned. Floats are never compared with
`==`; tolerances are sized to the precision:

| Comparison | Tolerance |
|---|---|
| Float64 reference math | 1e-9 to 1e-12 |
| Float32 model math | 1e-4 to 1e-5 |
| finite-difference gradient checks | ~1e-4 (limited by h, not dtype) |

Numerically stable formulations are the reference implementations, not
optimizations. Softmax subtracts the row maximum before exponentiating,
because attention scores overflow a naive `exp` in practice:

```mojo
var max_value = scores[r, 0]
for c in range(1, cols):
    if scores[r, c] > max_value:
        max_value = scores[r, c]
var denom = 0.0
for c in range(cols):
    var e = exp(scores[r, c] - max_value)   # largest exponent is exp(0) = 1
    out[r, c] = e
    denom += e
```

The test for this is direct: `softmax_row([1000.0, 1000.0, 1000.0])` must
return a clean uniform distribution, which the naive form fails with NaN.
Cross-entropy uses the log-sum-exp identity for the same reason. Every
epsilon (the LayerNorm denominator guard, the `log(0)` guard in Box-Muller)
is documented at its point of use with the reason it is there.

## The interop boundary

Mojo is the subject of the guide; Python is allowed at the edges. The
decision rule: **if a piece of code would appear in a chapter explaining how
an LLM works, it is Mojo.**

```mermaid
flowchart TD
    subgraph mojo["Mojo (the subject: every FLOP)"]
        A["tensors and ops"]
        B["all layers and attention"]
        C["BPE merge loop"]
        D["backward passes"]
        E["optimizer"]
        F["sampling and generation (with KV cache)"]
    end

    subgraph plumbing["Python interop (plumbing)"]
        G["file downloads"]
        H["vocab.json / merges.txt parsing"]
        I["safetensors reading (weight loading)"]
        J["regex pre-tokenizer split (one carve-out)"]
    end

    subgraph referee["Python interop (referee, tests only)"]
        K["NumPy gradient references"]
        L["OpenAI reference encoder"]
    end

    plumbing --> mojo
    mojo -.->|compared in tests| referee
```

The one carve-out inside tokenization: GPT-2's pre-tokenizer splits text with
a Unicode-category regex (`\p{L}`, `\p{N}`) that Python's `regex` module
provides and a from-scratch implementation would not teach anything about
LLMs. The split calls Python; every byte after the split is processed by
Mojo.

One pattern is banned outright: "NumPy scaffolding now, Mojo later." A
placeholder implementation never gets replaced, and the published code must
be the verified code. Referees live in test files, never in `src/`.

## Determinism

Every random choice takes an explicit seed, and identical seeds produce
identical results under the pinned toolchain on the supported platform
(`linux-64` today) — the project owns its RNG rather than leaning on a
standard-library generator whose stream could shift between versions.
Concretely: batch order for an epoch is a
Fisher-Yates permutation of window start offsets driven by `Rng(seed)`, so
"same seed, same batches" is an exact equality test, not a statistical claim.
Greedy generation is deterministic and asserted token for token. Golden
values (RNG outputs, frozen tokenizations, later frozen logits) catch
accidental behavior changes during refactors; a commit that claims "no
behavior change" while a golden test moves is lying, and the suite says so.

Determinism is what makes the guide reproducible (a reader gets the same loss
curve the chapter shows) and what makes failures bisectable (a regression
reproduces on the first try).

## The model

The main line builds up to the GPT-2 architecture in `transformer/`, which now
exists end to end. The forward pass processes one sequence at a time — ids come
in as a `List[Int]` and logits come out as a `Tensor2D [T, V]` — so the batch
dimension `B` in the shape letters below is the conceptual notation, not a
materialized tensor axis:

```mermaid
flowchart TD
    IDS["token ids [T]"] --> TE["token embedding [V, C]"]
    POS["positions 0..T-1"] --> PE["positional embedding [1024, C]"]
    TE --> ADD["x = tok + pos [T, C]"]
    PE --> ADD
    ADD --> BLK["12 x pre-LN block:<br/>x = x + attn(LN(x))<br/>x = x + mlp(LN(x))"]
    BLK --> LN["final LayerNorm"]
    LN --> HEAD["LM head [C, V]<br/>(weights tied to token embedding)"]
    HEAD --> LOGITS["logits [T, V]"]
```

Gradients are hand-written per-layer backward passes rather than a tape-based
autograd. The reasoning: the guide's job is to show what backpropagation
actually computes, and a reader can verify a `linear_backward` against the
chain rule directly, while an autograd hides the mechanics it is supposed to
teach. The cost is more code per layer; the mitigation is that every backward
is finite-difference checked, and the highest-risk ones are also checked
against NumPy.

Four contracts inside the model do a disproportionate amount of the work and are
worth calling out, because they are where a subtle change breaks everything
downstream at once.

**The parameter walk is the registry.** With no framework parameter dictionary,
the model's fixed traversal order — `wte` once, `wpe`, each block's twelve
parameters in layer order, then `ln_f` — *is* the registry. The optimizer state
(Adam's `m`/`v`), gradient-norm clipping, the checkpoint format, and
export/import all index that one order, so every walk method must visit the same
parameters in the same order. The per-block twelve are authored in a single
place (`transformer/block.mojo`) so the order lives once; order drift between
methods is the named failure mode, guarded by shape/count reconciliation, a
decay-partition inventory, a checkpoint round trip, and an against-oracle
optimizer run.

**Weight tying is one Parameter reached by two paths.** The token embedding and
the LM head share a single `[V, C]` Parameter. In one backward call it receives
gradient from *both* the head matmul and the embedding gather; those two
contributions are summed into one delta and accumulated with a single `+=`, so
that two backward passes produce a bit-for-bit `2×` gradient (an exact-doubling
test pins this — one of the exactness contracts, not a tolerance comparison).

**Forward caches are explicit; backward boundaries are hand-drawn.** Each layer's
`forward` returns the intermediate values its `backward` needs (a per-layer cache
struct that is moved, not copied), rather than a global tape. The boundary
between a layer's forward and backward is something a reader can see and a test
can check in isolation.

**Two weight formats, kept distinct.** A *trainer checkpoint* stores the full
training state (parameters plus optimizer `m`/`v`, step count, RNG) bit-for-bit
so a run resumes exactly; the *released-weight loader* reads OpenAI's published
GPT-2 into the `GPT2W v1` format — a one-way import of the f32 payload, widened
`f32 → f64` exactly on read — with no optimizer state. Conflating the two is a
category error the module boundaries prevent.

Before the full model, a small **encoder-decoder lab** (`src/llm/lab/`, Part XII)
built from the same blocks trains on copy and reverse tasks — the cheapest way to
prove cross-attention and the training loop end to end. The lab is quarantined
off the main line: nothing in the model imports it, and its one heavy test is the
file CI's gate skips (see [AGENTS.md](AGENTS.md)).

Generation comes in two forms. The **uncached** `generate` reruns the forward
pass over the whole sequence each token — the correct, readable baseline.
`generate_cached` (Part XVII) reuses a per-layer KV cache to avoid recomputing
attention over the past, producing logits bit-identical to the uncached path at
every prefix (pinned by a test); it is the optimization measured against the
baseline.

## Performance, last

The order is fixed: correct, then tested, then measured, then fast. Nothing
is optimized without a benchmark showing where time goes, and no optimization
may fail an existing test. When a SIMD version of an op lands, the scalar
version stays, both as the readable reference and as the oracle the SIMD
version is tested against.

The house example of the whole policy is matmul loop order: `ijk` and `ikj`
compute identical results (a test proves it), but `ikj` walks both inner-loop
operands contiguously and wins on realistic sizes from cache behavior alone —
same math, measured difference, zero readability cost. The production `@` kernel
then builds on the `ikj` order with SIMD over columns and threading (Part XVIII),
each step benchmarked and leaving the scalar reference in place as its oracle.
That is the shape of acceptable performance work here. The endgame bottleneck is
single-token decode, memory-bandwidth-bound matvec; the KV cache (Part XVII)
addresses the algorithmic side — never recompute attention over the prompt — and
produces output bit-identical to the uncached path, pinned by a test.

## Where code goes

| It is... | It goes to... |
|---|---|
| reusable implementation | `src/llm/` |
| a runnable demonstration | `examples/` |
| proof of correctness | `tests/` |
| a performance measurement | `benchmarks/` |
| a download/provenance script | `scripts/` |
| committed reference data | `data/` |

Reference data (GPT-2 vocabulary and merges, tiny Shakespeare; about 2.5 MB
total) is committed so the suite runs offline and CI is deterministic. Each
committed file has a download script recording its source URL and SHA-256.

## Stability and change

Each roadmap part lives on its own branch (`part-05-tokenization`,
`part-06-dataset`, ...) and merges to `main` only when formatting and the
full suite are green. Branches are kept after merging so the guide can point
at the exact state of the code as of any part.

Public surfaces (`__init__.mojo` re-exports) are the stable contract; struct
internals are not. One internal change remains planned: the model's working dtype
narrows from `Float64` to `Float32` (the Part XVIII performance work kept
`Float64` and needed no storage redesign — it vectorized through
`List.unsafe_ptr()`). It will happen behind existing interfaces, and the
definition of "safe to change" holds: every existing test still passes.
