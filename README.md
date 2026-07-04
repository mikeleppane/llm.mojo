# Build a Large Language Model from Scratch in Mojo

A GPT-2-faithful, decoder-only Transformer language model implemented entirely
from scratch in [Mojo](https://mojolang.org/docs/manual/): tokenizer, tensors,
attention, backpropagation, training loop, and generation, with **no ML
framework and no autograd underneath**. Every forward and backward FLOP is
written out, tested, and explained. It is the companion codebase to a written
guide currently in preparation: every chapter's code is built and verified here
first, so nothing gets published that doesn't compile and pass its tests.

## What this is

- A readable, tested path from a bigram model (a lookup table that predicts
  the next token from the current one, the simplest language model that can
  be trained) all the way to a working GPT-2 (124M) that loads the real
  released weights and generates coherent English.
- Teaching code with production discipline: every tensor operation documents
  its shapes, every module ships with tests in the same change, and every
  gradient is checked against finite differences.
- CPU-first. Everything through the core model runs on a laptop. Performance
  work comes late, behind benchmarks, and never at the cost of readability.

What the finished model is *for*: continuing text. Prime it with a few words
and it extends them in the style it learned: Shakespeare-like lines from a
tinyshakespeare run, fluent English prose from the real GPT-2 weights, with
greedy / temperature / top-k / top-p sampling to trade off between safe and
surprising. Facts it absorbed during training can be coaxed out by phrasing
them as completions rather than questions. What it is *not for*: reliable
question answering, following instructions, holding a conversation, or being
trusted on facts. A 124M base model from 2019 will wander and confidently
invent. Knowing exactly why it does that, because you built every part, is the
actual deliverable.

## What this is not

- Not a framework, not a PyTorch clone, not a benchmark contest.
- Not a demo that "mostly works": the capstone test is logit parity with a
  reference GPT-2 implementation on identical prompts.
- Not a chatbot. GPT-2 is a base *completion* model: it predicts the next
  token, nothing more. It has no instruction tuning and no notion of a
  conversation, so it continues text rather than answering questions. "The
  capital of France is" will complete to " Paris"; "What is the capital of
  France?" is more likely to be met with another question, because it is
  imitating the shape of its training text, not obliging a request. The
  assistant behavior people expect from ChatGPT comes from a later stage
  (instruction tuning and RLHF) that is outside this project's scope.

## Why build this

Most explanations of how LLMs work stop at diagrams, and most working
implementations hide the mechanics under framework calls. Building the whole
stack by hand, and proving each piece against an independent oracle, is the
one approach that leaves nothing to trust on faith. The repo optimizes for
**understanding you can verify**: if a claim matters, there is a test that
would fail if it were false.

Faithfulness to GPT-2 (124M) is a hard constraint, not nostalgia: the BPE
vocabulary of 50,257 tokens, learned positional embeddings, pre-LayerNorm
blocks, GELU, weight tying, `d_model=768`, 12 layers, 12 heads. Fidelity is
what makes the endgame possible: loading OpenAI's released weights into our
structs and matching a reference implementation token for token. Architectural
drift would break that, so the tests forbid it.

## Why Mojo

- **Python-family syntax with systems-level control.** The code reads like the
  math, but ownership, value semantics, and explicit copies are part of the
  language. The difference between borrowing and copying a KV cache is
  visible in the source.
- **Compile-time metaprogramming (`comptime`) and first-class SIMD** give a
  real path from "clear and correct" to "fast" without rewriting in another
  language. Correctness lands first; the same code is then optimized in place,
  behind benchmarks.
- **It is young, and that is part of the story.** Mojo evolves quickly; this
  repo pins an exact toolchain version and treats "it compiles under the pin"
  as the only proof of current syntax. The friction and the fixes are
  documented as we go. They are teaching material, not noise.

## How we are building it

The project proceeds in strictly ordered parts (see the roadmap below). Each
part follows the same loop:

1. **Specified before built.** Each part's scope is fixed up front: which
   files, which signatures, which tests, and what "done" means, before any
   implementation starts.
2. **Test-driven.** The failing test is written before the code. Favorite
   patterns: round trips (encode/decode, split/merge, save/load,
   cached-vs-uncached), hand-computed tiny cases, invariants (softmax rows sum
   to 1, cross-entropy gradients sum to 0), and finite-difference gradient
   checks for every backward pass.
3. **Independent oracles.** NumPy and reference implementations act as
   referees in tests, never as scaffolding inside the library. The GPT-2
   tokenizer is proven against OpenAI's reference encoder; attention and
   layer gradients are checked against NumPy; the final model is checked
   against reference logits.
4. **Reviewed and merged green.** Each part lives on its own branch
   (`part-05-tokenization`, `part-06-dataset`, …), passes the format and test
   gates, gets an independent review, and merges to `main` only when
   everything is green. Branches are kept so the guide can reference the exact
   code state of each part.
5. **Determinism everywhere.** A self-contained seeded LCG drives every random
   choice (init, shuffling, sampling). Same seed, same result, on any machine.

### The interop policy

The decision rule: **if a piece of code would appear in a chapter explaining
how an LLM works, it is Mojo.** Python interop is reserved for plumbing
(downloading files, parsing `vocab.json`, reading safetensors) and for test
oracles. One narrow carve-out: GPT-2's pre-tokenizer split uses a
Unicode-category regex, which calls Python's `regex` module. The BPE merge
loop itself is pure Mojo. "NumPy scaffolding now, Mojo later" is a forbidden
pattern: it never gets replaced, and published code must be the verified code.

### Numerics policy

Reference math is `Float64` with tight tolerances (`1e-9` to `1e-12`); `Float32`
enters later as a deliberate performance decision with retuned tolerances.
Floats are never compared with `==`. Softmax and log-sum-exp are implemented
in their numerically stable forms from day one. The stable version is the
reference, not an optimization.

## Architecture

```text
src/llm/          the library, one package per concern
  config.mojo       model/training configuration
  vocab.mojo        toy vocabulary (chapter II)
  tokenizer/        char tokenizer + byte-level BPE + GPT-2 tokenizer
  data/             corpus loading, train/val split, batching
  tensor/           Tensor2D/3D/4D and ops (matmul, softmax, cross-entropy, ...)
  nn/               linear, embedding, norms, activations, MLP
  transformer/      masks, attention, positional encoding, blocks, the model
  models/           small self-contained models (bigram, ...)
  training/         loss, optimizer, trainer, checkpointing
  generation/       sampling, KV cache, generation
  utils/            seeded RNG, timing helpers
examples/         runnable demonstrations
tests/            correctness checks (TestSuite; one file per unit)
benchmarks/       performance measurement (after correctness)
data/             committed reference data (GPT-2 vocab/merges, tiny Shakespeare)
scripts/          download scripts (provenance + checksums) and dev tooling
```

Dependencies flow in one direction only, and lower layers never import higher
ones:

```text
utils  →  tensor  →  nn  →  transformer  →  { training, generation }
config + vocab  →  nn, transformer
tokenizer  →  data  →  models  →  { training, generation }
```

Two conventions do a lot of work here. Every tensor-shaped value carries a
shape comment (`# [B, T, C]`), and every public tensor function documents four
facts: shapes in/out, whether it mutates, whether it allocates, whether it can
raise. [ARCHITECTURE.md](ARCHITECTURE.md) is the full tour: the design and the
reasoning behind it, with diagrams. Reference data (~2.5 MB total) is committed so tests and CI run
offline and deterministically; download scripts with pinned URLs and SHA-256
checksums document provenance.

## Roadmap

| Part | Topic |
|---|---|
| I-IV | Foundations: toolchain, Mojo language, numerics, tensors, ops, RNG |
| V | Tokenization: char tokenizer, byte-level BPE in Mojo, GPT-2 vocab parity |
| VI | Dataset pipeline: tiny Shakespeare, windows, batching, seeded shuffling |
| VII | Bigram language model: table logits, cross-entropy, first training loop |
| VIII-IX | Architecture overview; NN building blocks (linear, GELU, LayerNorm, MLP) |
| X | Attention: multi-head, causal + padding masks, `[B, H, T, D]` discipline |
| XI | Backpropagation by hand: every backward finite-difference-checked |
| XII | Encoder-decoder lab: copy/reverse tasks validate cross-attention |
| XIII | The GPT-2 model: pre-LN blocks, weight tying, 124,439,808 parameters |
| XIV | Training: AdamW, LR schedules, clipping, checkpoints, overfit-one-batch |
| XV | Generation: greedy, temperature, top-k, top-p, stop tokens |
| XVI | Loading real GPT-2 weights and generating coherent text (the MVP) |
| XVII | KV cache: exact-match cached generation, measured speedup |
| XVIII | Performance: profiling, allocation removal, tiling, SIMD |
| XIX | Validation and CI: the full gauntlet, logit-parity capstone |
| XX | Finetuning and extensions: SFT, LoRA, RoPE/RMSNorm/SwiGLU variants |

The order is deliberate, and the first model is deliberately weak. A *bigram*
language model predicts the next token from only the current token: no
context, no attention, just a `vocab × vocab` table of scores, trained with
gradient descent. It cannot write Shakespeare, but it exercises the complete
machinery end to end: tokenized data in, batches through a loss, gradients
into an optimizer, samples out. Because the model is trivial, every number it
produces can be verified by hand, including its theoretical best loss, which
the training loop must converge to. From there, each part replaces one piece
with something stronger (embeddings, attention, full Transformer blocks) while
the surrounding machinery (data pipeline, loss, training loop, sampling)
stays tested and familiar. When something breaks, you always know which layer
broke, because everything under it was already proven.

Live status per part is tracked in [PROGRESS.md](PROGRESS.md).

## Getting started

The only prerequisite is [pixi](https://pixi.sh); it installs the pinned Mojo
toolchain and Python environment from the lockfile.

```bash
git clone <this repo> && cd mojo-llm-from-scratch
pixi install            # exact environment from pixi.lock
pixi run mojo-version   # verify the pinned Mojo version
pixi run test           # run the full test suite (offline, deterministic)
```

Day-to-day commands:

```bash
pixi run fmt                                      # format all Mojo sources
pixi run mojo run -I src tests/test_matmul.mojo   # run a single test file
pixi run mojo run -I src examples/<example>.mojo  # run an example
```

`-I src` puts the `llm` package on the import path. The test suite needs no
network: all reference data is committed.

## Getting the most out of this if you are learning

- **Read the tests as specifications.** Each `tests/test_*.mojo` file states,
  in executable form, exactly what its module guarantees. The hand-computed
  tiny cases (a 2×3 matmul, a 5-token corpus) are worked examples.
- **Use the part branches.** Each part's branch contains the codebase exactly
  as that part left it. Check out `part-05-tokenization` to see the repo
  when only the tokenizer existed. History is small, atomic, and explained:
  `git log` reads as a build narrative.
- **Break things on purpose.** Remove the row-max subtraction from softmax and
  watch the stability test fail; flip a sign in a gradient and watch the
  finite-difference check catch it. The tests are cheap to run and localize
  failures precisely. That is what they are for.
- **Follow the shapes.** Every operation is annotated with `[B, T, C]`-style
  comments. Most LLM implementation bugs are shape bugs; the annotations are
  how you follow the dimensions without running anything.
- **Read the notes.** `notes/part-*.md` records what broke, what surprised us,
  and why decisions went the way they did: the parts of engineering that
  polished final code hides.

## Conventions and contributing

Working conventions (the Mojo syntax contract, quality gates, dependency
layering, commit format) live in [AGENTS.md](AGENTS.md), with deeper
task-specific guidance under [`.agents/skills/`](.agents/skills/). The floor
for any change: `pixi run fmt` leaves no diff and `pixi run test` is green.
CI enforces both on every push ([ci.yml](.github/workflows/ci.yml)).

## Data and references

- [Mojo manual](https://mojolang.org/docs/manual/), the language reference
  for the pinned version.
- Tiny Shakespeare corpus from Andrej Karpathy's
  [char-rnn](https://github.com/karpathy/char-rnn) data; his
  [nanoGPT](https://github.com/karpathy/nanoGPT) and
  [minbpe](https://github.com/karpathy/minbpe) shaped several design choices
  here, as did OpenAI's original GPT-2 release (tokenizer files and the
  reference encoder used as a test oracle).
- Sebastian Raschka's *Build a Large Language Model (From Scratch)*, a kindred
  spirit in a different language.

## License

TBD.
