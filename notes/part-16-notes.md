# Part XVI — Loading real GPT-2 weights (the MVP): build notes

The project's MVP: pour OpenAI's actual GPT-2 124M weights into the from-scratch
GPT struct and let the Mojo forward pass generate real English. Three
deliverables carry it — an offline Python converter (HF safetensors → our own
minimal binary format), a native Mojo loader, and the parity discipline that
proves the port is EXACT. The failure mode this part exists to defeat is SILENT
GARBAGE: a missing transpose on a square kernel, a swapped walk slot, an ingested
mask buffer — all of which shape-check fine and emit plausible-looking token soup.

## The byte-IO spike (day one, before building anything)

The one unverified assumption going in was Mojo-side binary file reading. The
spike: Python writes an ASCII header line then raw little-endian float32 with
`struct.pack("<f", …)`; Mojo reads it back and reconstructs the floats.

Outcome — it works, with two lessons:

- `open(path, "r").read_bytes()` returns all bytes (a `List[UInt8]`). **Mode
  `"rb"` is REJECTED** ("invalid mode: 'rb'. Can only be one of {r, w, rw, a}") —
  `read_bytes()` is a method on the ordinary `"r"` handle, not a binary-mode flag.
- Reconstructing a little-endian float32 from 4 bytes
  (`bitcast[DType.float32,1](SIMD[DType.uint32,1](bits))[0]`) then widening with
  `Float64(f32)` reproduces Python's packed value BIT-EXACT. The f32→f64 widening
  is exact by IEEE (every float32 is representable as a float64); a bit-level test
  pins 0.1f32 → 0x3FB99999A0000000 (4591870180174331904). No text-format fallback
  was needed.

## The GPT2W v1 format

```
line 0 (ASCII, newline-terminated):
    GPT2W v1 <V> <T> <C> <L> <H> <param_count>
then the raw little-endian float32 payload, every parameter tensor back to back
in THE MODEL'S WALK ORDER, row-major within each tensor.
```

No per-tensor shape records: shapes are DERIVED from (V,T,C,L,H) by both writer
and reader, so the walk IS the single source of truth. The header's declared
`param_count` and the payload's byte length cross-check each other and the dims.
Header validation raises named errors for: bad magic (family token != `GPT2W`),
unsupported version (tag != `v1`), malformed field count, invalid dims (via
`cfg.validate()` — e.g. C not divisible by H), header count mismatch, truncated
payload, and trailing bytes.

**Why it coexists with the Part XIV checkpoint (`GPTCKPT 1`).** The checkpoint is
hex-text for BIT-EXACT trainer resume and carries the AdamW moments (m, v), the
step counter, and the rng state a run needs to continue identically. A released
model has none of that. At 124M the checkpoint's hex-text would be multi-GB
parsed line by line; GPT2W stores the released float32 precision as raw bytes
(~475 MB payload; the .bin is 498 MB) and widens f32→f64 exactly on read. Two
jobs, two formats.

## The walk order, reconciled against Part XIV as LANDED

An earlier draft walk (qkv → proj → ln1 → ln2 → up → down) was NOT what Part XIV
shipped. The LANDED per-block order (authored once in
`transformer/block.mojo`, and the order every walk method uses) is:

```
ln1.w, ln1.b, qkv.w, qkv.b, proj.w, proj.b, ln2.w, ln2.b, up.w, up.b, down.w, down.b
```

with wte (once — tied head), wpe before the blocks and ln_f.w, ln_f.b after. The
converter, the Mojo loader, and both reference forwards all use this landed
order. Reconciling against the landed code (not an earlier draft ordering) was
load-bearing; the file order IS the model's walk order.

## The layout fixes as shipped (the silent-garbage list)

All in the converter (`scripts/convert_gpt2_weights.py`), the ONE place with
GPT-2 layout knowledge. The Mojo loader is deliberately dumb.

- **Transposed** every HF Conv1D kernel (HF stores [in, out]; our Linear is
  [out, in], y = x @ W^T + b): `c_attn.weight [C,3C]→qkv [3C,C]`,
  `attn.c_proj.weight [C,C]→proj [C,C]` (SQUARE — a wrong transpose still
  shape-checks), `mlp.c_fc.weight [C,4C]→up [4C,C]`,
  `mlp.c_proj.weight [4C,C]→down [C,4C]`.
- **NOT transposed**: `wte [V,C]`, `wpe [T,C]` (row-gather tables like ours), and
  every LayerNorm weight/bias.
- **Reshaped** every 1-D vector (all biases, all LN params) to a `[1, out]` row.
- **Skipped** the buffers `h.{i}.attn.bias` (the [1,1,1024,1024] causal mask) —
  12 of them, one per layer. `attn.masked_bias` is not present in this file.
- **lm_head**: not present in this safetensors (the head is tied and not stored).
  The converter asserts-equals-and-drops it IF present; here there was nothing to
  drop.
- **Prefix**: this file's names have NO `transformer.` prefix (the `GPT2Model`
  form). The converter strips it if present, so both HF namings load.
- **Column order**: needed NO fix — HF's c_attn packs Q|K|V in thirds and splits
  heads contiguously, exactly our attention convention. The 124M parity gate was
  the final arbiter.
- The converter PULLS each tensor by our inventory (raising on a missing name),
  never iterates the HF file pushing — so a buffer can never land in a weight slot.
  It reported "skipped 12 mask buffers" and no surprise tensors.

## The loader builds the GPT directly, fieldwise — no donor

`load_gpt2` constructs Parameter/Linear/Embedding/LayerNorm/MHA/MLP/
TransformerBlock/GPT via their `@fieldwise_init` constructors in walk order as it
streams the payload. No `init_random` donor (that would draw ~124M Box-Muller
normals only to overwrite them and perturb an rng for nothing), no new
`init_zeros` surface. A post-build `parameter_count_actual()` reconciliation pins
that the streamed walk matches the model's own inventory (124,439,808 at 124M).
The returned model's `cfg.dropout` is 0.0 — an inference artifact. `load_gpt2`
consumes no rng and touches no global state. Existing transformer files were
untouched (frozen); every layer already had the fieldwise constructor the loader
needed, so no Ask-first edit was required.

## Parity discipline — three gates, two scales

1. **Doll-house in the suite** (`tests/test_gpt2_weights.mojo`,
   `tests/oracles/gpt2_weights_reference.py`, V 11 T 8 C 8 L 1 H 2). The oracle
   writes a valid GPT2W file with asymmetric sentinels (per-slot base + different
   row/column coefficients) so a transposed square proj (proj.w[0,1] !=
   proj.w[1,0]) or a swapped walk slot lands a wrong number. Pins: shape/count
   reconciliation, the square-proj transpose, qkv/wte/ln_f slot values, bit-exact
   f32→f64 widening (a probe file sets wte[0,0]=0.1f32), five named header errors,
   and the end-to-end forward against the NumPy f64 reference at 1e-9. Every
   golden is frozen inline so a broken fixture writer is caught. The loaded model
   also survives a 3-token greedy `generate()` (the XV/XVI seam in-suite).
2. **Full-scale f64-vs-f64** (`examples/gpt2_parity_check.mojo`): the Mojo forward
   over "Hello, I'm a language model," matches `scripts/gpt2_reference_logits.py`
   (NumPy f64 over the SAME .bin bytes) at 1e-6, asserted BEFORE any text prints.
3. **HF-f32 agreement, ONE-TIME offline dev check** (below).

House rule honored: no statistics, no finite differences, no big files, no
network in the suite; test_seq_tasks stays excluded (#6554).

## The HF-f32 agreement check (run once, offline)

The pinned goldens are f64-vs-f64 on identical bytes (tight). Separately, once
during development, our f64 reference was checked against HuggingFace's OWN
`GPT2LMHeadModel` (float32) as a truly independent implementation.

- **Command** (throwaway env; torch/transformers are banned in the repo, so this
  was NOT added to pixi): `torch 2.12+cpu`, `transformers 5.13`,
  `GPT2LMHeadModel.from_pretrained("gpt2")`, compare its last-row logits to
  `scripts/gpt2_reference_logits.forward()` over `checkpoints/gpt2-124m.bin` for
  ids `[15496, 11, 314, 1101, 257, 3303, 2746, 11]`.
- **Observed gap**: max abs difference **6.02e-05** on the last-row logits (our
  f64 vs HF f32) — the expected cost of HF computing in float32, tighter than the
  the ~1e-3 expected for f32 vs f64. Both argmax id = 407 (" not"); our logit[407]
  −94.68738997956149 vs HF −94.68739318847656. The port is exact; the only
  difference is f32 rounding.

## safetensors provenance

- Source: `https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors`
- Size: 548 MB, 160 tensors, all F32, no `transformer.` prefix, no `lm_head`.
- **sha256**: `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`
- Never committed (checkpoints/ and *.safetensors are gitignored). The converted
  `checkpoints/gpt2-124m.bin` (498 MB) is likewise never committed.

## The MVP moment — the generated text

Prompt: `Hello, I'm a language model,`  (ids [15496, 11, 314, 1101, 257, 3303,
2746, 11]).

Parity check top-5 next tokens (by last-row logit):
`" not"` (−94.687), `" and"` (−94.768), `" I"` (−95.112), `" so"` (−95.482),
`" but"` (−95.506) — all coherent GPT-2 continuations.

`examples/gpt2_generate.mojo`, 25 new tokens each, stop token END_OF_TEXT_ID
(never hit here), seed 1337:

**Greedy (temperature 0):**

> Hello, I'm a language model, not a programming language. I'm a language model.
> I'm a language model. I'm a language model. I'm

**Nucleus (top-p 0.9, temperature 1.0):**

> Hello, I'm a language model, I know the basics...but here are some notes..."
>
> Melissa saw Gray showing a video of Alan Redge,

Both are recognizably GPT-2: the greedy run falls into GPT-2's classic
low-entropy repetition loop ("I'm a language model."), and the nucleus run
wanders into coherent, more varied prose. This is the MVP claim made good — our
Mojo code, our forward pass, OpenAI's weights, real English.

## Per-token timing (honest, no KV cache)

Measured on this CPU (scalar float64, no KV cache, seed 1337):

- Greedy: 25 tokens in **232.6 s → 9.30 s/token**.
- Nucleus: 25 tokens in **228.5 s → 9.14 s/token**.
- Whole example (both runs, load + compile): ~7:46 wall.

So a ~25-token continuation is ~4 minutes — "minutes for a sentence" on CPU. The
per-token cost is roughly flat here because the prompt is short and 25 new tokens
barely grows the O(T·C²·L) linear-layer cost that dominates at these lengths.

Every generated token re-runs the FULL forward over the whole growing context —
there is no KV cache. That is the deliberate cost this part documents as the
opening argument for the next parts (KV cache, then performance). The remedy when
wall-clock is too long is to shrink the token budget, never the model.

## Deviations from the original design

- **Walk order**: used Part XIV's LANDED order (ln1, qkv, proj, ln2, up, down),
  not an earlier draft sketch (qkv, proj, ln1, ...). Reconciled against
  the code as the precondition required.
- **Header has no separate integer version field**: the version lives in the
  magic's `v1` tag. The loader checks the family token (`GPT2W`) for "bad magic"
  and the version tag (`v1`) for "unsupported version" as two distinct named
  errors — two distinct named errors — without a redundant field.
- **lm_head / masked_bias absent** in this particular safetensors; the converter
  handles them if present (assert-tied-and-drop; skip) but here there was nothing
  to do. Documented so a future size that DOES carry them is covered.

## Mojo lessons this part

- **Binary reads use `open(path, "r").read_bytes()`** — there is no `"rb"` mode.
  Added to AGENTS.md.
- **A `comptime List[Int]` cannot be read at runtime** ("cannot materialize
  comptime value ... not ImplicitlyCopyable"). An example that pins expected token
  ids must build the list at runtime (`var expected = [...]`) or lift with
  `materialize`. Bit us in the parity example; already a known AGENTS.md note,
  re-confirmed.

## Review triage

Dual external review over `git diff main...part-16-gpt2-weights`, both read-only,
both asked to VERIFY THE PORT (walk the layout-fix list item by item, not just
read). Neither found a correctness blocker; all four Conv1D transposes (including
the square proj), the walk order against Part XIV's landed order, the buffer
skips, the pull-by-inventory, the truncation/trailing header checks, and the
exact f32→f64 widening all held under re-derivation.

- **Claude Opus 4.8 (xhigh): 0 blocker, 1 should-fix, 6 nits, all 13 checklist
  items confirmed correct.** Full text: `docs/plans/part-16-review-opus.md`
  (gitignored).
- **Codex (GPT-5.5, high): 0 blocker, 1 P3.** Full text:
  `docs/plans/part-16-review-codex.md` (gitignored).

Triage, fix/reject per finding:

- **Opus S1 — FIXED.** The converter (where every silent-garbage risk lives) had
  no in-suite automated test — the doll-house fixture writes the GPT2W file
  directly and bypassed the converter entirely. Added a `--self-test` mode to
  `scripts/convert_gpt2_weights.py`: it synthesizes a doll-house safetensors in
  HF's Conv1D convention (kernels [in, out], biases/LN 1-D, a `transformer.`
  prefix, an lm_head tied to wte, an attn.bias buffer), runs the FULL converter,
  and asserts the emitted bytes are IDENTICAL to the independently-written test
  fixture. Byte-equality pins all four transposes, the reshapes, the buffer skip,
  the tied-lm_head drop, and the prefix strip in one assertion. Runs without the
  500 MB download; `pixi run python scripts/convert_gpt2_weights.py --self-test`
  passes. (It is a manual gate — the automated `pixi run test` suite is Mojo-only
  and the converter is Python — documented alongside the other manual gates.)
- **Codex P3 / Opus N5 — FIXED.** The notes referenced the internal plan (label
  `D2`, "Deviations from plan", "the plan's sketch"); AGENTS.md bans internal-plan
  references in committed docs. Rephrased to state the reasoning directly without
  the plan labels; the docs commit message was likewise scrubbed.
- **Opus N1 — FIXED.** Three loader header branches (no newline / no header line,
  wrong token count, declared-count-vs-dims mismatch) were correct but untested.
  Added three fixture writers and three `assert_raises` cases to
  `test_header_errors_are_named`.
- **Opus N3 — FIXED.** The converter's surprise-tensor classifier used
  `endswith("attn.bias")`, which also matches `c_attn.bias`; harmless today
  (c_attn.bias is always pulled) but tightened to `endswith(".attn.bias")` so a
  future unpulled real bias would surface as a surprise, not be reclassified as a
  mask buffer.

Rejected / left as-is, with justification:

- **Opus N2 (doll-house sentinels give near-position-invariant logits, inter-row
  deltas ~1.3e-10).** The e2e forward test pins weight PLACEMENT strongly and the
  positional/causal path weakly, but that path is finite-difference- and
  oracle-tested in `test_gpt.mojo` / `test_block.mojo` (the forward is frozen this
  part), and the 124M parity gate exercises a real 8-position causal context.
  Adding a nonlinear sentinel would churn every frozen golden for coverage that
  already exists a layer down. Left.
- **Opus N4 (loader header parse is single-space/newline-brittle).** The GPT2W
  format is ours and we control both ends (the converter and the fixture writer
  both emit single-space-separated fields + one newline); a malformed file still
  fails safely via the token-count / dim / count / length checks. A tolerant
  whitespace parser would add surface for no real input. Left.
- **Opus N6 (surprise tensors → stderr warning, not a hard error).** Deliberate:
  an unexpected-but-unused extra tensor should be VISIBLE without aborting a
  conversion that is otherwise correct (the parameters we need were all pulled by
  name). A missing REQUIRED tensor already raises. Left.
