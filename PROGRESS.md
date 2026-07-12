# Progress

Build status per part of the from-scratch Transformer. "Test command" is what
proves the part green on a fresh checkout (`pixi install` first).

**Part XIX flipped the default gate.** `pixi run test` is now the canonical gate:
it runs the whole suite EXCEPT one #6554-stalling lab file BY DEFAULT, printing a
loud `SKIPPED (Mojo #6554)` line — which is what "green" has always meant. Before
XIX the default `test` *ran* the slow file and `test-fast` skipped it; now
`test-fast` is a retained alias of `test`, and `test-full` (`RUN_SLOW=1`) is the
opt-in that also runs the slow file (the toolchain-upgrade check). Rows dated
before this flip list `pixi run test-fast`, which still works. From XIX on, a part
is mergeable when the suite is green, `pixi run gauntlet` (the 124M release gate)
is green, and `fmt-check` passes; see [AGENTS.md](AGENTS.md).

| Part | Title | Status | Test command | Date |
|------|-------|--------|--------------|------|
| I | Foundations & config | ✅ green | `pixi run test-fast` | 2026-07-03 |
| II | Vocabulary | ✅ green | `pixi run test-fast` | 2026-07-03 |
| III | Tensors & ops | ✅ green | `pixi run test-fast` | 2026-07-03 |
| IV | Utilities (rng, math) | ✅ green (`Rng` + float draws, timing) | `pixi run test-fast` | 2026-07-03 |
| V | Tokenization | ✅ green | `pixi run test-fast` | 2026-07-03 |
| VI | Dataset pipeline | ✅ green | `pixi run test-fast` | 2026-07-03 |
| VII | Tiny bigram LM | ✅ green | `pixi run test-fast` | 2026-07-03 |
| VIII | Architecture family | ✅ green (preset + exact param count + comptime pin) | `pixi run test-fast` | 2026-07-03 |
| IX | NN building blocks | ✅ green (Parameter, Linear, Embedding, LayerNorm, GELU, Dropout, MLP — forward only) | `pixi run test-fast` | 2026-07-04 |
| X | Attention | ✅ green (additive masks, scaled-dot-product core self+cross, fused-QKV multi-head — forward only) | `pixi run test-fast` | 2026-07-04 |
| XI | Backpropagation by hand | ✅ green (every layer's backward finite-difference-checked; explicit per-layer caches; grads accumulate) | `pixi run test-fast` | 2026-07-04 |
| XII | Encoder-decoder lab | ✅ green (cross-attention + pre-LN blocks assembled into a seq2seq model in `src/llm/lab/`; trains copy/reverse to exact-match with a memory ablation; all lab code quarantined off the main line) | `pixi run test-fast` | 2026-07-04 |
| XIII | The GPT-2 model | ✅ green (pre-LN `TransformerBlock` + weight-tied `GPT`; three-site dropout; residual-init scaling; walk reconciles with the 124,439,808 formula) | `pixi run test-fast` | 2026-07-04 |
| XIV | Training | ✅ green (AdamW with decoupled decay + selective weight decay; warmup/cosine schedule; global-norm clipping; bit-exact checkpoints with a proven resume gate; `train_gpt` over `BatchLoader`; the parameter walk promoted to a load-bearing registry) | `pixi run test-fast` | 2026-07-05 |
| XV | Generation | ✅ green (top-k + top-p distribution filters in probability space; one `SamplerConfig` policy — greedy/temperature/top-k/top-p — with temperature 0 = argmax drawing zero rng; the autoregressive `generate` loop with sliding-window crop and append-then-halt stop tokens; LCG-replay sampled goldens; a memorize-then-speak capstone; the Shakespeare checkpoint speaking four ways) | `pixi run test-fast` | 2026-07-05 |
| XVI | Loading real GPT-2 weights (the MVP) | ✅ green (offline safetensors→`GPT2W v1` converter with every Conv1D transpose + buffer skip in one place; native `load_gpt2` builds the GPT fieldwise from the f32 payload, exact f32→f64 widening, named header validation; doll-house sentinel parity in-suite + f64 goldens at 124M; **the from-scratch Mojo forward, fed OpenAI's real weights, generates coherent English** — HF-f32 agreement 6e-5) | `pixi run test-fast` | 2026-07-05 |
| XVII | The KV cache | ✅ green (per-layer `KVCache` of preallocated `[context_length, C]` key/value buffers; additive `step` methods on attention/block/gpt that feed ONE token and reuse the cached past, reusing the frozen `scaled_dot_product_attention` with a zeros `[1, t]` mask; `matmul_transpose_b` for the tied head with no per-token transpose; `generate_cached` twin of `generate` with an up-front overflow raise; **step-vs-forward logits BIT-IDENTICAL at every prefix**, generation + rng-stream parity, and the 124M greedy text character-identical to Part XVI at ~8.3× (0.78 vs 6.49 s/token) — the ALGORITHM fix, arithmetic still scalar f64) | `pixi run test-fast` | 2026-07-11 |
| XVIII | Performance (SIMD + threading) | ✅ green (a private multi-accumulator SIMD `_simd_dot` under `matvec`/`matmul_transpose_b` — the one Class B reassociation, 1e-12-relative tested; SIMD-over-columns `@`, `matmul_transpose_a` for the backward, and `std.algorithm` threading of both reduction kernels — all Class A, bit-identical, exact-equality + determinism tested; call-site retrofits onto `matmul_transpose_b`/`_a` delete a transpose alloc per call; memcpy slice/concat hygiene; **greedy 124M decode 0.784 → 0.0249 s/token (31.5×), training step 367.8 → 20.2 ms (18.2×), text character-identical** — the ARITHMETIC fix, whole existing suite green UNCHANGED, XVII exact parity untouched) | `pixi run test-fast` | 2026-07-12 |
| XIX | The gauntlet: systematic validation & CI | ✅ green (two validation tiers formalized in pixi tasks — Tier 1 `pixi run test`: hermetic suite that now SKIPS the #6554 lab file BY DEFAULT with a loud SKIPPED line (**a behaviour flip of the old opt-in**), `test-full` the toolchain-upgrade opt-in, `build-examples` compile-checking every example+benchmark in CI; Tier 2 `pixi run gauntlet`: a 16-prompt 124M harness spanning English/contractions/accents/CJK/emoji/code/digits/URL/whitespace/Unicode-separators(NBSP,U+2028/9)/newlines/single-token/Finnish and the **1000- and exactly-1024-token context boundary**, checking tokenization+argmax+top-5 EXACT and probe logits+mean NLL @ 1e-6 against frozen NumPy-f64 goldens (provenance-checked against the `.bin`'s sha256 before the run), plus `generate` vs `generate_cached` token parity on a short subset; golden-lifecycle + merge-gate doctrine written into AGENTS.md; **gauntlet 16/16 green in ~41 s, NO model change, no bug surfaced**) | `pixi run test` + `pixi run gauntlet` | 2026-07-12 |

## Notes

- **Foundation restore (Parts II–III) landed on branch `foundation-restore`.**
  `config` (GPTConfig, TrainingConfig), `vocab` (toy whitespace Vocabulary), the
  `tensor` package (Tensor2D/3D, elementwise/matmul/softmax/cross-entropy ops,
  argmax, Xavier init), the `Rng` float draws (uniform/normal), and benchmark
  timing helpers. Every chapter deviation is recorded in
  [notes/part-07-notes.md](notes/part-07-notes.md). This unblocks Part VII, which
  needs float tensors for the bigram table.
- **Part V deliverables:** `CharTokenizer`, byte-level `BPETokenizer` (merge loop
  + didactic trainer), `GPT2Tokenizer` (GPT-2 vocab/merges, regex pre-tokenizer,
  byte↔unicode table), save/load for all three, and GPT-2 parity proven against a
  vendored OpenAI reference encoder. See [notes/part-05-notes.md](notes/part-05-notes.md).
- **Part VI deliverables:** a seeded `Rng` (LCG), and the tokenizer-agnostic data
  layer — `load_text`, `TokenDataset` + `train_val_split`, flat `[B, T]`
  `TokenBatch`, and `BatchLoader` (sliding windows, seeded epoch shuffle,
  remainder-drop) with `overfit_batch`. Tiny Shakespeare is committed for offline
  tests. See [notes/part-06-notes.md](notes/part-06-notes.md).

- **Part VII deliverables:** `BigramLM` (a single `[V, V]` logits table, filled
  either by `from_counts` with Laplace smoothing or trained from zeros/random),
  `loss_and_grad` with the fused `p − onehot` scatter-add, `perplexity`,
  `sgd_step`, the single-batch `train_bigram`, and `sample_categorical`. New
  `models/` package + scope. Trains char-level on tiny Shakespeare
  (`examples/bigram_shakespeare.mojo`); the `q → u` bigram is pinned. See
  [notes/part-07-notes.md](notes/part-07-notes.md).

- **Part VIII deliverables:** `GPTConfig.gpt2_124m()` (the reference GPT-2 small
  preset) and `GPTConfig.parameter_count()` (the exact GPT-2-layout total,
  124,439,808, derived in the docstring), pinned at compile time by
  `check_gpt2_contract()`. Plus two behavior-frozen metaprogramming cleanups:
  derived `comptime` constants replacing the magic mantissa literal in
  `utils/random.mojo`, and the GPT-2 byte↔unicode table bound at compile time in
  `tokenizer/gpt2.mojo` (`materialize`d at use). No layers, traits, or new
  packages — those are Part IX. See [notes/part-08-notes.md](notes/part-08-notes.md).

- **Part IX deliverables:** the `nn/` package — every layer the Transformer is
  assembled from except attention, forward passes only (backward comes later).
  `Parameter` (value + zeros grad + `zero_grad`), `Linear` (`x @ W^T + b`,
  `[out, in]` weight), `Embedding` (range-checked gather, one struct for token
  and positional use), `LayerNorm` (biased variance, eps 1e-5), `gelu`/`gelu_rows`
  (tanh approximation), `dropout` (inverted, mode as an argument, eval consumes no
  rng), and `MLP` (up → gelu → down). Layer factories draw from GPT-2's
  `normal(0, 0.02)`; `SQRT_2_OVER_PI` is bound at compile time. Goldens are frozen
  from `tests/oracles/nn_reference.py`. See [notes/part-09-notes.md](notes/part-09-notes.md).

- **Part X deliverables:** the `transformer/` package — attention, forward passes
  only. Two column ops at the tensor layer (`slice_cols`, `concat_cols`) for the
  head split/merge. Additive masks (`MASKED_SCORE = -1e9` finite on purpose,
  `no_mask`, `causal_mask`, `key_padding_mask`; compose by tensor add).
  `scaled_dot_product_attention` — one core serving self- AND cross-attention
  (separate q vs k/v lengths, tested with `T_q != T_k`), returning
  `AttentionResult{output, weights}` so causality is proven on the weights
  directly; order pinned scores → `1/sqrt(d_head)` → mask → softmax → `@v`.
  `MultiHeadAttention` with GPT-2's fused QKV (`Linear(C→3C)` + `Linear(C→C)`),
  contiguous head split `D=C/H`, parameter cost `4C^2+4C`. Goldens frozen from
  `tests/oracles/attention_reference.py`. See
  [notes/part-10-notes.md](notes/part-10-notes.md).

- **Part XI deliverables:** a hand-derived, finite-difference-checked backward
  pass for every layer Parts IX–X built — additive, no `forward` signature
  changed, no autograd. Each layer gains `forward_cached(x)` returning an
  explicit `<Layer>Cache` (holding exactly what backward needs, named in a
  comment) and `backward(cache, d_out)` that ACCUMULATES (`+=`) into
  `Parameter.grad` — the property later weight tying depends on, pinned by a
  per-layer exact-doubling test. Tensor layer: `softmax_rows_backward`
  (`dS = W ⊙ (dW − rowsum(dW ⊙ W))`, on the output W), and the batched
  `cross_entropy_rows` + `cross_entropy_rows_backward` (`(softmax − onehot)/N`).
  `nn/`: Linear, Embedding (scatter-add, repeated ids accumulate), LayerNorm
  (three-term `dx = r(a − mean(a) − x̂·mean(a⊙x̂))`, derived in the docstring),
  GELU (`gelu_derivative` + rows VJP), Dropout (cached mask), MLP.
  `transformer/`: the attention core backward (dV/dW/dS then dQ, dK with
  `1/sqrt(D)` applied once) and MHA backward reversing the fused-QKV plumbing.
  Every `d_input` and parameter grad is finite-difference checked (central diff
  h=1e-5, mixed tolerance); the chain test trains a real Embedding→LayerNorm→MLP
  stack with strictly decreasing loss. See
  [notes/part-11-notes.md](notes/part-11-notes.md).

- **Part XII deliverables:** a small encoder-decoder Transformer ASSEMBLED from
  the Parts IX–XI layers, quarantined in a new `src/llm/lab/` package (the main
  line is decoder-only, so none of this joins it; nothing under `tensor/`, `nn/`,
  `transformer/`, `training/`, `generation/` changed). `CrossMultiHeadAttention`
  (separate `q` Linear(C→C) + FUSED `kv` Linear(C→2C) + `proj`, cost 4C²+4C,
  reusing the Part X core and its backward; backward returns {d_x, d_memory}).
  Pre-LN `EncoderBlock`/`DecoderBlock` (x + sublayer(ln(x)); the residual backward
  `d_x = d_out + branch_backward(d_out)` is the one new gradient rule). `EncDec`
  seq2seq model: separate src/tgt token + positional embeddings, encoder stack +
  final LN → memory, decoder stack + final LN, untied head; teacher forcing with
  BOS; explicit `zero_grad`/`apply_sgd` enumerating every Parameter;
  `greedy_decode`. Toy copy/reverse tasks. The capstone trains to EXACT greedy
  decode (copy + reverse overfit) and a corrupted-memory ablation collapses it,
  proving cross-attention is load-bearing; held-out generalization + the
  anti-diagonal alignment map are in `examples/encdec_reverse.mojo`. NumPy oracle
  `tests/oracles/encdec_reference.py`. `scripts/test_all.sh` now precompiles the
  `llm` package (`build/llm.mojopkg`) so the suite runs in minutes not tens of
  minutes. See [notes/part-12-notes.md](notes/part-12-notes.md).

- **Part XIII deliverables:** the decoder-only GPT, ASSEMBLED on the main line
  from the Parts IX–XI layers (nothing under `tensor/`, `nn/`, `lab/`,
  `training/`, `generation/`, or `config.mojo` changed). `TransformerBlock`
  (GPT-2's pre-LN block, self-attention only: `x + dropout(attn(ln1(x)))` then
  `a + dropout(mlp(ln2(a)))`). `GPT` = token + learned positional embeddings → L
  blocks under a shared `causal_mask` → final LayerNorm → WEIGHT-TIED head
  (`logits = h @ wte^T`, no bias). The tied token table gets gradient through two
  paths — the head matmul (every row) and the embedding gather (used rows) —
  combined into one delta and added once so the grad doubles bit-exactly.
  GPT-2's three dropout sites, all on `cfg.dropout`: embedding, attention-weight
  (an additive train path over the frozen attention core:
  `scaled_dot_product_attention_train` + `forward_cached_train`/`backward_train`),
  and residual (on each sublayer branch, never the skip). Inference `forward`
  takes no rng — dropout is unrepresentable there. `init_random` applies GPT-2's
  residual-init scaling (proj/down weights ×1/√(2L)). `parameter_count_actual`
  walks every Parameter (wte once) and reconciles with
  `GPTConfig.parameter_count()`; `examples/gpt2_inventory.mojo` builds the real
  124M preset and asserts the walked total is 124,439,808. NumPy oracle
  `tests/oracles/gpt_reference.py`. See
  [notes/part-13-notes.md](notes/part-13-notes.md).

- **Part XIV deliverables:** the training machinery that turns "learns" into
  "trains" (nothing under `tensor/`, `lab/`, `generation/`, `models/`, or
  `config.mojo` changed). Per-Parameter optimizer math moved to its
  layering-legal home `nn/optim.mojo` (`sgd_update`; `apply_sgd` refactored to
  delegate, behavior-frozen; `adamw_update` — decoupled decay, bias correction
  from t=1, NumPy-oracle-checked). The model's fixed traversal order is promoted
  to a **parameter registry**: additive `GPT` walk methods (`parameter_shapes`,
  `parameter_decay_flags`, `grad_norm`, `scale_grads`, `export/import_parameters`,
  `export_gradients`, `apply_adamw`), all visiting the same parameters in the same
  order (wte once), with the GPT-family selective-decay partition (matrices decay,
  biases and LayerNorm vectors do not). `training/`: a pure warmup+cosine
  `lr_at`, `AdamWConfig`/`ScheduleConfig` (config.mojo untouched — TrainingConfig
  is fieldwise-init), global-norm `clip_grad_norm`, versioned bit-exact
  checkpoints (`save_checkpoint`/`load_checkpoint`, Float64 stored as its hex bit
  pattern), and `train_gpt` (AdamW + schedule + clip over `BatchLoader`, with
  `estimate_loss` and resumable segments). The capstone: overfit-one-batch through
  the real trainer, plus `examples/train_gpt_shakespeare.mojo` (char-level tiny
  Shakespeare, checkpoint + load-and-resume). NumPy oracle
  `tests/oracles/adamw_reference.py`. See
  [notes/part-14-notes.md](notes/part-14-notes.md).

- **Part XV deliverables:** generation, with ZERO new math below the top layer —
  pure assembly in `generation/` over the existing forward, temperature softmax,
  argmax, and inverse-CDF sampler (nothing under `tensor/`, `nn/`, `transformer/`,
  `training/`, `config.mojo`, `models/`, or `lab/` changed). `sampler.mojo` grows
  the two distribution filters (`filter_top_k`, `filter_top_p`, in probability
  space, each renormalizing, sharing one merge-sort tie rule), a `SamplerConfig`
  policy (temperature/top-k/top-p, with `validate` and greedy/standard presets),
  and `sample_next` (the single entry point: temperature 0 = argmax drawing zero
  rng; else softmax → top-k → top-p → categorical, exactly one draw). A new
  `generate.mojo` holds the autoregressive loop (sliding-window context crop,
  append-then-halt stop tokens, ids-in/ids-out, bound to nothing but `GPT`). The
  capstone overfits a tiny GPT and greedy-generates the memorized continuation
  exactly; `examples/generate_shakespeare.mojo` makes the Part XIV checkpoint
  speak under four policies. Python oracle
  `tests/oracles/sampling_reference.py` (filter goldens + LCG-replay exact
  sampled ids). See [notes/part-15-notes.md](notes/part-15-notes.md).

### Generation (Part XV)

| File | Covers |
|------|--------|
| `tests/test_sampling_filters.mojo` | top-k oracle goldens; k=0/k≥n exact identity; k=1 one-hot at argmax; k-th boundary tie keeps the lower index; renormalized sum (1e-12); k<0/empty raises. top-p goldens with a tie case; p=1.0 exact identity (no cumsum path); tiny p keeps exactly the argmax; renormalized sum; p≤0/p>1/empty raises. Composition golden pinning top-k-then-top-p (the two orders disagree by construction) |
| `tests/test_sample_next.mojo` | greedy (T=0) equals argmax with `rng.state` bit-unchanged (zero draws, across many calls); sampled advances the state exactly once per call; LCG-replay goldens (fixed seed + logit row → exact oracle-predicted ids) for three seeds; top_k=1 forces the argmax regardless of seed; `validate` raises per named field; presets validate |
| `tests/test_generate.mojo` | length == budget with no stop; max_new_tokens 0 → empty; negative raises; empty prompt raises; greedy twice → identical ids and untouched `rng.state`; equal seeds → identical sampled runs; stop token appended-then-halt (shorter than budget, a prefix of the unstopped run); a stop id in the PROMPT does not halt; empty stop list runs to budget; context-crop equivalence against a manual hand-cropped forward; the memorize-then-speak capstone (plain-SGD overfit → exact greedy continuation) |

### Loading real GPT-2 weights (Part XVI)

| File | Covers |
|------|--------|
| `tests/test_gpt2_weights.mojo` | doll-house (V11 T8 C8 L1 H2) load reconciles dims + `parameter_count_actual`; asymmetric sentinels pin the SQUARE proj kernel is not transposed (proj.w[0,1]≠proj.w[1,0]) and each of the 16 tensors landed in its named walk slot; a probe pins float32(0.1) widens to its exact f64 bit pattern; five named header errors (bad magic, wrong version, bad dims, truncated, trailing); file→loader→forward matches the NumPy f64 reference at 1e-9; a loaded model generates 3 tokens greedily. Fixture + goldens from `tests/oracles/gpt2_weights_reference.py`, frozen inline |

### The KV cache (Part XVII)

| File | Covers |
|------|--------|
| `tests/test_kv_cache.mojo` | `fresh` shapes/length/capacity; `check_compatible` named raises (wrong layer count, width, capacity); `GPT.step` fills to exactly capacity then raises NAMED when full (failed step does not advance length); **the centerpiece — step-vs-forward EXACT logits parity at every prefix on a TWO-layer doll-house** (`assert_equal`, all V columns, no tolerance); `reset` replays bit-identically; a bad token id raises and leaves `length` trustworthy |
| `tests/test_generate.mojo` (extended) | `generate_cached == generate` token-for-token AND `rng.state` identical (stream parity) — greedy and a temperature-0.9 top-k+top-p config on a two-layer doll-house; up-front overflow raise and the ==context_length success; contract parity (empty prompt, negative budget, zero-budget no-op with rng untouched, append-then-halt, prompt stop-ids ignored) |
| `tests/test_matmul.mojo` (extended) | `matmul_transpose_b` hand-computed; equals `matmul(a, transpose(b))` on seeded shapes incl. a `[1, k]` row (see Part XVIII — Class B, now 1e-12 relative); contraction-width-mismatch raise |
| `tests/test_slicing.mojo` (extended) | `slice_rows` hand-checked, prefix-as-cache-view, full-height identity, bad-range raises |

### Performance — SIMD + threading (Part XVIII)

| File | Covers |
|------|--------|
| `tests/test_matmul.mojo` (extended) | **Class B** `matmul_transpose_b` and `matvec` vs the scalar spelling at **1e-12 RELATIVE** on real dims (k=768, 3072) and ragged tails (5, 7, 13, 769, 3073); **Class A** `@` SIMD-over-columns **bit-identical** to a scalar ikj reference (`assert_equal`, ragged widths) and `matmul_transpose_a` **bit-identical** to `transpose(a) @ b`; threaded-path correctness + two-call **determinism** (exact) for both `matmul_transpose_b` and `@` |
| `tests/test_generate.mojo` (extended) | `generate_cached` **determinism** after threading — same seed, two runs, identical greedy AND nucleus tokens, on a model sized so the tied head crosses the threading threshold |
| `benchmarks/bench_kernels.mojo` (new) | decode-shape kernel table (measurement only, no assertions) — the before/after evidence per stage |

Everything else is implementation-only under stable contracts: the SIMD/threaded kernels slot in under `matmul`/`matmul_transpose_b`/`matvec`/`@`, and the whole pre-existing suite (XVII exact parity, gradient-doubling, 124M 1e-6 goldens) passes UNCHANGED. Gated stages: threading (D4) **in**; fused decode attention (D5) **skipped on profile evidence** — KV copies subdominant and the memcpy hygiene made them 5–9× cheaper; `matmul_transpose_a` (D6) **in** — the transpose alloc was 28–65% of each backward product.

### Training (Part XIV)

| File | Covers |
|------|--------|
| `tests/test_adamw.mojo` | `adamw_update` vs the NumPy oracle over several steps (decay on and off, value + m + v); the hand-computed step 1 (the 1−β^1 bias correction cancels the 1−β that formed the moment, so mhat=g, vhat=g²); the decoupled-decay pin (g=0 ⇒ moments stay exactly zero, value still shrinks by 1−lr·wd); t<1 and shape-mismatch raises; determinism |
| `tests/test_schedule.mojo` | hand-computed goldens (step 0→0, mid-warmup, warmup end = peak, cosine midpoint, max_steps = min_lr, past-the-end clamped); monotone non-increasing after warmup; degenerate warmup=0; `ScheduleConfig.validate` and `lr_at` guards |
| `tests/test_gpt_optimizer.mojo` | `parameter_shapes` reconciles with `parameter_count_actual` (wte once); the decay-partition inventory (per block 4 decayed matrices + 8 undecayed; embeddings decay, ln_f not); `apply_adamw` moves every Parameter and rejects a mis-sized m/v; walk stability; the against-oracle 2-step run — `apply_adamw` matches a flat reference driving the oracle-verified `adamw_update` over the model's published walk metadata |
| `tests/test_clipping.mojo` | hand-computed global norm across three different parameters (3,4,12 → 13, which no per-tensor norm reports); below-threshold exact no-op; above-threshold post-clip norm == clip with direction preserved; zero-grad edge (no 0/0); `scale_grads` doubles every gradient |
| `tests/test_checkpoint.mojo` | the resume gate (k + checkpoint + fresh model + (n−k) == n straight steps, bit-identical); bit-exact save/load round-trip (params, m, v, t, rng); the edge-value hex round-trip; and the bad-magic / shape / parameter-count / truncated guards |
| `tests/test_gpt_train_loop.mojo` | overfit-one-batch through the real `train_gpt` crushes the loss far below log V (seeded, deterministic); a dropout=0.1 run still ends below its start; a segmented run reproduces a straight run bit-for-bit; `estimate_loss` matches a hand-averaged dropout-free loss and preserves the cursor; TrainReport history lengths; the AdamW preset (β₂=0.95) pinned |

### The GPT-2 model (Part XIII)

| File | Covers |
|------|--------|
| `tests/test_attention_train.mojo` | attention-weight dropout train path: eval/p=0 == the cached path (outputs + all grads) and consume no rng (twin-generator); dropped entries 0, survivors ×inv_keep; dq/dk/dv finite-diff through the dropped core (re-seeded-rng convention); MHA `backward_train` grads double exactly |
| `tests/test_block.mojo` | pre-LN forward oracle golden (post-LN / LN-on-sum fails it), causality under `causal_mask`, `forward_cached(eval) == forward` (no rng), finite-diff d_x + all 12 parameter grads, exact doubling, residual-dropout placement (skip preserved exactly at p=0.9; branch demonstrably dropped) |
| `tests/test_gpt.mojo` | logits [T,V], named length errors (T=0, T>ctx), tied head == manual h@wte^T (no bias), tiny-model forward golden, model causality, positions used, init loss≈log V, same-seed determinism, residual-init std bands (proj/down 0.02/√(2L), qkv/up 0.02), walk==formula on 2 tiny configs (wte once), zero_grad/apply_sgd reach every Parameter |
| `tests/test_gpt_backward.mojo` | the capstone: model-level finite-diff of the wte-table grad (both tied paths summing), plus wpe, a mid-block qkv weight, and ln_f; exact model-level doubling (tied wte included); mismatched-cache raise |
| `tests/test_gpt_training.mojo` | overfit-one-batch smoke: dropout=0 deterministic, loss below log V and decreasing across checkpoints; dropout=0.1/training=True loss below init |

### Encoder-decoder lab (Part XII)

| File | Covers |
|------|--------|
| `tests/test_seq_tasks.mojo` | copy/reverse correctness, seeded determinism, values in [0,V_data), BOS one past the alphabet, teacher-forcing shift hand-pinned, unique-source distinctness |
| `tests/test_cross_attention.mojo` | cross-MHA forward oracle (T_q≠T_k), shape contract, param count 4C²+4C, init determinism, invalid-config raises, single-head=core+proj, finite-diff d_x/d_memory/all six param grads, exact doubling |
| `tests/test_encoder_block.mojo` | pre-LN forward oracle (a post-LN wiring fails it), zeroed-sublayer identity (skip carries x), finite-diff d_x + all 12 parameter grads |
| `tests/test_decoder_block.mojo` | pre-LN forward oracle (causal+cross), causality (perturb x@j ⇒ rows<j unchanged), memory-is-read, finite-diff d_x/d_memory/all 20 parameter grads |
| `tests/test_encdec_model.mojo` | logits [T,V], init loss≈log V, model causality in tgt_in, source-token-embedding grad finite-diff (longest path), n_dec=2 d_memory summing, zero_grad/apply_sgd touch every Parameter |
| `tests/test_encdec_training.mojo` | capstone: copy + reverse overfit to exact greedy-decode, corrupted-memory ablation collapses reverse exact-match, loss far below log V |

## Test suites (Parts II–X)

| File | Covers |
|------|--------|
| `tests/test_smoke.mojo` | toolchain + `-I src` import path |
| `tests/test_config.mojo` | GPTConfig/TrainingConfig validate (both paths), d_head, param counts, Writable |
| `tests/test_vocab.mojo` | encode/decode round trip, add idempotence, unknown/out-of-range raises |
| `tests/test_tensor2d.mojo` | shape/offset, set-get, ones/zeros/full, checked access, from_rows |
| `tests/test_tensor3d.mojo` | nested row-major offset layout, set-get, checked access |
| `tests/test_elementwise.mojo` | add (+ mismatch raise), scale, transpose round trip |
| `tests/test_matmul.mojo` | hand-computed matmul, ijk/ikj agreement, matvec, mismatch raises |
| `tests/test_softmax.mojo` | rows sum to 1, stability at 1000-magnitude logits, temperature limits |
| `tests/test_cross_entropy.mojo` | uniform → log V, grad sums to 0, logsumexp stability |
| `tests/test_grad_check.mojo` | finite-difference check of cross_entropy_grad |
| `tests/test_finite_difference_step.mojo` | h-selection study (x³ truncation error = h²) |
| `tests/test_argmax.mojo` | max index, deliberate first-wins tie |
| `tests/test_timing.mojo` | median (odd/even), sort-in-place, hand-computed GFLOP/s |
| `tests/test_bigram.mojo` | zeros→log V, hand-computed count model, finite-diff gradient, grad rows sum to 0, perplexity=exp(loss) |
| `tests/test_bigram_training.mojo` | monotone decrease, deterministic-batch overfit, convergence to count optimum, training determinism, tiny-Shakespeare loss drop |
| `tests/test_sampler.mojo` | degenerate/one-hot, seed determinism, support-respect, invalid-probs raises, q→u plausibility pin |
| `tests/test_char_tokenizer.mojo` | codepoint vocab, round trips, save/load, errors |
| `tests/test_bpe_core.mojo` | merge loop, rank order, trainer (hand-computed), save/load |
| `tests/test_gpt2_tokenizer.mojo` | vocab size 50257, byte↔unicode bijection, oracle parity, goldens, save/load |
| `tests/test_rng.mojo` | LCG goldens, same-seed determinism, `next_below` range, shuffle permutation, float draws (uniform/normal), Xavier shape/determinism/no-NaN |
| `tests/test_dataset.mojo` | train/val split arithmetic + partition, corpus load + missing-file error |
| `tests/test_token_batch.mojo` | flat `[B, T]` layout, shape/bounds checks |
| `tests/test_batch_loader.mojo` | window shapes, shift-by-one, seeded epochs, coverage, remainder drop, end-to-end |
| `tests/test_parameter.mojo` | grad zeros with value's shape, zero_grad clears grad only, value round trip |
| `tests/test_linear.mojo` | hand-computed forward (oracle), bias per-row broadcast, shape-mismatch raise, init_random determinism + zero bias |
| `tests/test_embedding.mojo` | gather returns table rows, repeated/positional ids, negative and ≥V raises, init_random shape/determinism |
| `tests/test_layernorm.mojo` | biased-variance 3x4 oracle golden (rejects unbiased), mean~0/std~1, per-column weight/bias, constant row → bias |
| `tests/test_gelu.mojo` | tanh-approx scalar goldens (reject erf), gelu(0)=0, asymptotes, gelu_rows elementwise |
| `tests/test_dropout.mojo` | eval identity + rng untouched (twin generator), p=0 no-draw, out-of-range raise, seed determinism, survivors 0-or-scaled, keep-rate band |
| `tests/test_mlp.mojo` | composition oracle golden, equals manual up/gelu/down, hidden width from constructor, shape contract |
| `tests/test_slicing.mojo` | slice_cols hand-checked + full-width identity, split→concat round trip, widths add, bad-range/empty-list/row-mismatch raises |
| `tests/test_masks.mojo` | causal_mask hand-checked, key_padding blocks False columns, causal+padding composition stays blocked, no_mask zeros |
| `tests/test_attention_core.mojo` | cross-shaped oracle (T_q≠T_k), hand-worked 2×2, rows sum to 1, causal weights 0 above diagonal, identical-v pass-through, diagonal-only one-hot, fully-blocked finite (tied + non-tied), 1/sqrt(d_head) scale, scale-before-mask order, shape-mismatch raises |
| `tests/test_multihead_attention.mojo` | shape contract, param count 4C²+4C reconcile, init determinism, invalid-config raises, single-head equivalence, contiguous-split forward oracle (H=2), causal row-0 locality |

### Backward (Part XI)

| File | Covers |
|------|--------|
| `tests/test_softmax_backward.mojo` | row-Jacobian VJP finite-diff, uniform-row mean-subtraction form, shape-mismatch raise |
| `tests/test_cross_entropy_rows.mojo` | agrees with mean of `cross_entropy_one`, backward finite-diff, rows sum to 0, 1/N mean factor (double rows → halve grad), bad-target/length raises |
| `tests/test_linear_backward.mojo` | dx/dW/db finite-diff (three loops), exact accumulation doubling, zero_grad reset |
| `tests/test_embedding_backward.mojo` | touched-row finite-diff, repeated-id accumulation, untouched rows exactly zero, cross-call doubling |
| `tests/test_layernorm_backward.mojo` | dx/dγ/dβ finite-diff, dx⊥ones (exact) and ⊥x̂ (up to eps) projection property, exact doubling |
| `tests/test_gelu_backward.mojo` | scalar-derivative finite-diff across a grid, 1/0 asymptotes, rows VJP elementwise, shape-mismatch raise |
| `tests/test_dropout_backward.mojo` | forward uses returned mask, cached-mask VJP finite-diff (exact), inv_keep scaling, eval/p=0 identity, eval draws no rng, shape-mismatch raise |
| `tests/test_mlp_backward.mojo` | dx and all four parameter grads finite-diff end to end |
| `tests/test_attention_backward.mojo` | core dq/dk/dv finite-diff under no_mask/causal/cross-shape (T_q≠T_k), no-gradient-leak into a fully-blocked key, MHA dx + qkv/proj parameter grads |
| `tests/test_backprop_chain.mojo` | Embedding→LayerNorm→MLP→cross_entropy_rows: embedding-table grad finite-diff end to end, 20 SGD steps strictly decreasing loss |
