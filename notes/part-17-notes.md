# Part XVII notes — the KV cache

Part XVI made the model COHERENT; this part makes it FAST enough to use, and does
it without touching a single flop of arithmetic. The whole part is one algorithmic
change — recompute-everything becomes cache-and-reuse — held to an exacting bar:
the cached decode step must produce logits **bit-identical** to the uncached batch
forward, not merely close.

## What shipped

- **`tensor/ops.mojo`** — two new ops:
  - `matmul_transpose_b(a, b)` = `a @ b^T` computed directly as
    `c[i,j] = sum_k a[i,k]*b[j,k]`, k ascending — the same accumulation order as
    `matmul(a, transpose(b))`, so the two spellings are bit-identical (a test pins
    exact equality on seeded shapes incl. a `[1, k]` row). It exists so the tied
    head can score one decode row against the `[V, C]` embedding table without
    allocating and copying the ~309 MB `[C, V]` transpose per token.
  - `slice_rows(x, lo, hi)` — the row mirror of `slice_cols`, for viewing the
    filled prefix `[0, length)` of a capacity-sized cache buffer as `[t, C]`.
    (`slice_rows` did NOT already exist — checked `test_slicing.mojo` and
    `ops.mojo` first; added it with goldens and range raises.)
- **`transformer/kv_cache.mojo`** (new) — `KVCache`: per-layer `[capacity, C]` key
  and value buffers preallocated to zeros, a shared `length`, a fixed `capacity ==
  context_length`. `fresh` (from a validated config), `reset` (length → 0, buffers
  untouched), `check_compatible` (named raises on layer-count / width / capacity
  mismatch). Rows are stored PRE-head-split (the full fused-qkv k-third and
  v-third), matching the batch path's `k_all`/`v_all` — which is what makes the
  bit-parity argument mechanical rather than a fresh proof.
- **Three additive `step` methods** (`attention.mojo`, `block.mojo`, `gpt.mojo`) —
  existing `forward`/`forward_cached`/`backward` paths byte-for-byte untouched.
  `MultiHeadAttention.step` is `forward` one row wide: fused qkv on the row, write
  the K/V thirds into cache row `pos`, view the valid prefix, and per head call the
  FROZEN `scaled_dot_product_attention(q_h [1,D], k_h [t,D], v_h [t,D],
  zeros_2d(1,t))`. `TransformerBlock.step` is the same pre-LN wiring one row wide.
  `GPT.step(token_id, mut cache)` embeds token+position, threads the row through
  every block's cache buffers, applies the tied head via `matmul_transpose_b`, and
  bumps `length` once after all layers wrote row `pos`. It takes NO rng.
- **`generation/generate.mojo`** — `generate_cached`, the KV-cached twin of
  `generate`: identical contract, internal cache, up-front overflow raise. Primes
  the prompt token-by-token through `step`, then feeds only each emitted token.
- **`examples/gpt2_generate_cached.mojo`** (new) — the before/after money demo
  against the real 124M weights. Part XVI's `gpt2_generate.mojo` is left untouched
  as the honest "before" exhibit.

## Why bit-exactness is achievable (and therefore demanded)

Every stage of the eval forward is row-independent: LayerNorm normalizes per row,
Linear computes each output row from one input row, GELU and the residual adds are
elementwise. The one stage where positions interact — attention — reads cached K/V
rows that are bit-identical to what the batch path would recompute (same Linear,
same input row), and the frozen core computes the last query row's scores over the
same ascending index order whether the query matrix has T rows or 1. Same inputs,
same ops, same order ⇒ same bits.

The subtle case is the softmax denominator and the value matmul. The batch path's
last query row softmaxes over ALL T keys, the future ones masked with `-1e9`;
`exp(-1e9 - max)` underflows to exactly `0.0`, and `x + 0.0 == x` for finite `x`,
so those masked keys contribute nothing to the denominator and `0.0 * v == 0.0`
contributes nothing to the output. The cached step softmaxes over only the `t =
pos+1` valid keys. Past keys accumulate in the same order in both; the batch path
merely adds exact zeros after them. So the two are bit-identical, not approximately
equal. This is why the step mask is `zeros_2d(1, t)` and NOT a causal mask:
causality is already enforced by what is IN the cache.

`matmul_transpose_b`'s inner sum runs over k ascending, matching `matmul`'s (and
the `@` operator's — all three accumulate each output cell over k ascending), so
the tied head is bit-identical to the batch spelling.

## The parity proof, three layers

1. **Doll-house step-vs-forward, every prefix** (`test_kv_cache.mojo`): V=11, T=8,
   C=8, L=2 (TWO layers, so a cross-layer cache-indexing bug cannot hide), H=2.
   For every prefix `t` in 1..8, `forward(ids[0:t])`'s last row equals the t-th
   `step` logits row EXACTLY — `assert_equal` per element, all V columns, no
   tolerance. This passed on the first run of the assembled step path.
2. **Generation parity** (`test_generate.mojo`): `generate_cached == generate`
   token-for-token, greedy AND a temperature-0.9 top-k+top-p config, on a
   two-layer doll-house, plus `rng.state` identical after both (stream parity).
3. **124M gate** (the example): the greedy continuation is character-identical to
   Part XVI's recorded text (quoted below), on all 124,439,808 real parameters.

Exact equality is the contract here, so `assert_equal` on Float64 is correct and
deliberate (the same policy as the deterministic-replay and gradient-doubling
tests); loosening it to a tolerance would hide exactly the bug class this part
exists to prevent, and is forbidden.

## Parity failures hit during development

None. The assembled step path produced bit-identical logits at every prefix on the
first green run of `test_kv_cache.mojo`. The reason is structural rather than luck:
the step path never re-derives attention math — it reuses `scaled_dot_product_
attention`, `slice_cols`, `concat_cols`, the row-wise `ln`/`mlp` forwards, and the
new `matmul_transpose_b` whose summation order was pinned equal to `matmul`'s by
its own test before it was ever wired into the head. The only genuinely new
bookkeeping — the wpe row index (`pos = cache.length`), the `pos+1` valid-region
bound, and incrementing `length` exactly once after all layers wrote — was written
against the D5 debugging order (positions → slices → hand-rolled math) as a
checklist, and there was nothing to debug. (If a mismatch HAD appeared, the fix is
never the test and never a tolerance — it is a position/slice/order bug in the
step.)

## Mojo exclusivity with the two `mut` cache buffers

`GPT.step` passes `cache.k[i]` and `cache.v[i]` as two separate `mut Tensor2D`
arguments into `blocks[i].step`, which threads them into `attn.step`. The plan
anticipated the borrow checker might object to two mutable list-element borrows in
one call. It did NOT: `cache.k` and `cache.v` are DISTINCT `List`s, so the two
element references have disjoint origins and the exclusivity check is satisfied
cleanly. No restructuring (transfer-rows-in-and-out) was needed. `self` is borrowed
immutably across the same call and does not conflict with the mutable cache
borrows. This is the AGENTS.md lesson worth keeping: two `mut` element args are
fine as long as they come from two different containers.

## Timing — the before/after on real 124M weights

Measured on this CPU (scalar float64, seed 1337, prompt "Hello, I'm a language
model,", 25 new tokens, peak uncached forward length 32):

- **One uncached token** (a full `gpt.forward` at the peak length 32 = prompt 8 +
  budget 25 − 1): **6.49 s**. This is the peak per-token cost the uncached path
  pays as the sequence fills (the last token conditions on the 31 before it); it
  is in line with Part XVI's recorded ~9.3 s/token, whose average ran over the
  growing 8→32 window plus the O(T²) attention the short prompt hides here.
- **Greedy, KV-cached**: 25 tokens in 19.54 s → **0.78 s/token**, a **8.3×**
  speedup versus the peak uncached token — and ~11.9× versus Part XVI's recorded
  9.3 s/token.
- **Nucleus (top-p 0.9, T 1.0), KV-cached**: 25 tokens in 19.22 s → **0.77
  s/token**, a **8.4×** speedup.

The speedup is bounded here by sequence length: the KV cache removes the
recomputation of past positions, which grows with T, but the constant "one pass"
per token (the memory-bound weight reads of the 124M linear layers) stays. At
these short lengths that constant still dominates, so the win is ~8-9×; it grows
toward T× as the context lengthens. And it composes with Part XVIII — SIMD and
blocked matmul shrink that constant next.

**Greedy continuation (character-identical to Part XVI):**

> Hello, I'm a language model, not a programming language. I'm a language model.
> I'm a language model. I'm a language model. I'm

**Nucleus continuation (character-identical to Part XVI):**

> Hello, I'm a language model, I know the basics...but here are some notes..."
>
> Melissa saw Gray showing a video of Alan Redge,

The bit-exact parity that the doll-house test pins at V=11 is thus demonstrated
to hold on all 124,439,808 real parameters: same weights, same forward, same
text — a fraction of the time.

## Deviations from plan

- The plan's D6 said the example prints "per-token wall-clock via `utils/timing`",
  but `utils/timing` holds benchmark STATISTICS (median, GFLOP/s), not a
  wall-clock reader — so the example times with `std.time.perf_counter_ns`
  directly, exactly as Part XVI's `gpt2_generate.mojo` does. Same measurement, no
  new dependency.
- The "one uncached token" before-number is timed at the FINAL sequence length
  (prompt + budget), not the prompt length. The uncached path re-runs the whole
  forward per token and that forward grows with the sequence, so the final-length
  forward is the honest peak per-token cost — and it lines up with Part XVI's
  recorded ~9.3 s/token. Timing over just the 8-token prompt understated the
  before-cost (it gave ~2.1 s, a misleadingly small ~2.6× speedup).
- The gate is `pixi run test-fast` (the canonical green gate, excluding
  `test_seq_tasks` per the standing #6554 rule), not `pixi run test`.

## What this part deliberately did NOT do (Part XVIII)

No SIMD, no threading, no blocked matmul, no batch-path retrofits, no batch
prefill, no cached sliding window (architecture-blocked by GPT-2's absolute
positions — documented in `generate_cached`'s docstring, not hacked around), no
beam/batched generation, no CI/gauntlet, no training/checkpoint/converter changes.
The arithmetic is still scalar f64. This part changed the ALGORITHM; the next
changes the arithmetic, and the two compose.

## Review triage

Dual external review over `git diff main...part-17-kv-cache`: Codex (GPT, high
reasoning, danger-full-access — it independently ran `pixi run test-fast` green)
and Claude Opus 4.8. Both independently VERIFIED the bit-parity argument (additive
diff, frozen-core reuse, zeros `[1, t]` mask with `t = pos+1`, matched summation
orders, every-prefix `assert_equal`, stream parity, layering). Findings:

- **FIXED (Codex, blocking) — zero-budget contract mismatch.** `generate_cached`
  ran the overflow gate BEFORE the `max_new_tokens == 0` no-op return, so an
  over-context prompt with budget 0 raised here while `generate` (whose loop runs
  zero times) returns `[]`. Moved the 0-budget return ahead of the overflow gate;
  added `test_cached_zero_budget_noop_ignores_overflow` pinning `[]` + untouched
  rng + agreement with `generate` on an over-length prompt.
- **FIXED (Codex, nice-to-have) — the example's peak-length benchmark was one
  position too long.** For a budget of N the uncached path forwards lengths
  `P .. P+N-1`, so the peak is `P+N-1`, not `P+N`. Corrected `final_len` and the
  wording; re-ran the example.
- **FIXED (Codex, nice-to-have) — `check_compatible` indexed `v[i]` without
  validating `len(v)`.** A publicly built `@fieldwise_init` cache with mismatched
  k/v list lengths would hit a bounds trap instead of the promised named error.
  Added the `len(v)` check and `test_check_compatible_mismatched_kv_layer_count_
  raises`.
- **FIXED (Opus, nit) — a test comment named the wrong tied-head spelling.**
  `test_matmul_transpose_b_equals_composed_spelling` compares against `matmul`
  (ijk), while `gpt.forward`'s head uses the `@` operator (ikj). All three
  accumulate k ascending and are bit-identical; clarified the comment to say so
  and to point at the end-to-end step parity test as the real-path cover.
- **REJECTED (Opus, nit) — overflow guard "one stricter than necessary".** The
  cached path could physically admit `len(prompt) + max_new_tokens ==
  context_length + 1` (peak row `P+N-2 < capacity`), and `generate` doesn't slide
  there either, so parity would hold. Kept the `<= context_length` guard anyway:
  it is the approved policy, and the invariant "prompt + budget fits the context"
  is a cleaner teaching statement than "fits with one to spare". Opus flagged it
  as defensible as-is. No behavior a caller can observe is lost — a rejected call
  is one token short of a hard architectural ceiling and the message says so.
