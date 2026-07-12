# Part XVIII notes — Performance: making the arithmetic worthy of the algorithm

Part XVII fixed the ALGORITHM (KV cache: one pass per token instead of T). This
part fixes the ARITHMETIC: the ~250M flops per decoded token were still running
through scalar f64 loops over `List[Float64]` storage. The discipline: measure,
optimize the proven bottleneck, re-measure, and prove seventeen parts of
correctness infrastructure caught everything that moved — every optimization
classified Class A (order-preserving, bit-identical, exact-equality tested) or
Class B (reassociating SIMD reduction, ~k·eps, 1e-12-relative tested), the whole
existing suite green UNCHANGED.

The gate throughout is `pixi run test-fast` (the canonical green gate; excludes
`test_seq_tasks.mojo` per the standing #6554 rule). All timings on this machine
(32 logical cores, AVX2 → SIMD width 4 for f64), best-of-N, release builds.

## The numerics contract, decided up front

`matmul_transpose_b` becomes a **Class B** kernel (multi-accumulator SIMD dot
over the contraction k). Its documented contract therefore changes from
"bit-identical to `matmul(a, transpose(b))`" to "identical to it within k·eps
reassociation error" (≈1e-13 at k=3072, six orders below the tightest suite
tolerance). This is the single deliberate change to an existing test:
`test_matmul_transpose_b_equals_composed_spelling` moves from `assert_equal`
(exact) to a 1e-12 **relative** comparison, and gains real-dimension and
ragged-tail shapes — i.e. it becomes the D1.4-mandated Class B kernel test. It is
NOT a loosened golden hiding a regression; it is the correct test for a kernel
whose contract deliberately reassociates. Every OTHER test stays byte-untouched:

- `matmul` keeps its scalar ijk teaching spelling — the reference oracle.
- `matmul_ikj` / the `@` operator get SIMD over the OUTPUT j-dimension (Class A):
  each output cell still sums k ascending, so they stay bit-identical to `matmul`.
- XVII's step-vs-forward EXACT parity (`test_kv_cache.mojo`) stays green because
  the batch tied head is retrofitted to the SAME `matmul_transpose_b` kernel the
  step path uses — both reach the identical Class B kernel on identical operand
  pairs, so identical bits come out of both.
- The doll-house 1e-9, the 124M 1e-6 logit goldens, and every layer test absorb
  the ~1e-13 wobble with orders of margin.

## Day-one spike findings (scratch code, not committed)

Run on the pinned Mojo 1.0.0b2, AVX2 host, SIMD width for `DType.float64` = **4**.

**(a) `List[Float64].unsafe_ptr()` + SIMD works.** `list.unsafe_ptr()` returns an
`UnsafePointer`; `ptr.load[width=W](i)` reads a `SIMD[DType.float64, W]` from a
contiguous span and `(v0+v1+...).reduce_add()` horizontally sums it. No storage
redesign needed — the flat row-major `List[Float64]` is directly SIMD-addressable.
The STOP-and-ask fallback (Tensor2D storage redesign) is NOT triggered.

**(b) The single-thread SIMD dot win is real and large.** A 4-accumulator,
width-4 unrolled dot with a scalar tail, vs the scalar loop, best-of-10 (100 dots
per sample):

| k      | scalar ns/100 | simd ns/100 | speedup |
|--------|---------------|-------------|---------|
| 768    | 77759         | 2958        | 26.3×   |
| 3072   | 308882        | 12226       | 25.3×   |
| 50257  | 5275222       | 345117      | 15.3×   |

Above the plan's expected 8–16× — the scalar loop is latency-bound on its add
chain, exactly as predicted; the multi-accumulator SIMD breaks that dependency.
The 50257 case (the tied-head row count) is bandwidth-bound so the ratio drops,
still >15×. On this integer-valued spike data the reassociation error was exactly
0 (values exactly representable); the ~k·eps error appears only on real
fractional weights, where the 1e-12 tests bound it.

**(c) `std.algorithm.parallelize` is usable.** Signature (from the compiler's
candidate list):
`parallelize[origins, //, func: def(Int) capturing -> None](num_work_items: Int, num_workers: Int = ..., ctx = None)`.
The closure must be `@parameter def(Int)`; captures (e.g. `unsafe_ptr`s to the
operands and output) are inferred. A parallel row-partitioned matvec over
[4096, 3072] gave ~1.9× over 32 cores — bandwidth-bound, NOT core-bound (100 MB
of `a` streamed once per matvec). Realistic expectation for Stage 2: the decode
kernels read multi-hundred-MB weight matrices, so threading is bounded by memory
bandwidth, not the 32× core count. The API works → the Stage 2 gate is OPEN;
whether it earns its place is a per-shape benchmark question answered later.

## Baseline table (branch point = current main, scalar f64)

Recorded before the first optimization commit. Same machine, release builds
(`mojo run -I build`). Kernel table from `benchmarks/bench_kernels.mojo`
(median of N samples); e2e from `examples/gpt2_generate_cached.mojo` (seed 1337,
prompt "Hello, I'm a language model,", 25 new tokens); training step from
`benchmarks/bench_gpt_step.mojo` (doll-house V=256 T=64 C=128 L=6 H=4, median of
21). Every scalar kernel sits at ~1.6 GFLOP/s — the latency-bound add chain the
SIMD dot is about to break.

**Decode-shape kernels (single token, m=1), 124M dims — `matmul_transpose_b`:**

| kernel   | shape                    | baseline median | GFLOP/s |
|----------|--------------------------|-----------------|---------|
| c_attn   | [1,768] · [2304,768]^T   | 2155 µs         | 1.64    |
| c_proj   | [1,768] · [768,768]^T    | 715 µs          | 1.65    |
| mlp_up   | [1,768] · [3072,768]^T   | 2903 µs         | 1.63    |
| mlp_down | [1,3072] · [768,3072]^T  | 2898 µs         | 1.63    |
| tiedhead | [1,768] · [50257,768]^T  | 49545 µs        | 1.56    |

**Batch/prefill shapes (T=64), 124M dims:**

| kernel        | shape                      | baseline median | GFLOP/s |
|---------------|----------------------------|-----------------|---------|
| c_attn batch  | [64,768] · [2304,768]^T    | 136061 µs       | 1.66    |
| tiedhead batch| [64,768] · [50257,768]^T   | 3129594 µs      | 1.58    |
| attn wv (ikj) | [64,64] @ [64,64]          | 353 µs          | 1.48    |

**End-to-end (124M greedy/nucleus) and training step:**

| metric                        | baseline        |
|-------------------------------|-----------------|
| one uncached token (len 32)   | 6.371 s         |
| greedy, KV-cached             | 0.784 s/token   |
| nucleus (top-p 0.9), KV-cached| 0.797 s/token   |
| training step (doll-house)    | 367.76 ms       |

**Greedy continuation (the text golden — character-identical to Parts XVI/XVII):**

> Hello, I'm a language model, not a programming language. I'm a language model.
> I'm a language model. I'm a language model. I'm

Per-token decode is dominated by the twelve layers' four Linear kernels plus the
one tied head: 12·(2155+715+2903+2898) µs + 49545 µs ≈ 153 ms of pure
`matmul_transpose_b` per token at the kernel level — the SIMD dot targets exactly
this. The tied head alone (~49.5 ms) is the single biggest kernel.

## Per-stage results

### Stage 1 — call-site retrofits + SIMD dot kernel + SIMD-over-j `@`

Commits: Linear forward retrofit (Class A), attention-score + batch-tied-head
retrofit (Class A), `_simd_dot` under matvec/matmul_transpose_b (Class B),
`@` vectorized over output columns (Class A). Whole suite green UNCHANGED
(the one deliberate test edit is the Class B contract change described above);
XVII exact step-vs-forward parity untouched and green; 124M 1e-6 goldens green.

**Kernels (median µs, decode m=1):**

| kernel   | baseline | Stage 1 | speedup |
|----------|----------|---------|---------|
| c_attn   | 2155     | 155     | 13.9×   |
| c_proj   | 715      | 49.4    | 14.5×   |
| mlp_up   | 2903     | 235     | 12.3×   |
| mlp_down | 2898     | 232     | 12.5×   |
| tiedhead | 49545    | 12319   | 4.0×    |
| c_attn batch [64] | 136061 | 10059 | 13.5× |
| tiedhead batch [64] | 3129594 | 789810 | 4.0× |
| attn_wv (`@`, ikj) | 353 | 26.5 | 13.3× |

The Linear/score kernels hit ~20–24 GFLOP/s (SIMD-bound); the tied head stays at
~6.3 GFLOP/s — it is memory-bound, streaming the ~300 MB [50257,768] table once
per token, so its 4× is bandwidth-limited, not compute-limited. It is now the
single biggest decode kernel (~12.3 ms of a ~53 ms token).

**End-to-end and training step:**

| metric                         | baseline      | Stage 1        | speedup |
|--------------------------------|---------------|----------------|---------|
| one uncached token (len 32)    | 6.371 s       | 0.740 s        | 8.6×    |
| greedy, KV-cached              | 0.784 s/token | 0.0531 s/token | 14.8×   |
| nucleus (top-p 0.9), KV-cached | 0.797 s/token | 0.0616 s/token | 12.9×   |
| training step (doll-house)     | 367.76 ms     | 38.59 ms       | 9.5×    |

**Greedy text unchanged** — character-identical to the baseline/XVII golden. The
Class B reassociation (~1e-13 relative) flipped no greedy argmax over the 25
tokens, so D1's near-tie re-pin clause does not trigger; the frozen 124M 1e-6
logit goldens remain the arbiter and pass. Nucleus text also identical (stream
parity holds).

### Stage 2 — threading (D4 gate: OPEN)

The day-one spike proved `parallelize` usable, so the gate opened. Both reduction
kernels thread above a ~1M multiply-add threshold (below it the dispatch costs
more than it saves, so c_proj and the tiny attention products stay serial):
`matmul_transpose_b` partitions the output columns, the `@` operator partitions
the output rows. Both are Class A by construction — a static partition, each
output element computed entirely by one worker in the unchanged order, no shared
accumulators — so bit-identical to serial and run-to-run stable. Two determinism
tests pin it (a kernel-level two-call exact check at n=4096, and a generate_cached
determinism test sized so the tied head crosses the threshold).

**Kernels (median µs, decode m=1), Stage 1 -> Stage 2:**

| kernel   | Stage 1 | Stage 2 (threaded) | speedup |
|----------|---------|--------------------|---------|
| c_attn   | 155     | 31                 | 5.0×    |
| mlp_up   | 235     | 41                 | 5.7×    |
| mlp_down | 232     | 46                 | 5.0×    |
| tiedhead | 12319   | 4753               | 2.6×    |
| c_attn batch [64] | 10059 | 876          | 11.5×   |
| tiedhead batch [64] | 789810 | 148533     | 5.3×    |
| c_proj (serial, sub-threshold) | 49 | ~70 (noise) | — |

The tied head's 2.6× is bandwidth-bound (32 cores pull more aggregate RAM
bandwidth on the 300 MB table, but not linearly). The compute-bound Linear
kernels hit 100-115 GFLOP/s.

### D6 — matmul_transpose_a (gate: OPEN, profile evidence below)

Profiling the backward products AFTER the matmul itself was vectorized showed the
`transpose()` allocation — a slow strided element-copy — had become the majority
of each product's time:

| backward product (doll-house) | transpose µs | full (transpose+@) µs | transpose frac |
|-------------------------------|--------------|-----------------------|----------------|
| mlp_fc  d_out^T @ x           | 176          | 271                   | **65%**        |
| c_attn  d_out^T @ x           | 133          | 202                   | **66%**        |
| tiedhead d_logits^T @ h       | 89           | 140                   | **64%**        |
| mlp_fc @ 124M widths          | 1120         | 4010                  | **28%**        |

So the gate opened. `matmul_transpose_a(a, b) = a^T @ b` computes these directly
(no transpose copy), accumulating over the shared row dimension in the SAME
ascending order as `transpose(a) @ b` — Class A, bit-identical, pinned by an
exact-equality test and by every backward test (the exact gradient-doubling and
finite-difference checks) staying green. Retrofitted the big-transpose backward
sites: Linear dW, the tied head d_table, attention dV and dK. The two small
`d_out @ transpose(v)` products (v is ~[64,64], a negligible transpose) stay on
the Class A `@` path — routing them through the Class B matmul_transpose_b would
change backward bits for no real gain, and D6 is Class A throughout.

### Stage 3 — fused decode attention (D5 gate: SKIPPED, profile evidence below)

D5 gates on whether the KV-cache slice/copy volume dominates decode. It does not
at representative contexts, and the cheaper allocation-hygiene fix (contiguous
memcpy in slice/concat, Class A) addressed the copy cost without a fused-attention
rewrite:

| KV copies per layer (slice_rows + slice_cols) | element-loop | memcpy |
|-----------------------------------------------|--------------|--------|
| context 32 (the demo)                         | 46.7 µs      | 5.0 µs |
| context 512                                   | 798 µs       | 152 µs |

At the demo's context 32 the copies were ~0.56 ms/token (~2% of a 28 ms token)
BEFORE hygiene, and ~0.06 ms after — nowhere near dominant. They grow with
context, but the memcpy hygiene (5-9× cheaper) keeps them subdominant far longer,
and re-spelling attention in a fused step under XVII's exact-parity constraint is
not justified by the profile. Stage 3 is SKIPPED; a fused decode step remains the
right lever only once contexts are long enough that even the memcpy'd copies
dominate — future work, with XVII's parity test as the tripwire. Either outcome is
valid per the plan; this is the evidence for choosing the skip.

### Allocation hygiene (Class A, evidence: the D5 copy table above)

slice_rows / slice_cols / concat_cols copied element by element; each output row
is a contiguous run, so one memcpy per row replaces the inner loop. Same bytes
(the exact-value slicing tests pass unchanged), 5-9× cheaper copies, and it lifted
the training step further (24.6 -> 20.2 ms) by speeding the attention head splits.
zeros_2d bulk allocation was already done on main (verified, not redone).

## Final table — baseline -> final (the payoff arc)

**Decode-shape kernels (median µs, m=1):**

| kernel   | baseline | Stage 1 SIMD | Stage 2 +threads | total |
|----------|----------|--------------|------------------|-------|
| c_attn   | 2155     | 155          | 31               | 69×   |
| c_proj   | 715      | 49           | 49 (serial)      | 15×   |
| mlp_up   | 2903     | 235          | 41               | 71×   |
| mlp_down | 2898     | 232          | 46               | 63×   |
| tiedhead | 49545    | 12319        | 4753             | 10×   |

**End-to-end (124M) and training step — baseline -> final:**

| metric                         | baseline      | final          | speedup |
|--------------------------------|---------------|----------------|---------|
| one uncached token (len 32)    | 6.371 s       | 0.161 s        | 39.5×   |
| greedy, KV-cached              | 0.784 s/token | 0.0249 s/token | **31.5×** |
| nucleus (top-p 0.9)            | 0.797 s/token | 0.0341 s/token | 23.4×   |
| training step (doll-house)     | 367.76 ms     | 20.18 ms       | **18.2×** |

The blog's three-act arc, greedy s/token: Part XVI scalar+no-cache ~9.3 ->
Part XVII cache 0.784 -> Part XVIII cache+SIMD+threads **0.0249** — ~373× since
the uncached forward, on the same 124M weights and the same text.

**Greedy and nucleus text: character-identical to the baseline/XVII golden at
every stage.** The Class B reassociation (matmul_transpose_b/matvec/_simd_dot,
~1e-13 relative) flipped no greedy argmax over the 25 tokens across any stage, so
D1's near-tie re-pin clause never fired; the frozen 124M 1e-6 logit goldens remain
the arbiter and pass unchanged.

## Class A / Class B inventory of every changed path

The classification is of the FINAL code, not of each commit in isolation — a
subtlety worth stating because the forward call-site retrofits (Stage 1a) were
Class A *when committed* (matmul_transpose_b was still scalar then) but became
transitively Class B once Stage 1b made the kernel reassociate. The final-state
truth:

- **Class B (reassociating, 1e-12-relative tested):** `_simd_dot`, and through it
  `matvec` and `matmul_transpose_b` — the only reassociating kernel — AND, by
  transitivity, every forward product now routed through it: the Linear forwards,
  the attention `q @ k^T` scores, and the tied head (batch and cached step). These
  paths reassociate the reduction (~k·eps vs their old scalar `@ transpose(·)`
  spelling); they are covered by matmul_transpose_b's 1e-12 kernel test, the
  per-layer hand-computed/finite-diff tests, and the frozen 124M 1e-6 logit
  goldens. That they are Class B does not threaten XVII's exact parity: batch and
  cached-step both moved to the SAME kernel together, so they stay bit-identical
  to *each other* even though both now differ from the old scalar bits.
- **Class A (order-preserving, exact-equality tested):** the `@` operator's
  SIMD-over-columns and its row threading; `matmul_transpose_b`'s column threading
  (bit-identical to the serial `_simd_dot`, pinned by a threaded-vs-serial test);
  `matmul_transpose_a` and its backward retrofits (bit-identical to the scalar
  `matmul(transpose(a), b)` oracle); the slice/concat memcpy hygiene (bit-exact,
  fractional-value tested). `matmul` is untouched — the scalar oracle every fast
  kernel is tested against.

## Elementwise/reduction ops (gelu, layernorm, softmax) — SIMD SKIPPED with evidence

Measured at decode shapes after Stage 1/2 (median µs): gelu_rows [1,3072] 18.9,
layernorm [1,768] 3.4, softmax_rows [1,1024] 4.1, vs one c_attn kernel 18.8.
Per token that is ~gelu 12·18.9 + layernorm 24·3.4 + softmax 144·~1 ≈ 0.45 ms —
**~1.8% of the ~25 ms token** and ~6% of the ~7 ms of matmul kernels. Below the
bar worth a hand-written SIMD elementwise kernel; the plan gated this on "the
post-matmul profile row says they matter," and it does not. SKIPPED, measured not
gold-plated.

## Goldens that failed during development

None failed unexpectedly. The one deliberate test change was converting
`test_matmul_transpose_b_equals_composed_spelling` from exact (`assert_equal`) to
1e-12 relative when matmul_transpose_b became Class B — a contract change, not a
regression, documented at the top of these notes and mandated by D1.4 for every
Class B kernel. Every other test passed unchanged at every stage, including XVII's
exact step-vs-forward parity and the exact gradient-doubling backward tests.

## Deviations from plan

- The `@` operator threading and `matmul_transpose_a` both turned out to be
  bit-identical to the scalar spelling (no FMA-contraction divergence on this
  toolchain), so their Class A exact-equality tests use `assert_equal`, as the
  plan hoped rather than merely tolerated.
- Stage 3 (fused decode) skipped on profile evidence (copies subdominant, memcpy
  hygiene sufficient) — an explicitly-valid outcome, not a cut corner.
- The gate is `pixi run test-fast` throughout (excludes test_seq_tasks per the
  standing #6554 rule).

## Review triage

Dual external review over `git diff main...part-18-performance` — Codex GPT-5.6
(high, danger-full-access; it independently ran `pixi run fmt-check` and
`pixi run test-fast` green) and Claude Opus 4.8 (xhigh; independently ran the
affected suite green). Both VERIFIED the code is correct, the threading is a sound
static partition, XVII's exact step-vs-forward parity still holds with
matmul_transpose_b now Class B, and every category was audited clean (naive
`matmul` oracle intact, `_simd_dot` tail correct, no unsafe_ptr across a
reallocation, kv_cache/step untouched, no fast-math/Float32/GPU/BLAS/syntax
drift, all edits in-scope). The findings were about test rigor and classification
honesty, not runtime bugs. All were FIXED:

- **FIXED (both, classification) — forward retrofits mislabelled order-preserving.**
  The Linear-forward, attention-score, and tied-head comments claimed "the same
  k-ascending accumulation" / "bit-identical to the spellings they replace," but
  those paths now route through the Class B `_simd_dot` and reassociate. Corrected
  the comments (linear.mojo, attention.mojo) and the Class A/B inventory above to
  state the forward products are transitively Class B; gpt.mojo's tied-head
  comment was already worded correctly.
- **FIXED (Codex, blocking) — the Class B test's 1e-9 floor was too loose.**
  `_assert_close_rel` mixed a 1e-9 absolute floor with the 1e-12 relative term, so
  the floor dominated at unit scale and the test was effectively non-relative.
  Measured the actual reassociation error (max ~2.3e-13 at k=4, <1e-13 at
  k=768/3072) and tightened to a genuine `1e-12·|want| + 1e-13` bound — the
  relative term governs every real cell with ~4× margin.
- **FIXED (both) — matmul_transpose_a compared against the fast `@`.** Both
  matmul_transpose_a and `@` vectorize over columns, so that was a fast-vs-fast
  check (sound only transitively). Now compares exactly against the scalar
  `matmul(transpose(a), b)` oracle.
- **FIXED (Codex, blocking) — memcpy slice/concat lacked a bit-exact test.** Added
  `test_slice_concat_bit_exact_fractional`: slice_rows/slice_cols/concat_cols on
  not-exactly-representable fractional values, asserting bit-for-bit equality
  (`assert_equal`) against the source — a byte-misaligned or short copy can't hide
  under a tolerance.
- **FIXED (Codex + Opus) — threading tests strengthened.** Added
  `test_matmul_transpose_b_threaded_matches_serial` (threaded result EXACTLY equals
  the serial `_simd_dot` via `matvec(b, a.row(i))` — proves threading is Class A,
  not merely deterministic) and `test_matmul_transpose_a_threaded_is_deterministic`
  (two-call exact agreement, matching the coverage the other two kernels had).
- **FIXED (both) — elementwise-SIMD skip lacked a profile row.** Added the measured
  evidence above (elementwise ~1.8% of a token) and the explicit SKIP.
- **FIXED (Codex, nit) — trailing whitespace + EOF blank line** in these notes.

No findings were rejected; the reviewers converged on the same substance at
different severities (Codex graded the classification/rigor items blocking, Opus
graded them nice-to-have/nit), and fixing all of them was cheap and correct.
