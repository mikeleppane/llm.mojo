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

<!-- one column appended per stage below -->

