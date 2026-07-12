"""Decode-shape kernel timing table.

Each decoded token is one forward pass whose flops run almost entirely through
`matmul_transpose_b` (Linear forwards, attention scores, the tied head) and the
`@` matmul (attention weights @ v). This harness times those kernels at the
shapes the 124M model actually hits during greedy decode, plus a couple of batch
shapes, so an optimization shows up as a kernel time, not a guess. It measures,
it does not assert. Each result is the median of `runs` samples after `warmup`
untimed calls.

Run (ideally a release build) and record the numbers:
    pixi run mojo precompile src/llm -o build/llm.mojopkg
    pixi run mojo run -I build benchmarks/bench_kernels.mojo
"""

from std.time import perf_counter_ns
from std.collections import List

from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.tensor.ops import matmul_transpose_b, matmul_ikj
from llm.utils.timing import median_ns


def _filled(rows: Int, cols: Int) -> Tensor2D:
    """Return a rows x cols tensor of small non-trivial values.

    Timing is content-blind, but non-zero data avoids any zero-special-casing
    surprises.

    Args:
        rows: Row count.
        cols: Column count.

    Returns:
        A newly allocated [rows, cols] tensor.
    """
    var t = zeros_2d(rows, cols)
    for i in range(rows):
        for j in range(cols):
            t[i, j] = Float64((i * 7 + j * 3) % 19) * 0.1 - 0.9
    return t^


def bench_mtb(
    name: String, m: Int, k: Int, n: Int, warmup: Int, runs: Int
) raises:
    """Time matmul_transpose_b at [m, k] . [n, k]^T and print median us + GFLOP/s.

    This kernel is x @ W^T for a Linear (m rows, k in-features, n out-features),
    the attention score kernel, and the tied head (m=1, k=C, n=V).

    Args:
        name: Label for the printed row.
        m: Rows of the left operand.
        k: Shared inner dimension (in-features).
        n: Rows of the right operand (out-features).
        warmup: Untimed warmup calls.
        runs: Timed calls; the median is reported.
    """
    var a = _filled(m, k)
    var b = _filled(n, k)
    for _ in range(warmup):
        _ = matmul_transpose_b(a, b)
    var samples = List[Int]()
    for _ in range(runs):
        var start = perf_counter_ns()
        var c = matmul_transpose_b(a, b)
        var end = perf_counter_ns()
        samples.append(Int(end - start))
        _ = c  # keep the result live so the call is not optimized away
    var med = median_ns(samples)
    var gflops = (
        2.0 * Float64(m) * Float64(k) * Float64(n) / (Float64(med) / 1.0e9)
    ) / 1.0e9
    print(
        name,
        " mtb [",
        m,
        "x",
        k,
        "] . [",
        n,
        "x",
        k,
        "]^T  median_us ",
        Float64(med) / 1.0e3,
        " GFLOP/s ",
        gflops,
    )


def bench_ikj(
    name: String, m: Int, k: Int, n: Int, warmup: Int, runs: Int
) raises:
    """Time matmul_ikj at [m, k] @ [k, n] and print median us + GFLOP/s.

    This is the true-matmul path (attention weights @ v, and the `@` operator it
    delegates to).

    Args:
        name: Label for the printed row.
        m: Rows of the left operand.
        k: Shared inner dimension.
        n: Columns of the right operand.
        warmup: Untimed warmup calls.
        runs: Timed calls; the median is reported.
    """
    var a = _filled(m, k)
    var b = _filled(k, n)
    for _ in range(warmup):
        _ = matmul_ikj(a, b)
    var samples = List[Int]()
    for _ in range(runs):
        var start = perf_counter_ns()
        var c = matmul_ikj(a, b)
        var end = perf_counter_ns()
        samples.append(Int(end - start))
        _ = c
    var med = median_ns(samples)
    var gflops = (
        2.0 * Float64(m) * Float64(k) * Float64(n) / (Float64(med) / 1.0e9)
    ) / 1.0e9
    print(
        name,
        " ikj [",
        m,
        "x",
        k,
        "] @ [",
        k,
        "x",
        n,
        "]  median_us ",
        Float64(med) / 1.0e3,
        " GFLOP/s ",
        gflops,
    )


def main() raises:
    """Run the kernel timing table over decode, prefill, and attention shapes.
    """
    print("== decode shapes (single-token, m=1), 124M dims ==")
    bench_mtb("c_attn  ", 1, 768, 2304, warmup=3, runs=11)
    bench_mtb("c_proj  ", 1, 768, 768, warmup=3, runs=11)
    bench_mtb("mlp_up  ", 1, 768, 3072, warmup=3, runs=11)
    bench_mtb("mlp_down", 1, 3072, 768, warmup=3, runs=11)
    bench_mtb("tiedhead", 1, 768, 50257, warmup=3, runs=11)

    print("== batch/prefill shapes (T=64), 124M dims ==")
    bench_mtb("c_attn  ", 64, 768, 2304, warmup=2, runs=5)
    bench_mtb("tiedhead", 64, 768, 50257, warmup=1, runs=3)

    print("== attention weights @ v (true matmul), one head T=64 D=64 ==")
    bench_ikj("attn_wv ", 64, 64, 64, warmup=3, runs=11)
