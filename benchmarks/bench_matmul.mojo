"""Naive-scalar vs production-kernel matmul timing harness.

`matmul` is the textbook scalar triple loop in ijk order: its inner loop strides
down a column of b, which is cache-unfriendly. `matmul_ikj` delegates to the `@`
operator — the production kernel: contiguous ikj streaming, SIMD over columns,
and threading once the work crosses ~1M multiply-adds (so size 128 and 256 run
threaded here, size 64 does not). The printed gap is therefore the *total*
speedup of the optimized kernel over the naive baseline — loop order plus
vectorization plus, at the larger sizes, multithreading — not loop order in
isolation. It measures, it does not assert.

Run (ideally a release build) and record your own numbers:
    pixi run mojo run -I src benchmarks/bench_matmul.mojo
"""

from std.time import perf_counter_ns
from std.collections import List

from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.tensor.ops import matmul, matmul_ikj
from llm.utils.timing import median_ns, gflops_matmul


def _filled(size: Int) -> Tensor2D:
    """Return a size x size tensor of small non-trivial values.

    Timing is content-blind, but non-zero data avoids any zero-special-casing
    surprises.

    Args:
        size: Square matrix dimension.

    Returns:
        A newly allocated [size, size] tensor.
    """
    var t = zeros_2d(size, size)
    for i in range(size):
        for j in range(size):
            t[i, j] = Float64((i * 7 + j * 3) % 19) * 0.1 - 0.9
    return t^


def bench_one(name: String, size: Int, warmup: Int, runs: Int) raises:
    """Time a size x size matmul in loop order `name` and print median + GFLOP/s.

    Args:
        name: "ijk" selects matmul (the naive scalar baseline); anything else
            selects matmul_ikj (the optimized `@` kernel).
        size: Square matrix dimension.
        warmup: Untimed warmup calls.
        runs: Timed calls; the median is reported.
    """
    var a = _filled(size)
    var b = _filled(size)

    for _ in range(warmup):
        if name == "ijk":
            _ = matmul(a, b)
        else:
            _ = matmul_ikj(a, b)

    var samples = List[Int]()
    for _ in range(runs):
        var start = perf_counter_ns()
        var c = matmul(a, b) if name == "ijk" else matmul_ikj(a, b)
        var end = perf_counter_ns()
        samples.append(Int(end - start))
        _ = c  # keep the result live so the call is not optimized away
    var med = median_ns(samples)
    print(
        name,
        "size",
        size,
        "median_ns",
        med,
        "GFLOP/s",
        gflops_matmul(size, size, size, med),
    )


def main() raises:
    """Benchmark both loop orders across a range of matrix sizes."""
    var sizes = [64, 128, 256]
    for i in range(len(sizes)):
        var size = sizes[i]
        bench_one("ijk", size, warmup=2, runs=5)
        bench_one("ikj", size, warmup=2, runs=5)
