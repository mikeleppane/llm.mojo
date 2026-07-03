# ijk vs ikj matmul timing harness.
#
# The two loop orders compute identical results (proven in tests/test_matmul.mojo),
# so any timing difference is pure memory behavior: ijk's inner loop strides down
# a column of b; ikj streams both operands contiguously. Expect the gap to widen
# as the matrices outgrow L1 cache.
#
# This is outside the test gate — it measures, it does not assert. Run it (ideally
# a release build) and record your own numbers:
#     pixi run mojo run -I src benchmarks/bench_matmul.mojo
#
# The timer uses perf_counter_ns; the statistics (median, GFLOP/s) come from the
# unit-tested helpers in llm.utils.timing.

from std.time import perf_counter_ns
from std.collections import List

from llm.tensor.tensor2d import zeros_2d
from llm.tensor.ops import matmul, matmul_ikj
from llm.utils.timing import median_ns, gflops_matmul


def bench_one(name: String, size: Int, warmup: Int, runs: Int) raises:
    var a = zeros_2d(size, size)
    var b = zeros_2d(size, size)

    for _ in range(warmup):
        if name == "ijk":
            _ = matmul(a, b)
        else:
            _ = matmul_ikj(a, b)

    var samples = List[Int]()
    for _ in range(runs):
        var start = perf_counter_ns()
        if name == "ijk":
            _ = matmul(a, b)
        else:
            _ = matmul_ikj(a, b)
        var end = perf_counter_ns()
        samples.append(Int(end - start))

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
    var sizes = [64, 128, 256]
    for i in range(len(sizes)):
        var size = sizes[i]
        bench_one("ijk", size, warmup=2, runs=5)
        bench_one("ikj", size, warmup=2, runs=5)
