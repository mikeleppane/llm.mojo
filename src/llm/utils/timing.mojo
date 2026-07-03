# Benchmark statistics, isolated from any timing API.
#
# The measurement math is independent of how you read the clock, so it lives
# here where it can be unit-tested with hand-computed inputs. The actual timer
# call (perf_counter_ns) stays in the benchmark harness, which is the part that
# might drift across Mojo versions.

from std.collections import List


def median_ns(mut samples: List[Int]) -> Int:
    # Median of `samples`, sorting them in place first (insertion sort — sample
    # counts are tiny). For an even count this returns the upper-middle element
    # rather than averaging the two middles: a deliberate simplification that
    # never matters for a benchmark median. Mutates the argument (sorts it).
    for i in range(1, len(samples)):
        var key = samples[i]
        var j = i - 1
        while j >= 0 and samples[j] > key:
            samples[j + 1] = samples[j]
            j -= 1
        samples[j + 1] = key
    var n = len(samples)
    return samples[n // 2]


def gflops_matmul(m: Int, k: Int, n: Int, ns: Int) -> Float64:
    # Throughput of an [m, k] @ [k, n] matmul that took `ns` nanoseconds, in
    # GFLOP/s. A matmul does 2*m*k*n floating-point ops (one multiply, one add
    # per accumulation).
    var flops = 2.0 * Float64(m) * Float64(k) * Float64(n)
    var seconds = Float64(ns) / 1.0e9
    return (flops / seconds) / 1.0e9
