"""Tests for the benchmark statistics helpers.

Pure math, independent of any clock, so they get hand-computed oracles: the
median of a known list (odd and even counts, unsorted input) and a GFLOP/s figure
worked out by hand.
"""

from std.testing import assert_equal, assert_almost_equal, TestSuite

from llm.utils.timing import median_ns, gflops_matmul


def test_median_odd_count() raises:
    """`median_ns` of an odd-count unsorted list is the middle value."""
    var samples = [30, 10, 20, 50, 40]
    assert_equal(median_ns(samples), 30)


def test_median_even_count_upper_middle() raises:
    """`median_ns` of an even-count list returns the upper-middle element."""
    # Sorted: [10, 20, 30, 40]; n // 2 == 2 -> the upper-middle element, 30.
    var samples = [40, 10, 30, 20]
    assert_equal(median_ns(samples), 30)


def test_median_sorts_in_place() raises:
    """`median_ns` sorts its input list in place."""
    var samples = [3, 1, 2]
    _ = median_ns(samples)
    assert_equal(samples[0], 1)
    assert_equal(samples[1], 2)
    assert_equal(samples[2], 3)


def test_gflops_hand_computed() raises:
    """`gflops_matmul` matches a hand-computed GFLOP/s figure."""
    # A 100x100x100 matmul is 2*1e6 = 2e6 flops. In 1e6 ns (1 ms) that is
    # 2e6 / 1e-3 = 2e9 flop/s = 2 GFLOP/s.
    var g = gflops_matmul(100, 100, 100, 1_000_000)
    assert_almost_equal(g, 2.0, atol=1e-9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
