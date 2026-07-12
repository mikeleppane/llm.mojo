"""Tests for stable softmax and its temperature variant.

Stability is the whole point: a naive exp(1000) overflows to NaN; the
max-subtracting version returns a clean uniform distribution. The temperature
limits (high -> uniform, low -> argmax) re-prove stability, since a small
temperature scales logits into overflow territory before the max subtraction.
"""

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.tensor.ops import softmax_row, softmax_rows, softmax_row_temperature
from llm.tensor.tensor2d import from_rows, zeros_2d


def test_rows_sum_to_one() raises:
    """`softmax_row` output sums to 1."""
    var logits = [1.0, 2.0, 3.0]
    var p = softmax_row(logits)
    var s = 0.0
    for i in range(len(p)):
        s += p[i]
    assert_almost_equal(s, 1.0, atol=1e-12)


def test_monotonic() raises:
    """`softmax_row` preserves the ordering of the logits."""
    var logits = [1.0, 2.0, 3.0]
    var p = softmax_row(logits)
    assert_true(p[0] < p[1] and p[1] < p[2])


def test_stable_under_large_values() raises:
    """Equal large logits give a uniform distribution, not NaN from overflow."""
    var logits = [1000.0, 1000.0, 1000.0]
    var p = softmax_row(logits)
    for i in range(len(p)):
        assert_almost_equal(p[i], 1.0 / 3.0, atol=1e-12)


def test_empty_input() raises:
    """`softmax_row` of an empty list is empty."""
    var p = softmax_row(List[Float64]())
    assert_equal(len(p), 0)


def test_softmax_rows_each_row_sums_to_one() raises:
    """`softmax_rows` normalizes each row to sum 1 independently."""
    var scores = from_rows([[1.0, 2.0, 3.0], [0.0, 0.0, 1000.0]])
    var p = softmax_rows(scores)
    for r in range(p.rows):
        var s = 0.0
        for c in range(p.cols):
            s += p[r, c]
        assert_almost_equal(s, 1.0, atol=1e-12)
    # Second row's huge logit dominates.
    assert_almost_equal(p[1, 2], 1.0, atol=1e-9)


def test_softmax_rows_zero_columns() raises:
    """`softmax_rows` returns a zero-column tensor unchanged, no out-of-bounds read.
    """
    # Mirrors softmax_row's empty handling.
    var empty = zeros_2d(3, 0)
    var p = softmax_rows(empty)
    assert_equal(p.rows, 3)
    assert_equal(p.cols, 0)


def test_high_temperature_approaches_uniform() raises:
    """A very high temperature flattens the distribution toward uniform."""
    var logits = [1.0, 2.0, 3.0]
    var p = softmax_row_temperature(logits, 1e6)
    for i in range(len(p)):
        assert_almost_equal(p[i], 1.0 / 3.0, atol=1e-5)


def test_low_temperature_approaches_argmax() raises:
    """A very low temperature sharpens the distribution toward the argmax."""
    var logits = [1.0, 2.0, 3.0]
    var p = softmax_row_temperature(logits, 1e-3)
    assert_almost_equal(p[2], 1.0, atol=1e-9)
    assert_almost_equal(p[0], 0.0, atol=1e-9)


def test_temperature_rejects_nonpositive() raises:
    """`softmax_row_temperature` raises on a non-positive temperature."""
    var logits = [1.0, 2.0]
    with assert_raises(contains="temperature"):
        _ = softmax_row_temperature(logits, 0.0)


def test_extreme_low_temperature_is_stable() raises:
    """A near-zero temperature stays finite (argmax -> 1, rest -> 0)."""
    # It scales logits far past Float64's overflow point; subtracting the row max
    # before dividing keeps the result finite. The naive "divide then softmax"
    # form produces inf/inf = NaN here.
    var logits = [1000.0, 0.0]
    var p = softmax_row_temperature(logits, 1e-307)
    assert_almost_equal(p[0], 1.0, atol=1e-12)
    assert_almost_equal(p[1], 0.0, atol=1e-12)


def test_temperature_empty_input() raises:
    """`softmax_row_temperature` of an empty list is empty."""
    var p = softmax_row_temperature(List[Float64](), 0.5)
    assert_equal(len(p), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
