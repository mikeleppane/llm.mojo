# Tests for stable softmax and its temperature variant.
#
# The stability test is the whole point: a naive exp(1000) overflows to NaN; the
# max-subtracting version returns a clean uniform distribution. The temperature
# limits (high -> uniform, low -> argmax) also re-prove stability, since a small
# temperature scales logits into overflow territory before the max subtraction.

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.tensor.ops import softmax_row, softmax_rows, softmax_row_temperature
from llm.tensor.tensor2d import from_rows


def test_rows_sum_to_one() raises:
    var p = softmax_row([1.0, 2.0, 3.0])
    var s = 0.0
    for i in range(len(p)):
        s += p[i]
    assert_almost_equal(s, 1.0, atol=1e-12)


def test_monotonic() raises:
    var p = softmax_row([1.0, 2.0, 3.0])
    assert_true(p[0] < p[1] and p[1] < p[2])


def test_stable_under_large_values() raises:
    var p = softmax_row([1000.0, 1000.0, 1000.0])
    for i in range(len(p)):
        assert_almost_equal(p[i], 1.0 / 3.0, atol=1e-12)


def test_empty_input() raises:
    var p = softmax_row(List[Float64]())
    assert_equal(len(p), 0)


def test_softmax_rows_each_row_sums_to_one() raises:
    var scores = from_rows([[1.0, 2.0, 3.0], [0.0, 0.0, 1000.0]])
    var p = softmax_rows(scores)
    for r in range(p.rows):
        var s = 0.0
        for c in range(p.cols):
            s += p[r, c]
        assert_almost_equal(s, 1.0, atol=1e-12)
    # Second row's huge logit dominates.
    assert_almost_equal(p[1, 2], 1.0, atol=1e-9)


def test_high_temperature_approaches_uniform() raises:
    var p = softmax_row_temperature([1.0, 2.0, 3.0], 1e6)
    for i in range(len(p)):
        assert_almost_equal(p[i], 1.0 / 3.0, atol=1e-5)


def test_low_temperature_approaches_argmax() raises:
    var p = softmax_row_temperature([1.0, 2.0, 3.0], 1e-3)
    assert_almost_equal(p[2], 1.0, atol=1e-9)
    assert_almost_equal(p[0], 0.0, atol=1e-9)


def test_temperature_rejects_nonpositive() raises:
    with assert_raises(contains="temperature"):
        _ = softmax_row_temperature([1.0, 2.0], 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
