# Tests for the batched cross-entropy loss and its gradient.
#
# cross_entropy_rows is the mean of cross_entropy_one over N rows, and its
# backward is (softmax - onehot) / N. Three properties matter beyond the raw
# finite-difference check: the batched loss must not drift from the per-row one
# it is built on; the backward's rows must each sum to 0 (the softmax-minus-
# onehot structure); and the 1/N mean factor must be present (doubling the rows
# halves the per-row gradient).
#
# Finite-difference convention (D5, shared across this part's backward tests):
#   L = sum(d_out * f(x)); central diff h = 1e-5; tolerance
#   |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|. Here the loss is already a
#   scalar, so no projection cotangent is needed — we differentiate L directly.

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.tensor.ops import (
    cross_entropy_one,
    cross_entropy_rows,
    cross_entropy_rows_backward,
)
from llm.tensor.tensor2d import Tensor2D, from_rows


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    # D5 mixed tolerance |a - n| <= 1e-7 + 1e-5 * |n|.
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


def sample_logits() raises -> Tensor2D:
    # Asymmetric 3x4 logits (V = 4), so no axis or sign error can hide.
    return from_rows(
        [
            [0.5, -1.0, 2.0, 0.25],
            [-0.3, 0.8, -2.1, 1.4],
            [1.1, 0.0, -0.7, 0.9],
        ]
    )


def test_agrees_with_mean_of_cross_entropy_one() raises:
    # The batched loss must equal the hand-averaged per-row loss exactly (same
    # summation), or the two will drift as later parts lean on the batched form.
    var logits = sample_logits()
    var targets = [2, 3, 0]
    var manual = 0.0
    for i in range(logits.rows):
        var row = List[Float64]()
        for j in range(logits.cols):
            row.append(logits[i, j])
        manual += cross_entropy_one(row, targets[i])
    manual = manual / Float64(logits.rows)
    assert_almost_equal(cross_entropy_rows(logits, targets), manual, atol=1e-12)


def test_backward_matches_finite_difference() raises:
    var logits = sample_logits()
    var targets = [2, 3, 0]
    var analytic = cross_entropy_rows_backward(logits, targets)

    var h = 1e-5
    for i in range(logits.rows):
        for j in range(logits.cols):
            var plus = logits.copy()
            plus[i, j] = plus[i, j] + h
            var minus = logits.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                cross_entropy_rows(plus, targets)
                - cross_entropy_rows(minus, targets)
            ) / (2.0 * h)
            assert_grad_close(analytic[i, j], numeric)


def test_backward_rows_sum_to_zero() raises:
    # Each row is (softmax - onehot)/N; softmax sums to 1 and onehot sums to 1,
    # so every row of the gradient sums to exactly 0.
    var logits = sample_logits()
    var targets = [2, 3, 0]
    var g = cross_entropy_rows_backward(logits, targets)
    for i in range(g.rows):
        var row_sum = 0.0
        for j in range(g.cols):
            row_sum += g[i, j]
        assert_almost_equal(row_sum, 0.0, atol=1e-12)


def test_mean_factor_halves_gradient_when_rows_double() raises:
    # The 1/N mean factor: stacking the same row twice (N: 1 -> 2) must halve the
    # per-row gradient. This pins the mean (not sum) reduction — a sum loss would
    # leave the per-row gradient unchanged.
    var one = from_rows([[0.5, -1.0, 2.0, 0.25]])
    var two = from_rows([[0.5, -1.0, 2.0, 0.25], [0.5, -1.0, 2.0, 0.25]])
    var g1 = cross_entropy_rows_backward(one, [2])
    var g2 = cross_entropy_rows_backward(two, [2, 2])
    for j in range(4):
        assert_almost_equal(g2[0, j], 0.5 * g1[0, j], atol=1e-12)
        assert_almost_equal(g2[1, j], 0.5 * g1[0, j], atol=1e-12)


def test_bad_target_raises() raises:
    var logits = sample_logits()
    with assert_raises(contains="target out of range"):
        _ = cross_entropy_rows(logits, [2, 3, 9])
    with assert_raises(contains="target out of range"):
        _ = cross_entropy_rows_backward(logits, [2, 3, 9])


def test_length_mismatch_raises() raises:
    var logits = sample_logits()
    with assert_raises(contains="must equal logits rows"):
        _ = cross_entropy_rows(logits, [2, 3])
    with assert_raises(contains="must equal logits rows"):
        _ = cross_entropy_rows_backward(logits, [2, 3])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
