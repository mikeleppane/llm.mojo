"""Finite-difference check for softmax_rows_backward — the row Jacobian's VJP.

softmax_rows_backward(W, dW) must return dS = dL/dscores given W = softmax(S) and
dW = dL/dW. We verify it as a vector-Jacobian product: pick a fixed cotangent dW,
form L(scores) = sum(dW * softmax_rows(scores)), and compare the analytic dS to a
central finite difference of L (h = 1e-5, tolerance
|analytic - numeric| <= 1e-7 + 1e-5 * |numeric|).
"""

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.tensor.ops import softmax_rows, softmax_rows_backward
from llm.tensor.tensor2d import Tensor2D, from_rows, zeros_2d


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    """Assert |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|.

    Pins the additive convention exactly, not assert_almost_equal's max()-form.

    Args:
        analytic: Backprop gradient.
        numeric: Finite-difference estimate.

    Raises:
        Error: If the two are not within the mixed tolerance.
    """
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


def projected_loss(scores: Tensor2D, cotangent: Tensor2D) -> Float64:
    """L = sum(cotangent * softmax_rows(scores)) — the VJP projection.

    Args:
        scores: Input scores.
        cotangent: Fixed cotangent dW.

    Returns:
        The scalar projected loss.
    """
    var w = softmax_rows(scores)
    var total = 0.0
    for i in range(w.rows):
        for j in range(w.cols):
            total += cotangent[i, j] * w[i, j]
    return total


def test_softmax_backward_matches_finite_difference() raises:
    """`softmax_rows_backward` matches a central finite difference of the VJP loss.
    """
    # Asymmetric 3x4 scores and cotangent: symmetric data could hide a sign or
    # axis error, so nothing here is symmetric.
    var scores = from_rows(
        [
            [0.5, -1.0, 2.0, 0.25],
            [-0.3, 0.8, -2.1, 1.4],
            [1.1, 0.0, -0.7, 0.9],
        ]
    )
    var cotangent = from_rows(
        [
            [0.7, -0.2, 1.3, -0.5],
            [0.1, 0.9, -1.1, 0.4],
            [-0.6, 0.3, 0.2, -0.8],
        ]
    )
    var w = softmax_rows(scores)
    var analytic = softmax_rows_backward(w, cotangent)

    var h = 1e-5
    for i in range(scores.rows):
        for j in range(scores.cols):
            var plus = scores.copy()
            plus[i, j] = plus[i, j] + h
            var minus = scores.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                projected_loss(plus, cotangent)
                - projected_loss(minus, cotangent)
            ) / (2.0 * h)
            assert_grad_close(analytic[i, j], numeric)


def test_uniform_weights_row() raises:
    """On a uniform row the output has mean zero: dS_j = p (dW_j - mean(dW))."""
    # The uniform row is softmax's maximum-entropy point (best-conditioned
    # Jacobian), a clean analytic check independent of the finite-difference path.
    var w = from_rows([[0.25, 0.25, 0.25, 0.25]])
    var cotangent = from_rows([[1.0, -2.0, 3.0, -0.5]])
    var ds = softmax_rows_backward(w, cotangent)
    var row_sum = 0.0
    for j in range(ds.cols):
        row_sum += ds[0, j]  # ds is [1, 4]
    # Each entry is 0.25 * (dW_j - mean(dW)); the entries sum to 0.25 * 0 = 0.
    assert_almost_equal(row_sum, 0.0, atol=1e-12)
    var mean_cot = (1.0 - 2.0 + 3.0 - 0.5) / 4.0
    for j in range(4):
        assert_almost_equal(
            ds[0, j], 0.25 * (cotangent[0, j] - mean_cot), atol=1e-12
        )


def test_shape_mismatch_raises() raises:
    """`softmax_rows_backward` raises when W and dW shapes differ."""
    var w = zeros_2d(3, 4)
    var bad = zeros_2d(3, 5)
    with assert_raises(contains="shape mismatch"):
        _ = softmax_rows_backward(w, bad)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
