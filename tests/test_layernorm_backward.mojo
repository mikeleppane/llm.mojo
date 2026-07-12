"""Tests for LayerNorm backward: finite-difference checks on the three-term dx, dgamma, dbeta, plus an analytic orthogonality test (dx is orthogonal to the ones vector exactly and to the normalized row x-hat up to eps). Finite-difference convention: L = sum(cotangent * y); central diff h = 1e-5; mixed absolute/relative tolerance 1e-7 + 1e-5*|numeric|."""

from std.testing import assert_almost_equal, assert_true, TestSuite

from llm.nn.layernorm import LayerNorm
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, from_rows


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    """Assert |analytic - numeric| <= 1e-7 + 1e-5 * |numeric| (mixed absolute/relative tolerance).
    """
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


def make_layer() raises -> LayerNorm:
    """Build a LayerNorm with non-trivial weight and bias (C = 4) so a = d_out ⊙ γ has a nonzero mean; a weight of ones would make the projection terms degenerate.
    """
    var weight = from_rows([[0.5, 1.0, 1.5, 2.0]])
    var bias = from_rows([[0.1, -0.1, 0.2, -0.2]])
    return LayerNorm(Parameter(weight^), Parameter(bias^))


def sample_input() raises -> Tensor2D:
    """A [N=3, C=4] input, asymmetric with distinct per-row variances."""
    return from_rows(
        [[1.0, 2.0, 3.0, 4.0], [2.0, -4.0, 6.0, -8.0], [-1.0, 0.0, 2.0, 5.0]]
    )


def cotangent() raises -> Tensor2D:
    """A fixed asymmetric d_out [N=3, C=4], each row with a nonzero mean."""
    return from_rows(
        [[0.7, -0.2, 1.3, -0.5], [0.1, 0.9, -1.1, 0.4], [-0.6, 0.3, 0.2, -0.8]]
    )


def projected(layer: LayerNorm, x: Tensor2D, cot: Tensor2D) raises -> Float64:
    var y = layer.forward(x)
    var total = 0.0
    for i in range(y.rows):
        for j in range(y.cols):
            total += cot[i, j] * y[i, j]
    return total


def test_d_x_matches_finite_difference() raises:
    """Backward's dx matches a central finite difference at every input element.
    """
    var layer = make_layer()
    var x = sample_input()
    var cot = cotangent()
    var fwd = layer.forward_cached(x.copy())
    var d_x = layer.backward(fwd.cache, cot)

    var h = 1e-5
    for i in range(x.rows):
        for j in range(x.cols):
            var plus = x.copy()
            plus[i, j] = plus[i, j] + h
            var minus = x.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                projected(layer, plus, cot) - projected(layer, minus, cot)
            ) / (2.0 * h)
            assert_grad_close(d_x[i, j], numeric)


def test_d_weight_matches_finite_difference() raises:
    """Backward's dgamma matches a central finite difference at every weight element.
    """
    var layer = make_layer()
    var x = sample_input()
    var cot = cotangent()
    layer.weight.zero_grad()
    layer.bias.zero_grad()
    var fwd = layer.forward_cached(x.copy())
    _ = layer.backward(fwd.cache, cot)

    var h = 1e-5
    for j in range(layer.weight.value.cols):
        var w_plus = layer.weight.value.copy()
        w_plus[0, j] = w_plus[0, j] + h
        var layer_plus = LayerNorm(
            Parameter(w_plus^), Parameter(layer.bias.value.copy())
        )
        var w_minus = layer.weight.value.copy()
        w_minus[0, j] = w_minus[0, j] - h
        var layer_minus = LayerNorm(
            Parameter(w_minus^), Parameter(layer.bias.value.copy())
        )
        var numeric = (
            projected(layer_plus, x, cot) - projected(layer_minus, x, cot)
        ) / (2.0 * h)
        assert_grad_close(layer.weight.grad[0, j], numeric)


def test_d_bias_matches_finite_difference() raises:
    """Backward's dbeta matches a central finite difference at every bias element.
    """
    var layer = make_layer()
    var x = sample_input()
    var cot = cotangent()
    layer.weight.zero_grad()
    layer.bias.zero_grad()
    var fwd = layer.forward_cached(x.copy())
    _ = layer.backward(fwd.cache, cot)

    var h = 1e-5
    for j in range(layer.bias.value.cols):
        var b_plus = layer.bias.value.copy()
        b_plus[0, j] = b_plus[0, j] + h
        var layer_plus = LayerNorm(
            Parameter(layer.weight.value.copy()), Parameter(b_plus^)
        )
        var b_minus = layer.bias.value.copy()
        b_minus[0, j] = b_minus[0, j] - h
        var layer_minus = LayerNorm(
            Parameter(layer.weight.value.copy()), Parameter(b_minus^)
        )
        var numeric = (
            projected(layer_plus, x, cot) - projected(layer_minus, x, cot)
        ) / (2.0 * h)
        assert_grad_close(layer.bias.grad[0, j], numeric)


def test_d_x_orthogonal_to_ones_and_xhat() raises:
    """Backward's dx is orthogonal to the ones vector (sum(dx) = 0 exactly) and to x̂ (sum(dx ⊙ x̂) = 0 up to eps); dropping either subtracted term breaks one orthogonality by an O(1) amount, caught here without any finite difference.
    """
    var layer = make_layer()
    var x = sample_input()
    var cot = cotangent()
    var fwd = layer.forward_cached(x.copy())
    var d_x = layer.backward(fwd.cache, cot)

    for r in range(x.rows):
        var mean = fwd.cache.mean[r]
        var rstd = fwd.cache.rstd[r]
        var sum_dx = 0.0
        var sum_dx_xhat = 0.0
        for j in range(x.cols):
            var xhat = (x[r, j] - mean) * rstd
            sum_dx += d_x[r, j]
            sum_dx_xhat += d_x[r, j] * xhat
        # Ones orthogonality is exact (mean projection is subtracted cleanly).
        assert_almost_equal(sum_dx, 0.0, atol=1e-10)
        # x̂ orthogonality holds up to eps: sum(dx⊙x̂) = r·C·mean(a⊙x̂)·eps/(v+eps),
        # which is ~1e-5 here — far below the O(1) a dropped term would give.
        assert_almost_equal(sum_dx_xhat, 0.0, atol=1e-3)


def test_backward_accumulates_exactly() raises:
    """Two backward passes double the weight and bias gradients exactly (grads accumulate).
    """
    var layer = make_layer()
    var x = sample_input()
    var cot = cotangent()
    layer.weight.zero_grad()
    layer.bias.zero_grad()
    var fwd = layer.forward_cached(x.copy())
    _ = layer.backward(fwd.cache, cot)
    var w_once = layer.weight.grad.copy()
    var b_once = layer.bias.grad.copy()
    _ = layer.backward(fwd.cache, cot)
    for j in range(layer.weight.value.cols):
        assert_true(layer.weight.grad[0, j] == 2.0 * w_once[0, j])
        assert_true(layer.bias.grad[0, j] == 2.0 * b_once[0, j])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
