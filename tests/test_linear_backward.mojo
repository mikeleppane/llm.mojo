# Finite-difference and accumulation tests for Linear.backward.
#
# Linear is affine, so three gradients must all check out: dL/dx (returned),
# dL/dW and dL/db (accumulated into the Parameters). Each gets its own
# perturbation loop against the projected scalar loss L = sum(cotangent ⊙ y).
# Separately, the accumulation contract is pinned exactly: two backward calls
# without a zero_grad() between them double the parameter grads (same floats,
# same order — exact equality is legitimate here), and zero_grad() resets them.
#
# Finite-difference convention (D5, shared across this part's backward tests):
#   L = sum(cotangent ⊙ y); central diff h = 1e-5; tolerance
#   |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|.

from std.testing import assert_almost_equal, assert_true, TestSuite

from llm.nn.linear import Linear, LinearCache
from llm.nn.parameter import Parameter
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


def make_layer() raises -> Linear:
    # weight [out=2, in=4], bias [1, 2]; asymmetric so no axis error can hide.
    var w = from_rows([[0.5, -1.0, 2.0, 0.25], [-0.3, 0.8, -2.1, 1.4]])
    var b = from_rows([[0.2, -0.7]])
    return Linear(Parameter(w^), Parameter(b^))


def sample_input() raises -> Tensor2D:
    # [N=3, in=4].
    return from_rows(
        [[1.0, 0.5, -1.0, 0.3], [0.2, -0.4, 0.9, -1.1], [-0.7, 1.2, 0.1, 0.6]]
    )


def cotangent() raises -> Tensor2D:
    # Fixed asymmetric d_out [N=3, out=2].
    return from_rows([[0.7, -0.2], [0.1, 0.9], [-0.6, 0.3]])


def projected(layer: Linear, x: Tensor2D, cot: Tensor2D) raises -> Float64:
    var y = layer.forward(x)
    var total = 0.0
    for i in range(y.rows):
        for j in range(y.cols):
            total += cot[i, j] * y[i, j]
    return total


def test_d_x_matches_finite_difference() raises:
    var layer = make_layer()
    var x = sample_input()
    var cot = cotangent()
    var fwd = layer.forward_cached(x)
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
    var layer = make_layer()
    var x = sample_input()
    var cot = cotangent()
    layer.weight.zero_grad()
    layer.bias.zero_grad()
    var fwd = layer.forward_cached(x)
    _ = layer.backward(fwd.cache, cot)

    var h = 1e-5
    for o in range(layer.weight.value.rows):
        for i in range(layer.weight.value.cols):
            var w_plus = layer.weight.value.copy()
            w_plus[o, i] = w_plus[o, i] + h
            var layer_plus = Linear(
                Parameter(w_plus^), Parameter(layer.bias.value.copy())
            )
            var w_minus = layer.weight.value.copy()
            w_minus[o, i] = w_minus[o, i] - h
            var layer_minus = Linear(
                Parameter(w_minus^), Parameter(layer.bias.value.copy())
            )
            var numeric = (
                projected(layer_plus, x, cot) - projected(layer_minus, x, cot)
            ) / (2.0 * h)
            assert_grad_close(layer.weight.grad[o, i], numeric)


def test_d_bias_matches_finite_difference() raises:
    var layer = make_layer()
    var x = sample_input()
    var cot = cotangent()
    layer.weight.zero_grad()
    layer.bias.zero_grad()
    var fwd = layer.forward_cached(x)
    _ = layer.backward(fwd.cache, cot)

    var h = 1e-5
    for o in range(layer.bias.value.cols):
        var b_plus = layer.bias.value.copy()
        b_plus[0, o] = b_plus[0, o] + h
        var layer_plus = Linear(
            Parameter(layer.weight.value.copy()), Parameter(b_plus^)
        )
        var b_minus = layer.bias.value.copy()
        b_minus[0, o] = b_minus[0, o] - h
        var layer_minus = Linear(
            Parameter(layer.weight.value.copy()), Parameter(b_minus^)
        )
        var numeric = (
            projected(layer_plus, x, cot) - projected(layer_minus, x, cot)
        ) / (2.0 * h)
        assert_grad_close(layer.bias.grad[0, o], numeric)


def test_backward_accumulates_exactly() raises:
    # Two backward calls without a zero_grad() between them must exactly double
    # the parameter grads — same floats added in the same order, so exact
    # equality is the right assertion, not a tolerance. This is the contract a
    # tied weight (one Parameter, two backward paths) later depends on.
    var layer = make_layer()
    var x = sample_input()
    var cot = cotangent()
    layer.weight.zero_grad()
    layer.bias.zero_grad()
    var fwd = layer.forward_cached(x)
    _ = layer.backward(fwd.cache, cot)
    var w_once = layer.weight.grad.copy()
    var b_once = layer.bias.grad.copy()
    _ = layer.backward(fwd.cache, cot)
    for o in range(layer.weight.value.rows):
        for i in range(layer.weight.value.cols):
            assert_equal_exact(layer.weight.grad[o, i], 2.0 * w_once[o, i])
    for o in range(layer.bias.value.cols):
        assert_equal_exact(layer.bias.grad[0, o], 2.0 * b_once[0, o])


def test_zero_grad_resets() raises:
    var layer = make_layer()
    var x = sample_input()
    var cot = cotangent()
    var fwd = layer.forward_cached(x)
    _ = layer.backward(fwd.cache, cot)
    layer.weight.zero_grad()
    layer.bias.zero_grad()
    for o in range(layer.weight.value.rows):
        for i in range(layer.weight.value.cols):
            assert_almost_equal(layer.weight.grad[o, i], 0.0, atol=1e-15)
    for o in range(layer.bias.value.cols):
        assert_almost_equal(layer.bias.grad[0, o], 0.0, atol=1e-15)


def assert_equal_exact(a: Float64, b: Float64) raises:
    # The accumulation test wants bit-for-bit equality (same additions, same
    # order), not a tolerance — a doubled grad that is off by any ulp is a bug.
    assert_true(
        a == b,
        String("expected exact equality, got ")
        + String(a)
        + " vs "
        + String(b),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
