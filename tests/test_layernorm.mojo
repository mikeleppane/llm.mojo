"""Tests for LayerNorm: per-row normalization with biased variance (÷C) and eps 1e-5."""

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.nn.layernorm import LayerNorm
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, from_rows, ones_2d


def ln_input() raises -> Tensor2D:
    """Return the shared 3x4 fixture the oracle goldens were computed from."""
    return from_rows(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [-1.0, 0.0, 2.0, 5.0]]
    )


def test_ones_zeros_row_mean_zero_std_one() raises:
    """With weight=1, bias=0, each output row has mean ~0 and population std ~1.
    """
    # eps makes the std slightly below 1 (hence the loose std tolerance); the
    # mean is essentially exact.
    var ln = LayerNorm.init_default(4)
    var y = ln.forward(ln_input())
    for r in range(y.rows):
        var mean = 0.0
        for c in range(y.cols):
            mean += y[r, c]
        mean = mean / 4.0
        assert_almost_equal(mean, 0.0, atol=1e-12)
        var var_ = 0.0
        for c in range(y.cols):
            var d = y[r, c] - mean
            var_ += d * d
        var_ = var_ / 4.0  # population (biased) variance of the output row
        assert_almost_equal(var_**0.5, 1.0, atol=1e-4)


def test_biased_variance_golden() raises:
    """Freeze the biased-variance (÷C) golden so a ÷(C-1) regression fails."""
    # Golden from tests/oracles/nn_reference.py ("LayerNorm 3x4, weight=ones
    # bias=zeros"). If the code divides by C-1 instead, row 0 becomes
    # [-1.1618915181928988, -0.3872971727309663, +sym...] and this test fails.
    var ln = LayerNorm.init_default(4)
    var y = ln.forward(ln_input())
    assert_almost_equal(y[0, 0], -1.3416354199689269, atol=1e-12)
    assert_almost_equal(y[0, 1], -0.447211806656309, atol=1e-12)
    assert_almost_equal(y[0, 2], 0.447211806656309, atol=1e-12)
    assert_almost_equal(y[0, 3], 1.3416354199689269, atol=1e-12)
    assert_almost_equal(y[1, 0], -1.3416394448610998, atol=1e-12)
    assert_almost_equal(y[1, 3], 1.3416394448610998, atol=1e-12)
    assert_almost_equal(y[2, 0], -1.0910884120486357, atol=1e-12)
    assert_almost_equal(y[2, 1], -0.6546530472291815, atol=1e-12)
    assert_almost_equal(y[2, 2], 0.21821768240972714, atol=1e-12)
    assert_almost_equal(y[2, 3], 1.52752377686809, atol=1e-12)
    # Explicitly reject the unbiased result for row 0, so a ÷(C-1) regression
    # can't slip through under a loose tolerance.
    assert_true(abs(y[0, 0] - (-1.1618915181928988)) > 1e-3)


def test_weight_and_bias_applied_per_column() raises:
    """Per-column weight and bias are applied after normalization."""
    # Golden from tests/oracles/nn_reference.py ("weight/bias per-column").
    # weight = [0.5, 1.0, 1.5, 2.0], bias = [0.1, -0.1, 0.2, -0.2].
    var weight = from_rows([[0.5, 1.0, 1.5, 2.0]])
    var bias = from_rows([[0.1, -0.1, 0.2, -0.2]])
    var ln = LayerNorm(Parameter(weight^), Parameter(bias^))
    var y = ln.forward(ln_input())
    assert_almost_equal(y[0, 0], -0.5708177099844635, atol=1e-12)
    assert_almost_equal(y[0, 1], -0.547211806656309, atol=1e-12)
    assert_almost_equal(y[0, 2], 0.8708177099844634, atol=1e-12)
    assert_almost_equal(y[0, 3], 2.4832708399378536, atol=1e-12)
    assert_almost_equal(y[2, 3], 2.85504755373618, atol=1e-12)


def test_constant_row_maps_to_bias() raises:
    """A constant row (variance 0) maps exactly to the bias via the eps path."""
    # (x - mean) is 0 everywhere, so the output is exactly the bias; the eps path
    # keeps 1/sqrt(0 + eps) finite instead of dividing by zero.
    var weight = ones_2d(1, 3)
    var bias = from_rows([[0.5, -0.5, 2.0]])
    var ln = LayerNorm(Parameter(weight^), Parameter(bias^))
    var y = ln.forward(from_rows([[3.0, 3.0, 3.0]]))
    assert_almost_equal(y[0, 0], 0.5, atol=1e-12)
    assert_almost_equal(y[0, 1], -0.5, atol=1e-12)
    assert_almost_equal(y[0, 2], 2.0, atol=1e-12)


def test_bias_shape_mismatch_raises() raises:
    """forward raises when weight and bias are not both [1, C]."""
    # A bias narrower than the weight would otherwise read out of bounds at the
    # per-column bias add.
    var weight = from_rows([[1.0, 1.0, 1.0, 1.0]])  # C = 4
    var bad_bias = from_rows([[0.0, 0.0, 0.0]])  # C = 3, mismatched
    var ln = LayerNorm(Parameter(weight^), Parameter(bad_bias^))
    with assert_raises(contains="bias"):
        _ = ln.forward(ln_input())


def test_init_default_shape_and_values() raises:
    """init_default builds [1, C] weight of ones and bias of zeros."""
    var ln = LayerNorm.init_default(4)
    assert_equal(ln.weight.value.rows, 1)
    assert_equal(ln.weight.value.cols, 4)
    for c in range(4):
        assert_almost_equal(ln.weight.value[0, c], 1.0, atol=1e-15)
        assert_almost_equal(ln.bias.value[0, c], 0.0, atol=1e-15)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
