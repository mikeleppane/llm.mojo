"""Tests for Linear: the affine layer y = x @ W^T + b, plus init_random determinism."""

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    TestSuite,
)

from llm.nn.linear import Linear
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import from_rows, zeros_2d
from llm.utils.random import Rng


def make_linear() raises -> Linear:
    """Build a Linear with weight [out=2, in=3], bias [1, 2] matching the nn_reference.py golden.
    """
    var w = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = from_rows([[0.5, -0.5]])
    return Linear(Parameter(w^), Parameter(b^))


def test_forward_hand_computed() raises:
    """forward matches the hand-computed y = x @ W^T + b golden."""
    # Golden from tests/oracles/nn_reference.py ("Linear [out,in]=[2,3]").
    var layer = make_linear()
    var x = from_rows([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])  # [N=2, in=3]
    var y = layer.forward(x)  # [N=2, out=2]
    assert_equal(y.rows, 2)
    assert_equal(y.cols, 2)
    assert_almost_equal(y[0, 0], -1.5, atol=1e-12)
    assert_almost_equal(y[0, 1], -2.5, atol=1e-12)
    assert_almost_equal(y[1, 0], 4.5, atol=1e-12)
    assert_almost_equal(y[1, 1], 12.5, atol=1e-12)


def test_bias_added_to_every_row() raises:
    """With a zero weight, forward broadcasts the bias to every row (catches a wrong-axis bias).
    """
    var w = zeros_2d(2, 3)
    var b = from_rows([[7.0, -4.0]])
    var layer = Linear(Parameter(w^), Parameter(b^))
    var x = from_rows([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0], [-1.0, 0.0, 5.0]])
    var y = layer.forward(x)
    for r in range(y.rows):
        assert_almost_equal(y[r, 0], 7.0, atol=1e-12)
        assert_almost_equal(y[r, 1], -4.0, atol=1e-12)


def test_shape_mismatch_raises() raises:
    """forward raises when the input feature count does not match in."""
    var layer = make_linear()  # expects in=3
    var x = from_rows([[1.0, 2.0]])  # in=2, wrong
    with assert_raises(contains="shape mismatch"):
        _ = layer.forward(x)


def test_bias_shape_mismatch_raises() raises:
    """forward raises when the bias is not [1, out]."""
    # A bias whose width doesn't match out (the weight's row count) would
    # otherwise read out of bounds when broadcasting the bias.
    var w = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # out=2
    var bad_bias = from_rows([[0.5]])  # [1, 1], should be [1, 2]
    var layer = Linear(Parameter(w^), Parameter(bad_bias^))
    var x = from_rows([[1.0, 0.0, -1.0]])
    with assert_raises(contains="bias"):
        _ = layer.forward(x)


def test_init_random_is_deterministic() raises:
    """The same seed replays identical [out, in] weights (downstream model tests rely on this).
    """
    var rng_a = Rng(1234)
    var rng_b = Rng(1234)
    var la = Linear.init_random(rng_a, 4, 5)
    var lb = Linear.init_random(rng_b, 4, 5)
    assert_equal(la.weight.value.rows, 5)  # [out, in]
    assert_equal(la.weight.value.cols, 4)
    for r in range(5):
        for c in range(4):
            assert_almost_equal(
                la.weight.value[r, c], lb.weight.value[r, c], atol=1e-15
            )


def test_init_random_bias_is_zero() raises:
    """init_random initializes the bias to [1, out] zeros."""
    var rng = Rng(7)
    var layer = Linear.init_random(rng, 4, 5)
    assert_equal(layer.bias.value.rows, 1)  # [1, out]
    assert_equal(layer.bias.value.cols, 5)
    for c in range(5):
        assert_almost_equal(layer.bias.value[0, c], 0.0, atol=1e-15)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
