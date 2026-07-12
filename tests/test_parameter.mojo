"""Tests for Parameter — the value + gradient pair every layer owns.

Pin the invariants the optimizer and backward pass rely on: the grad exists,
matches the value's shape, starts at zero, and zero_grad clears it without
touching the value.
"""

from std.testing import assert_almost_equal, assert_equal, TestSuite

from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import from_rows


def test_grad_zeros_with_value_shape() raises:
    """A new Parameter's grad mirrors the value's shape and starts all-zero."""
    var w = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var p = Parameter(w^)
    # grad mirrors the value's shape...
    assert_equal(p.grad.rows, 2)
    assert_equal(p.grad.cols, 3)
    # ...and every entry starts at zero.
    for r in range(p.grad.rows):
        for c in range(p.grad.cols):
            assert_almost_equal(p.grad[r, c], 0.0, atol=1e-15)


def test_value_survives_round_trip() raises:
    """A Parameter preserves the value tensor it was constructed from."""
    var w = from_rows([[1.5, -2.5], [0.0, 7.0]])
    var p = Parameter(w^)
    assert_almost_equal(p.value[0, 0], 1.5, atol=1e-15)
    assert_almost_equal(p.value[0, 1], -2.5, atol=1e-15)
    assert_almost_equal(p.value[1, 0], 0.0, atol=1e-15)
    assert_almost_equal(p.value[1, 1], 7.0, atol=1e-15)


def test_zero_grad_clears_a_dirtied_grad() raises:
    """`zero_grad` resets the grad to zero and leaves the value untouched."""
    var w = from_rows([[1.0, 2.0]])
    var p = Parameter(w^)
    # Dirty the grad the way a backward pass eventually would.
    p.grad[0, 0] = 9.0
    p.grad[0, 1] = -3.0
    p.zero_grad()
    assert_almost_equal(p.grad[0, 0], 0.0, atol=1e-15)
    assert_almost_equal(p.grad[0, 1], 0.0, atol=1e-15)
    # zero_grad touches only the grad, never the value.
    assert_almost_equal(p.value[0, 0], 1.0, atol=1e-15)
    assert_almost_equal(p.value[0, 1], 2.0, atol=1e-15)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
