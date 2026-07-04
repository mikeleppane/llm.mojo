# Tests for the GELU backward — the scalar derivative and its elementwise VJP.
#
# GELU is elementwise, so its backward reduces to a per-element product with
# gelu_derivative. Two things to pin: the scalar derivative matches a central
# finite difference of gelu across a grid (including 0 and ±large, where the
# derivative approaches its 1 and 0 asymptotes), and gelu_rows_backward applies
# exactly that derivative entrywise.
#
# Finite-difference convention (D5, shared across this part's backward tests):
#   central diff h = 1e-5; tolerance |analytic - numeric| <= 1e-7 + 1e-5 * |n|.
# GELU's tanh form is smooth everywhere (no kink), so the finite difference is
# well behaved at every grid point.

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.nn.gelu import gelu, gelu_derivative, gelu_rows_backward
from llm.tensor.tensor2d import from_rows, zeros_2d


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    # D5 mixed tolerance |a - n| <= 1e-7 + 1e-5 * |n|.
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


def test_scalar_derivative_matches_finite_difference() raises:
    var grid = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]
    var h = 1e-5
    for i in range(len(grid)):
        var x = grid[i]
        var numeric = (gelu(x + h) - gelu(x - h)) / (2.0 * h)
        assert_grad_close(gelu_derivative(x), numeric)


def test_derivative_asymptotes() raises:
    # gelu'(x) -> 1 as x -> +inf and -> 0 as x -> -inf. At ±20 the tanh has
    # saturated, so the derivative is within a hair of its asymptote.
    assert_almost_equal(gelu_derivative(20.0), 1.0, atol=1e-6)
    assert_almost_equal(gelu_derivative(-20.0), 0.0, atol=1e-6)


def test_rows_backward_applies_scalar_derivative() raises:
    # gelu_rows_backward(x, d_out)[i, j] == d_out[i, j] * gelu_derivative(x[i, j]).
    var x = from_rows([[0.5, -1.0, 2.0], [-0.3, 0.8, -2.1]])
    var d_out = from_rows([[0.7, -0.2, 1.3], [0.1, 0.9, -1.1]])
    var got = gelu_rows_backward(x, d_out)
    for r in range(x.rows):
        for c in range(x.cols):
            assert_almost_equal(
                got[r, c],
                d_out[r, c] * gelu_derivative(x[r, c]),
                atol=1e-12,
            )


def test_shape_mismatch_raises() raises:
    var x = zeros_2d(2, 3)
    var d_out = zeros_2d(2, 4)
    with assert_raises(contains="shape mismatch"):
        _ = gelu_rows_backward(x, d_out)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
