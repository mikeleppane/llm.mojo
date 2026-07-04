# Tests for dropout backward — the cached-mask VJP.
#
# Dropout is linear once the mask is fixed (output = (mask * inv_keep) ⊙ x), so
# backward must reuse the SAME mask the forward drew. The tests pin exactly that:
# the forward output equals x scaled by the mask; backward equals d_out scaled by
# the same mask; a central finite difference of the mask-fixed forward matches
# backward to machine precision (linear -> exact, not the loose D5 tolerance);
# eval / p == 0 backward is the identity; and eval mode consumes no rng draws.

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.nn.dropout import dropout_backward, dropout_cached
from llm.tensor.tensor2d import Tensor2D, from_rows, zeros_2d
from llm.utils.random import Rng


def sample_input() raises -> Tensor2D:
    # [N=3, C=4], asymmetric.
    return from_rows(
        [
            [1.0, 0.5, -1.0, 0.3],
            [0.2, -0.4, 0.9, -1.1],
            [-0.7, 1.2, 0.1, 0.6],
        ]
    )


def forward_with_mask(x: Tensor2D, mask: Tensor2D, p: Float64) -> Tensor2D:
    # The dropout forward with a *given* mask: output = (mask * inv_keep) ⊙ x.
    # Used to finite-diff backward against a fixed mask (dropout draws a fresh
    # mask each call, so this is the only way to hold it constant).
    var inv_keep = 1.0 / (1.0 - p)
    var out = zeros_2d(x.rows, x.cols)
    for r in range(x.rows):
        for c in range(x.cols):
            out[r, c] = x[r, c] * mask[r, c] * inv_keep
    return out^


def test_forward_uses_the_returned_mask() raises:
    # dropout_cached's output must be exactly x * mask * inv_keep — the mask it
    # returns is the one it actually applied, not a decoration.
    var rng = Rng(42)
    var x = sample_input()
    var p = 0.3
    var res = dropout_cached(x, p, True, rng)
    var inv_keep = 1.0 / (1.0 - p)
    for r in range(x.rows):
        for c in range(x.cols):
            assert_almost_equal(
                res.output[r, c],
                x[r, c] * res.mask[r, c] * inv_keep,
                atol=1e-12,
            )
            # The mask is a strict 0/1 indicator.
            assert_true(res.mask[r, c] == 0.0 or res.mask[r, c] == 1.0)


def test_backward_matches_finite_difference_through_cached_mask() raises:
    # Given the mask, dropout is linear in x, so the central difference of the
    # mask-fixed forward equals backward to machine precision. A tight tolerance
    # is legitimate here — and a backward that drew a fresh mask instead of
    # reusing this one would miss it by whole elements.
    var rng = Rng(7)
    var x = sample_input()
    var p = 0.4
    var res = dropout_cached(x, p, True, rng)
    var cotangent = from_rows(
        [[0.7, -0.2, 1.3, -0.5], [0.1, 0.9, -1.1, 0.4], [-0.6, 0.3, 0.2, -0.8]]
    )
    var analytic = dropout_backward(res.mask, p, cotangent)

    var h = 1e-5
    for i in range(x.rows):
        for j in range(x.cols):
            var plus = x.copy()
            plus[i, j] = plus[i, j] + h
            var minus = x.copy()
            minus[i, j] = minus[i, j] - h
            var y_plus = forward_with_mask(plus, res.mask, p)
            var y_minus = forward_with_mask(minus, res.mask, p)
            var numeric = 0.0
            for a in range(x.rows):
                for b in range(x.cols):
                    numeric += (
                        cotangent[a, b] * (y_plus[a, b] - y_minus[a, b])
                    ) / (2.0 * h)
            assert_almost_equal(analytic[i, j], numeric, atol=1e-9)


def test_backward_scales_by_inv_keep() raises:
    # d_x = d_out * mask / (1 - p), entry for entry.
    var rng = Rng(123)
    var x = sample_input()
    var p = 0.25
    var res = dropout_cached(x, p, True, rng)
    var d_out = sample_input()  # any tensor of the right shape
    var d_x = dropout_backward(res.mask, p, d_out)
    var inv_keep = 1.0 / (1.0 - p)
    for r in range(x.rows):
        for c in range(x.cols):
            assert_almost_equal(
                d_x[r, c], d_out[r, c] * res.mask[r, c] * inv_keep, atol=1e-12
            )


def test_eval_and_p_zero_backward_is_identity() raises:
    # Eval mode (or p == 0): the mask is all ones and inv_keep = 1, so backward
    # returns d_out unchanged.
    var rng = Rng(1)
    var x = sample_input()
    var res_eval = dropout_cached(x, 0.5, False, rng)  # eval mode
    var d_out = from_rows(
        [[0.7, -0.2, 1.3, -0.5], [0.1, 0.9, -1.1, 0.4], [-0.6, 0.3, 0.2, -0.8]]
    )
    var d_x = dropout_backward(res_eval.mask, 0.0, d_out)
    for r in range(x.rows):
        for c in range(x.cols):
            assert_almost_equal(d_x[r, c], d_out[r, c], atol=1e-15)


def test_eval_mode_consumes_no_rng() raises:
    # Eval mode must not draw: the generator state is unchanged, so a downstream
    # seeded draw is unperturbed.
    var rng = Rng(99)
    var state_before = rng.state
    var x = sample_input()
    _ = dropout_cached(x, 0.5, False, rng)
    assert_true(rng.state == state_before)


def test_shape_mismatch_raises() raises:
    var mask = zeros_2d(3, 4)
    var d_out = zeros_2d(3, 5)
    with assert_raises(contains="shape mismatch"):
        _ = dropout_backward(mask, 0.3, d_out)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
