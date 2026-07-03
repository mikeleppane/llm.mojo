# Tests for dropout — inverted dropout with mode as an argument.
#
# The rng discipline is the subtle part and gets the most attention: eval mode
# (and p == 0) must be the identity AND must not consume a single rng draw, or
# disabling dropout would silently shift every downstream seeded result. That is
# checked by running a generator through dropout and comparing its next draw
# against an untouched twin. Training mode is inverted dropout: survivors scale by
# 1/(1-p) so the expectation is unchanged, and every element is either 0 or the
# scaled input — nothing in between.

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.nn.dropout import dropout
from llm.tensor.tensor2d import from_rows, full_2d
from llm.utils.random import Rng


def test_eval_mode_is_identity() raises:
    var x = from_rows([[1.0, -2.0, 3.5], [0.0, 4.0, -1.0]])
    var rng = Rng(42)
    var y = dropout(x, 0.5, False, rng)  # training=False
    for r in range(x.rows):
        for c in range(x.cols):
            assert_almost_equal(y[r, c], x[r, c], atol=1e-15)


def test_eval_mode_leaves_rng_untouched() raises:
    var x = from_rows([[1.0, 2.0], [3.0, 4.0]])
    var rng = Rng(7)
    var twin = Rng(7)  # same seed, never passed through dropout
    _ = dropout(x, 0.5, False, rng)  # eval: must not draw
    # If dropout consumed any draw, these would diverge.
    assert_almost_equal(rng.uniform(), twin.uniform(), atol=1e-15)


def test_p_zero_in_training_is_identity_and_untouched_rng() raises:
    var x = from_rows([[1.0, 2.0], [3.0, 4.0]])
    var rng = Rng(11)
    var twin = Rng(11)
    var y = dropout(x, 0.0, True, rng)  # training but p == 0: still identity
    for r in range(x.rows):
        for c in range(x.cols):
            assert_almost_equal(y[r, c], x[r, c], atol=1e-15)
    # p == 0 short-circuits before any draw, so the generator is untouched.
    assert_almost_equal(rng.uniform(), twin.uniform(), atol=1e-15)


def test_p_out_of_range_raises() raises:
    var x = from_rows([[1.0]])
    var rng = Rng(1)
    with assert_raises(contains="p must be in"):
        _ = dropout(x, -0.1, True, rng)
    with assert_raises(contains="p must be in"):
        _ = dropout(x, 1.0, True, rng)


def test_p_nan_raises() raises:
    # A NaN p is neither < 0 nor >= 1, so a naive two-sided guard would let it
    # through — silently zeroing the output while consuming rng draws. It must
    # raise like any other out-of-range p.
    var inf = 1.0e308 * 10.0  # overflows to +inf
    var nan_p = inf - inf  # inf - inf = NaN
    var x = from_rows([[1.0]])
    var rng = Rng(1)
    with assert_raises(contains="p must be in"):
        _ = dropout(x, nan_p, True, rng)


def test_training_is_deterministic_for_a_seed() raises:
    var x = full_2d(8, 8, 1.0)
    var rng_a = Rng(2024)
    var rng_b = Rng(2024)
    var ya = dropout(x, 0.3, True, rng_a)
    var yb = dropout(x, 0.3, True, rng_b)
    for r in range(x.rows):
        for c in range(x.cols):
            assert_almost_equal(ya[r, c], yb[r, c], atol=1e-15)


def test_survivors_are_zero_or_scaled_input() raises:
    var x = full_2d(16, 16, 2.0)  # every input is 2.0
    var rng = Rng(5)
    var p = 0.25
    var scaled = 2.0 / (1.0 - p)  # the only nonzero value allowed
    var y = dropout(x, p, True, rng)
    for r in range(y.rows):
        for c in range(y.cols):
            var v = y[r, c]
            var is_zero = abs(v) < 1e-15
            var is_scaled = abs(v - scaled) < 1e-12
            assert_true(is_zero or is_scaled)


def test_empirical_keep_rate_in_band() raises:
    # Over a large seeded tensor the fraction of survivors should sit near the
    # keep probability 1 - p. Seeded, so this is a fixed number, not flaky.
    var x = full_2d(100, 100, 1.0)
    var rng = Rng(99)
    var p = 0.3
    var y = dropout(x, p, True, rng)
    var kept = 0
    for r in range(y.rows):
        for c in range(y.cols):
            if abs(y[r, c]) > 1e-15:
                kept += 1
    var rate = Float64(kept) / 10000.0
    assert_true(abs(rate - (1.0 - p)) < 0.02)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
