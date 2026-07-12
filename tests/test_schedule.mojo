"""Tests for training.schedule — linear warmup then cosine decay.

All goldens are hand-derivable (adamw_reference.py prints the same numbers as a
cross-check). Config: peak = 1.0, warmup = 10, max_steps = 100, min_lr = 0.1,
giving 0.0 at step 0, 0.5 at 5, 1.0 at 10, 0.55 at 55, 0.1 from step 100 on.
Also: monotone non-increasing after warmup, the warmup=0 case, and the guards.
"""

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.training.schedule import lr_at, ScheduleConfig


def test_schedule_goldens() raises:
    """`lr_at` matches the hand-derived goldens across the whole schedule."""
    var peak = 1.0
    var warmup = 10
    var maxs = 100
    var floor = 0.1
    assert_almost_equal(lr_at(0, peak, warmup, maxs, floor), 0.0, atol=1e-12)
    assert_almost_equal(lr_at(5, peak, warmup, maxs, floor), 0.5, atol=1e-12)
    assert_almost_equal(lr_at(10, peak, warmup, maxs, floor), 1.0, atol=1e-12)
    assert_almost_equal(lr_at(55, peak, warmup, maxs, floor), 0.55, atol=1e-12)
    assert_almost_equal(lr_at(100, peak, warmup, maxs, floor), 0.1, atol=1e-12)
    assert_almost_equal(lr_at(150, peak, warmup, maxs, floor), 0.1, atol=1e-12)


def test_warmup_end_equals_peak_and_is_continuous() raises:
    """The warmup boundary is continuous and hits exactly peak at step warmup.
    """
    # step warmup-1 is still below peak, step warmup is exactly peak, step
    # warmup+1 is just below peak (cosine started).
    var peak = 2.0
    var warmup = 10
    var maxs = 100
    var floor = 0.0
    assert_true(lr_at(9, peak, warmup, maxs, floor) < peak, "step 9 not < peak")
    assert_almost_equal(lr_at(10, peak, warmup, maxs, floor), 2.0, atol=1e-12)
    assert_true(
        lr_at(11, peak, warmup, maxs, floor) < peak, "step 11 not < peak"
    )


def test_monotone_non_increasing_after_warmup() raises:
    """From warmup to max_steps the cosine branch never increases."""
    var peak = 1.0
    var warmup = 10
    var maxs = 100
    var floor = 0.1
    var prev = lr_at(warmup, peak, warmup, maxs, floor)
    for step in range(warmup + 1, maxs + 5):
        var cur = lr_at(step, peak, warmup, maxs, floor)
        assert_true(
            cur <= prev + 1e-15,
            "lr rose after warmup at step " + String(step),
        )
        prev = cur


def test_warmup_zero_degenerate() raises:
    """`warmup_steps` = 0 starts at peak (step 0) and never divides by zero."""
    var peak = 1.0
    var maxs = 100
    var floor = 0.1
    assert_almost_equal(lr_at(0, peak, 0, maxs, floor), 1.0, atol=1e-12)
    assert_almost_equal(lr_at(50, peak, 0, maxs, floor), 0.55, atol=1e-12)
    assert_almost_equal(lr_at(100, peak, 0, maxs, floor), 0.1, atol=1e-12)


def test_lr_at_guards() raises:
    """`lr_at` raises on a negative step, negative warmup, or max_steps <= warmup.
    """
    with assert_raises(contains="step must be >= 0"):
        _ = lr_at(-1, 1.0, 10, 100, 0.1)
    with assert_raises(contains="warmup_steps must be >= 0"):
        _ = lr_at(0, 1.0, -1, 100, 0.1)
    with assert_raises(contains="must exceed warmup_steps"):
        _ = lr_at(0, 1.0, 100, 100, 0.1)  # max_steps == warmup_steps


def test_schedule_config_validate() raises:
    """ScheduleConfig.validate accepts valid configs and raises on each bad one.
    """
    # Valid: warmup < max_steps, 0 <= min_lr <= peak.
    ScheduleConfig(10, 0.1).validate(100, 1.0)
    with assert_raises(contains="must be < max_steps"):
        ScheduleConfig(100, 0.1).validate(100, 1.0)
    with assert_raises(contains="min_lr must be >= 0"):
        ScheduleConfig(10, -0.1).validate(100, 1.0)
    with assert_raises(contains="must be <= peak"):
        ScheduleConfig(10, 2.0).validate(100, 1.0)
    # A NaN or +inf min_lr must raise too (isfinite guards the one-sided bound).
    var nan_lr: Float64 = FloatLiteral.nan
    with assert_raises(contains="min_lr must be >= 0"):
        ScheduleConfig(10, nan_lr).validate(100, 1.0)
    var inf_lr: Float64 = FloatLiteral.infinity
    with assert_raises(contains="min_lr must be >= 0"):
        ScheduleConfig(10, inf_lr).validate(100, 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
