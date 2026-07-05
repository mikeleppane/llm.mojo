# Tests for training.schedule — linear warmup then cosine decay.
#
# All goldens are hand-computable (adamw_reference.py prints the same numbers as
# a cross-check, but every value here is derived in the comment). Config:
# peak = 1.0, warmup = 10, max_steps = 100, min_lr = 0.1.
#   step 0    -> 0.0    (warmup starts at 0)
#   step 5    -> 0.5    (linear: 1.0 * 5/10)
#   step 10   -> 1.0    (warmup end = peak; cosine progress 0)
#   step 55   -> 0.55   (cosine midpoint: progress 45/90 = 0.5, cos(pi/2)=0,
#                        cosine factor 0.5, 0.1 + 0.9*0.5)
#   step 100  -> 0.1    (== min_lr at max_steps)
#   step 150  -> 0.1    (clamped past the end)
# Plus: monotone non-increasing after warmup, the warmup=0 degenerate case, and
# ScheduleConfig.validate / lr_at guards.

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.training.schedule import lr_at, ScheduleConfig


def test_schedule_goldens() raises:
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
    # The warmup boundary is continuous: step warmup-1 is still below peak, step
    # warmup is exactly peak, step warmup+1 is just below peak (cosine started).
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
    # From warmup to max_steps the cosine branch never increases.
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
    # warmup_steps = 0: no warmup phase, step 0 is already the peak (cosine
    # progress 0). This must NOT divide by zero.
    var peak = 1.0
    var maxs = 100
    var floor = 0.1
    assert_almost_equal(lr_at(0, peak, 0, maxs, floor), 1.0, atol=1e-12)
    assert_almost_equal(lr_at(50, peak, 0, maxs, floor), 0.55, atol=1e-12)
    assert_almost_equal(lr_at(100, peak, 0, maxs, floor), 0.1, atol=1e-12)


def test_lr_at_guards() raises:
    with assert_raises(contains="step must be >= 0"):
        _ = lr_at(-1, 1.0, 10, 100, 0.1)
    with assert_raises(contains="warmup_steps must be >= 0"):
        _ = lr_at(0, 1.0, -1, 100, 0.1)
    with assert_raises(contains="must exceed warmup_steps"):
        _ = lr_at(0, 1.0, 100, 100, 0.1)  # max_steps == warmup_steps


def test_schedule_config_validate() raises:
    # Valid: warmup < max_steps, 0 <= min_lr <= peak.
    ScheduleConfig(10, 0.1).validate(100, 1.0)
    with assert_raises(contains="must be < max_steps"):
        ScheduleConfig(100, 0.1).validate(100, 1.0)
    with assert_raises(contains="min_lr must be >= 0"):
        ScheduleConfig(10, -0.1).validate(100, 1.0)
    with assert_raises(contains="must be <= peak"):
        ScheduleConfig(10, 2.0).validate(100, 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
