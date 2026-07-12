"""Tests for SamplerConfig and sample_next — the single decoding entry point.

sample_next composes the whole policy: temperature 0 is greedy (argmax, ZERO rng
draws), any positive temperature runs softmax -> top-k -> top-p -> categorical
consuming EXACTLY ONE draw. Both draw-counts are pinned via rng.state so a mixed
seeded pipeline replays exactly. Sampling goldens come from
tests/oracles/sampling_reference.py, which replays the same LCG and inverse-CDF
walk and derives nothing from the Mojo code.
"""

from std.testing import assert_equal, assert_raises, TestSuite

from llm.generation.sampler import SamplerConfig, sample_next
from llm.tensor.ops import argmax
from llm.utils.random import Rng

# The fixed logit row the sampled goldens are computed over. argmax is index 1.
comptime LOGITS: List[Float64] = [1.0, 2.0, 0.5, -1.0, 0.0]


def _logits() -> List[Float64]:
    """Return the fixed logit row [1.0, 2.0, 0.5, -1.0, 0.0]."""
    return [1.0, 2.0, 0.5, -1.0, 0.0]


# --- greedy path: argmax, zero draws ------------------------------------------


def test_greedy_equals_argmax_and_draws_nothing() raises:
    """`temperature` == 0.0 returns argmax and leaves rng.state bit-identical.
    """
    var logits = _logits()
    var cfg = SamplerConfig.greedy()
    var rng = Rng(123456)
    var before = rng.state
    var id = sample_next(logits, cfg, rng)
    assert_equal(id, argmax(logits))  # index 1
    assert_equal(id, 1)
    assert_equal(rng.state, before)  # ZERO draws consumed


def test_greedy_draws_nothing_across_many_calls() raises:
    """Repeated greedy calls never advance the generator."""
    var logits = _logits()
    var cfg = SamplerConfig.greedy()
    var rng = Rng(42)
    var before = rng.state
    for _ in range(50):
        _ = sample_next(logits, cfg, rng)
    assert_equal(rng.state, before)


# --- sampled path: exactly one draw per call ----------------------------------


def test_sampled_consumes_exactly_one_draw() raises:
    """A sampled call advances the state by exactly one next_u64."""
    # Compare against a twin generator stepped once by hand: equal state => exactly
    # one draw (not zero, not two).
    var logits = _logits()
    var cfg = SamplerConfig(1.0, 0, 1.0)  # pure temperature sampling
    var a = Rng(999)
    var b = Rng(999)
    _ = sample_next(logits, cfg, a)
    _ = b.next_u64()  # one draw, by hand
    assert_equal(a.state, b.state)


def test_sampled_lcg_replay_golden_seed42() raises:
    """Seed 42 replays the exact oracle-predicted id sequence."""
    # Oracle (sampling_reference.py, seed 42): [1,1,1,1,1,0,0,0,1,0,1,1].
    #
    # Hand inverse-CDF check for the first draw: softmax of LOGITS is
    #   [0.2071, 0.5630, 0.1256, 0.0280, 0.0762], cdf
    #   [0.2071, 0.7701, 0.8958, 0.9238, 1.0000].
    # Rng(42).uniform() = 0.5682303266439076 falls in [0.2071, 0.7701), so the
    # first id is index 1. The rest follow by replaying the LCG.
    var logits = _logits()
    var cfg = SamplerConfig(1.0, 0, 1.0)
    var rng = Rng(42)
    var expected: List[Int] = [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1]
    for step in range(len(expected)):
        assert_equal(sample_next(logits, cfg, rng), expected[step])


def test_sampled_lcg_replay_golden_seed7() raises:
    """Seed 7 replays the oracle sequence, exercising the low-probability tail.
    """
    # Oracle (seed 7): [1,4,3,1,1,0,1,1,4,1,1,1] — includes tail tokens (indices 3
    # and 4), so it exercises the full CDF walk, not just the dominant bucket.
    var logits = _logits()
    var cfg = SamplerConfig(1.0, 0, 1.0)
    var rng = Rng(7)
    var expected: List[Int] = [1, 4, 3, 1, 1, 0, 1, 1, 4, 1, 1, 1]
    for step in range(len(expected)):
        assert_equal(sample_next(logits, cfg, rng), expected[step])


def test_sampled_lcg_replay_golden_seed0() raises:
    """Seed 0 replays the oracle sequence."""
    # Oracle (seed 0): [0,0,1,1,1,1,0,1,1,1,1,1].
    var logits = _logits()
    var cfg = SamplerConfig(1.0, 0, 1.0)
    var rng = Rng(0)
    var expected: List[Int] = [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    for step in range(len(expected)):
        assert_equal(sample_next(logits, cfg, rng), expected[step])


def test_top_k_one_forces_argmax_regardless_of_seed() raises:
    """`top_k` == 1 forces the draw to the argmax for every seed."""
    # The filter is a one-hot at the argmax, so the outcome is index 1 regardless
    # of seed — the sampled path still draws its one uniform.
    var logits = _logits()
    var cfg = SamplerConfig(1.0, 1, 1.0)  # temperature on, top_k = 1
    for seed in [42, 7, 0, 123456]:
        var rng = Rng(UInt64(seed))
        assert_equal(sample_next(logits, cfg, rng), 1)


# --- SamplerConfig validation and presets -------------------------------------


def test_validate_raises_per_field() raises:
    """`validate` raises on each out-of-range field with the field named."""
    with assert_raises(contains="temperature"):
        SamplerConfig(-0.1, 0, 1.0).validate()
    with assert_raises(contains="top_k"):
        SamplerConfig(1.0, -1, 1.0).validate()
    with assert_raises(contains="top_p"):
        SamplerConfig(1.0, 0, 0.0).validate()  # p <= 0
    with assert_raises(contains="top_p"):
        SamplerConfig(1.0, 0, 1.5).validate()  # p > 1


def test_presets_validate() raises:
    """The greedy and standard presets pass validation."""
    SamplerConfig.greedy().validate()
    SamplerConfig.standard().validate()


def test_preset_values() raises:
    """The greedy and standard presets carry their expected field values."""
    var g = SamplerConfig.greedy()
    assert_equal(g.temperature, 0.0)
    assert_equal(g.top_k, 0)
    assert_equal(g.top_p, 1.0)
    var s = SamplerConfig.standard()
    assert_equal(s.temperature, 1.0)
    assert_equal(s.top_k, 0)
    assert_equal(s.top_p, 1.0)


def test_sample_next_validates_bad_config() raises:
    """`sample_next` validates before sampling, raising on the named field."""
    var logits = _logits()
    var rng = Rng(1)
    with assert_raises(contains="temperature"):
        _ = sample_next(logits, SamplerConfig(-1.0, 0, 1.0), rng)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
