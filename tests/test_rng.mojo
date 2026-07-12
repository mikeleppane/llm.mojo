"""Tests for the deterministic LCG random generator.

The Rng seeds every "seeded" behavior downstream (shuffled batches, weight init).
These pin the three things a reader must trust: the recurrence produces the exact
hand-computed values (golden oracle against the Knuth MMIX constants), the same
seed always replays the same sequence, and the derived helpers (next_below,
shuffle) stay in range and deterministic.
"""

from std.testing import (
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_raises,
    TestSuite,
)
from std.math import isnan

from llm.utils import Rng
from llm.tensor.init_weights import xavier_2d


def test_same_seed_same_sequence() raises:
    """Equal seeds replay identical streams; a different seed diverges."""
    var a = Rng(42)
    var b = Rng(42)
    for _ in range(100):
        assert_equal(a.next_u64(), b.next_u64())

    var c = Rng(42)
    var d = Rng(43)
    var diverged = False
    for _ in range(4):
        if c.next_u64() != d.next_u64():
            diverged = True
    assert_true(diverged)


def test_first_values_golden() raises:
    """`next_u64` matches goldens computed independently from the MMIX constants.
    """
    # Recurrence: state_{n+1} = state_n * A + C (mod 2**64), returning the new
    # state, with A = 6364136223846793005, C = 1442695040888963407. A wrong
    # constant or a non-wrapping multiply fails instantly.
    #
    # Note: two goldens below (seed0[2] and seed42[0]) exceed 2**63 - 1. They flow
    # straight into UInt64(...) via Mojo's arbitrary-precision IntLiteral; keep
    # them going directly into UInt64 — an Int intermediate would overflow.
    var r0 = Rng(0)
    assert_equal(r0.next_u64(), UInt64(1442695040888963407))
    assert_equal(r0.next_u64(), UInt64(1876011003808476466))
    assert_equal(r0.next_u64(), UInt64(11166244414315200793))

    var r42 = Rng(42)
    assert_equal(r42.next_u64(), UInt64(10481999410520546993))
    assert_equal(r42.next_u64(), UInt64(4159066171780167020))
    assert_equal(r42.next_u64(), UInt64(7615522811268512075))


def test_next_below_in_range() raises:
    """1000 draws below 7 all land in [0, 7) and every value appears."""
    var r = Rng(123)
    var seen: List[Bool] = []
    for _ in range(7):
        seen.append(False)
    for _ in range(1000):
        var v = r.next_below(7)
        assert_true(v >= 0 and v < 7)
        seen[v] = True
    for i in range(7):
        assert_true(seen[i])


def test_next_below_invalid_raises() raises:
    """`next_below` raises on a non-positive bound."""
    var r = Rng(1)
    with assert_raises():
        _ = r.next_below(0)
    with assert_raises():
        _ = r.next_below(-1)


def test_shuffle_is_permutation() raises:
    """`shuffle` preserves the multiset and actually reorders."""
    var items: List[Int] = []
    for i in range(100):
        items.append(i)
    var r = Rng(1)
    r.shuffle(items)

    var changed = False
    for i in range(100):
        if items[i] != i:
            changed = True
    assert_true(changed)

    sort(items)
    for i in range(100):
        assert_equal(items[i], i)


def test_shuffle_deterministic() raises:
    """Same seed produces an identical permutation, element for element."""
    var a: List[Int] = []
    var b: List[Int] = []
    for i in range(50):
        a.append(i)
        b.append(i)
    var ra = Rng(7)
    var rb = Rng(7)
    ra.shuffle(a)
    rb.shuffle(b)
    for i in range(50):
        assert_equal(a[i], b[i])


def test_uniform_in_unit_interval() raises:
    """1000 uniform draws all land in [0, 1)."""
    var r = Rng(7)
    for _ in range(1000):
        var u = r.uniform()
        assert_true(u >= 0.0 and u < 1.0)


def test_uniform_first_values_golden() raises:
    """`uniform` matches goldens derived from the next_u64 stream."""
    # uniform() is (next_u64() >> 11) / 2**53. Multiplying by the exact power of
    # two 2**-53 is bit-exact, so these pin the transform, not just its range — a
    # stubbed uniform() returning 0.5 fails here.
    var r0 = Rng(0)
    assert_almost_equal(r0.uniform(), 0.07820865487829387, atol=1e-15)
    var r42 = Rng(42)
    assert_almost_equal(r42.uniform(), 0.5682303266439076, atol=1e-15)


def test_uniform_produces_distinct_values() raises:
    """`uniform` spreads across many draws (catches a stuck generator)."""
    var r = Rng(55)
    var first = r.uniform()
    var all_same = True
    for _ in range(50):
        if r.uniform() != first:
            all_same = False
    assert_true(not all_same)


def test_uniform_deterministic() raises:
    """Same seed produces an identical uniform stream, draw for draw."""
    # Exact equality is the property under test (bit-reproducibility).
    var a = Rng(99)
    var b = Rng(99)
    for _ in range(100):
        assert_equal(a.uniform(), b.uniform())


def test_uniform_range_within_bounds() raises:
    """`uniform_range`(lo, hi) draws all land in [lo, hi)."""
    var r = Rng(3)
    for _ in range(1000):
        var u = r.uniform_range(-2.0, 5.0)
        assert_true(u >= -2.0 and u < 5.0)


def test_normal_is_finite() raises:
    """Box-Muller normal draws are always finite (never NaN or infinity)."""
    # The log(0) guard is what makes this hold across many draws.
    var r = Rng(11)
    for _ in range(1000):
        var z = r.normal(0.0, 1.0)
        assert_true(not isnan(z))
        assert_true(z > -1.0e6 and z < 1.0e6)


def test_normal_first_value_golden() raises:
    """`normal` matches an independent Box-Muller oracle from the seed-0 stream.
    """
    # With u1, u2 the first two uniform() draws, z = sqrt(-2 ln u1) * cos(2 pi u2).
    # A transcription bug (wrong constant, sin instead of cos, one draw instead of
    # two) fails here. Looser tolerance because sqrt/log/cos may differ by a ULP
    # across math libraries.
    var r = Rng(0)
    assert_almost_equal(r.normal(0.0, 1.0), 1.812167873138187, atol=1e-12)


def test_normal_deterministic() raises:
    """Same seed produces an identical normal stream, draw for draw."""
    var a = Rng(5)
    var b = Rng(5)
    for _ in range(100):
        assert_equal(a.normal(0.0, 1.0), b.normal(0.0, 1.0))


def test_xavier_shape() raises:
    """`xavier_2d`(fan_in, fan_out) produces a [fan_out, fan_in] weight tensor.
    """
    var r = Rng(1)
    var w = xavier_2d(r, 4, 6)
    assert_equal(w.rows, 6)  # fan_out
    assert_equal(w.cols, 4)  # fan_in


def test_xavier_deterministic() raises:
    """Same seed produces identical xavier weights."""
    var ra = Rng(2)
    var rb = Rng(2)
    var wa = xavier_2d(ra, 3, 5)
    var wb = xavier_2d(rb, 3, 5)
    for i in range(wa.rows):
        for j in range(wa.cols):
            assert_equal(wa[i, j], wb[i, j])


def test_xavier_no_nan() raises:
    """`xavier_2d` produces no NaN entries."""
    var r = Rng(4)
    var w = xavier_2d(r, 8, 8)
    for i in range(w.rows):
        for j in range(w.cols):
            assert_true(not isnan(w[i, j]))


def test_xavier_rejects_nonpositive_fan() raises:
    """`xavier_2d` raises on a non-positive fan_in or fan_out."""
    var r = Rng(4)
    with assert_raises(contains="must be positive"):
        _ = xavier_2d(r, 0, 4)
    with assert_raises(contains="must be positive"):
        _ = xavier_2d(r, 4, -1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
