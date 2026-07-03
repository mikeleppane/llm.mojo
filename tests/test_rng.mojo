# Tests for the deterministic LCG random generator.
#
# The Rng is the seed of every "seeded" behavior downstream — shuffled batches,
# and later weight init. These tests pin the three things a reader must trust:
# the recurrence produces the exact hand-computed values (golden oracle against
# the Knuth MMIX constants), the same seed always replays the same sequence, and
# the derived helpers (next_below, shuffle) stay in range and stay deterministic.
# Everything here is integer-valued, so every assertion is exact.

from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from llm.utils import Rng


def test_same_seed_same_sequence() raises:
    # Two generators seeded alike replay identical streams; a different seed
    # diverges almost immediately.
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
    # Oracle for the recurrence itself: state_{n+1} = state_n * A + C (mod 2**64),
    # returning the new state. Values computed independently in Python from the
    # MMIX constants A = 6364136223846793005, C = 1442695040888963407. A wrong
    # constant or a non-wrapping multiply fails this instantly.
    #
    # Note: two goldens below (seed0[2] and seed42[0]) exceed 2**63 - 1. They are
    # fine as written because Mojo's arbitrary-precision IntLiteral flows straight
    # into UInt64(...); keep them going directly into UInt64 — routing either
    # through an Int intermediate would overflow.
    var r0 = Rng(0)
    assert_equal(r0.next_u64(), UInt64(1442695040888963407))
    assert_equal(r0.next_u64(), UInt64(1876011003808476466))
    assert_equal(r0.next_u64(), UInt64(11166244414315200793))

    var r42 = Rng(42)
    assert_equal(r42.next_u64(), UInt64(10481999410520546993))
    assert_equal(r42.next_u64(), UInt64(4159066171780167020))
    assert_equal(r42.next_u64(), UInt64(7615522811268512075))


def test_next_below_in_range() raises:
    # 1000 draws below 7 all land in [0, 7) and every value shows up at least
    # once (a sanity check on coverage, not a statistical test).
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
    var r = Rng(1)
    with assert_raises():
        _ = r.next_below(0)
    with assert_raises():
        _ = r.next_below(-1)


def test_shuffle_is_permutation() raises:
    # Shuffling preserves the multiset (sort the result, compare to the input)
    # and actually reorders (seed 1 does not leave [0..99] fixed).
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
    # Same seed -> identical permutation, element for element.
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
