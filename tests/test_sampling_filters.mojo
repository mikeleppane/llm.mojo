# Tests for the top-k and top-p distribution filters.
#
# The filters operate in probability space: each maps a distribution to a
# distribution (keep a subset, zero the rest, renormalize to sum 1). Every golden
# here is produced independently by tests/oracles/sampling_reference.py (NumPy-free
# Python, deriving nothing from the Mojo code) and frozen inline. The assertions
# are exact where the algebra is exact (identities, sums) and matched to the
# oracle's printed precision otherwise.
#
# The properties that matter and are pinned: the tie rule (equal probabilities ->
# lower index survives, in BOTH filters, matching argmax's first-wins), the
# disabled sentinels (k == 0 and p >= 1.0 are the identity), the degenerate ends
# (k == 1 and a tiny p collapse to the argmax one-hot), renormalization, the
# guarded errors, and the pinned pipeline order (top-k THEN top-p).

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    TestSuite,
)

from llm.generation.sampler import filter_top_k, filter_top_p

comptime TOL = 1e-12


def _assert_dist_equal(got: List[Float64], expected: List[Float64]) raises:
    # Elementwise compare two distributions at TOL, after checking the length.
    assert_equal(len(got), len(expected))
    for i in range(len(expected)):
        assert_almost_equal(got[i], expected[i], atol=TOL)


def _sum(xs: List[Float64]) -> Float64:
    var s = 0.0
    for i in range(len(xs)):
        s += xs[i]
    return s


# --- top-k --------------------------------------------------------------------


def test_top_k_goldens() raises:
    # Fixture A8 (sums to 1); oracle goldens for several k.
    var a8: List[Float64] = [0.05, 0.10, 0.30, 0.20, 0.15, 0.08, 0.07, 0.05]

    _assert_dist_equal(
        filter_top_k(a8, 1),
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    _assert_dist_equal(
        filter_top_k(a8, 2),
        [0.0, 0.0, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0],
    )
    _assert_dist_equal(
        filter_top_k(a8, 3),
        [
            0.0,
            0.0,
            0.4615384615384615,
            0.3076923076923077,
            0.23076923076923075,
            0.0,
            0.0,
            0.0,
        ],
    )
    _assert_dist_equal(
        filter_top_k(a8, 5),
        [
            0.0,
            0.12048192771084339,
            0.3614457831325301,
            0.24096385542168677,
            0.18072289156626506,
            0.0963855421686747,
            0.0,
            0.0,
        ],
    )


def test_top_k_disabled_is_identity() raises:
    # k == 0 (the DISABLED sentinel) and k >= n both return the input unchanged.
    var a8: List[Float64] = [0.05, 0.10, 0.30, 0.20, 0.15, 0.08, 0.07, 0.05]
    _assert_dist_equal(filter_top_k(a8, 0), a8)
    _assert_dist_equal(filter_top_k(a8, 8), a8)  # k == n
    _assert_dist_equal(filter_top_k(a8, 99), a8)  # k > n


def test_top_k_one_is_argmax_one_hot() raises:
    # k == 1 collapses to a one-hot at the argmax (index 2 here).
    var a8: List[Float64] = [0.05, 0.10, 0.30, 0.20, 0.15, 0.08, 0.07, 0.05]
    var got = filter_top_k(a8, 1)
    assert_almost_equal(got[2], 1.0, atol=TOL)
    for i in range(len(got)):
        if i != 2:
            assert_almost_equal(got[i], 0.0, atol=TOL)


def test_top_k_boundary_tie_keeps_lower_index() raises:
    # T4 ties 0.4 at indices 1 and 3, and 0.1 at indices 0 and 2. k == 1 must
    # keep index 1 (lower of the tied max); k == 3 must break the 0.1 tie toward
    # index 0. This is the pinned tie rule: equal probs -> lower index survives.
    var t4: List[Float64] = [0.1, 0.4, 0.1, 0.4]
    _assert_dist_equal(filter_top_k(t4, 1), [0.0, 1.0, 0.0, 0.0])
    _assert_dist_equal(filter_top_k(t4, 2), [0.0, 0.5, 0.0, 0.5])
    _assert_dist_equal(
        filter_top_k(t4, 3),
        [
            0.11111111111111112,
            0.4444444444444445,
            0.0,
            0.4444444444444445,
        ],
    )


def test_top_k_renormalizes_to_one() raises:
    var a8: List[Float64] = [0.05, 0.10, 0.30, 0.20, 0.15, 0.08, 0.07, 0.05]
    for k in range(1, 9):
        assert_almost_equal(_sum(filter_top_k(a8, k)), 1.0, atol=TOL)


def test_top_k_raises() raises:
    var a8: List[Float64] = [0.05, 0.10, 0.30, 0.20, 0.15, 0.08, 0.07, 0.05]
    with assert_raises(contains="negative"):
        _ = filter_top_k(a8, -1)
    with assert_raises(contains="empty"):
        _ = filter_top_k(List[Float64](), 3)


# --- top-p --------------------------------------------------------------------


def test_top_p_goldens() raises:
    var a8: List[Float64] = [0.05, 0.10, 0.30, 0.20, 0.15, 0.08, 0.07, 0.05]

    # p = 0.5: smallest prefix reaching 0.5 is {i2, i3} (0.30 + 0.20).
    _assert_dist_equal(
        filter_top_p(a8, 0.5),
        [0.0, 0.0, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0],
    )
    # p = 0.9: the descending order is [2,3,4,1,5,6,0,7]. The running sum after
    # i6 is 0.30+0.20+0.15+0.10+0.08+0.07 = 0.8999999999999999 — one ULP UNDER 0.9
    # in IEEE-754 — so the `>= p` test does NOT fire there. i0 is admitted too
    # (sum 0.95) and only i7 (the second tied 0.05) is dropped. A reminder that a
    # cumulative-sum threshold is an exact floating-point comparison, not the
    # rounded decimal it looks like.
    _assert_dist_equal(
        filter_top_p(a8, 0.9),
        [
            0.052631578947368425,
            0.10526315789473685,
            0.3157894736842105,
            0.2105263157894737,
            0.15789473684210525,
            0.08421052631578949,
            0.0736842105263158,
            0.0,
        ],
    )
    # tiny p keeps exactly the argmax (never an empty distribution).
    _assert_dist_equal(
        filter_top_p(a8, 0.01),
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_top_p_tie_case() raises:
    # T4: 0.4 tie at 1,3 and 0.1 tie at 0,2. Sorted order is 1,3,0,2. p in
    # {0.5, 0.8} both keep {1,3}; p = 0.85 adds index 0 (the lower of the 0.1
    # tie) to reach 0.9.
    var t4: List[Float64] = [0.1, 0.4, 0.1, 0.4]
    _assert_dist_equal(filter_top_p(t4, 0.5), [0.0, 0.5, 0.0, 0.5])
    _assert_dist_equal(filter_top_p(t4, 0.8), [0.0, 0.5, 0.0, 0.5])
    _assert_dist_equal(
        filter_top_p(t4, 0.85),
        [
            0.11111111111111112,
            0.4444444444444445,
            0.0,
            0.4444444444444445,
        ],
    )


def test_top_p_disabled_is_identity() raises:
    # p == 1.0 is the DISABLED sentinel: the identity, returned BEFORE any
    # cumulative sum runs. (p > 1.0 is out of range and raises — see
    # test_top_p_raises — so 1.0 is the exact disabled value.)
    var a8: List[Float64] = [0.05, 0.10, 0.30, 0.20, 0.15, 0.08, 0.07, 0.05]
    _assert_dist_equal(filter_top_p(a8, 1.0), a8)


def test_top_p_tiny_keeps_only_argmax() raises:
    var a8: List[Float64] = [0.05, 0.10, 0.30, 0.20, 0.15, 0.08, 0.07, 0.05]
    var got = filter_top_p(a8, 1e-9)
    assert_almost_equal(got[2], 1.0, atol=TOL)
    for i in range(len(got)):
        if i != 2:
            assert_almost_equal(got[i], 0.0, atol=TOL)


def test_top_p_renormalizes_to_one() raises:
    var a8: List[Float64] = [0.05, 0.10, 0.30, 0.20, 0.15, 0.08, 0.07, 0.05]
    for pi in range(1, 10):
        var p = Float64(pi) / 10.0
        assert_almost_equal(_sum(filter_top_p(a8, p)), 1.0, atol=TOL)


def test_top_p_raises() raises:
    var a8: List[Float64] = [0.05, 0.10, 0.30, 0.20, 0.15, 0.08, 0.07, 0.05]
    with assert_raises(contains="in (0, 1]"):
        _ = filter_top_p(a8, 0.0)
    with assert_raises(contains="in (0, 1]"):
        _ = filter_top_p(a8, -0.5)
    with assert_raises(contains="in (0, 1]"):
        _ = filter_top_p(a8, 1.5)  # p > 1 is out of range, not disabled
    with assert_raises(contains="empty"):
        _ = filter_top_p(List[Float64](), 0.5)


# --- composition (pinned order) -----------------------------------------------


def test_composition_order_is_top_k_then_top_p() raises:
    # The pipeline applies top-k THEN top-p. The fixture is built so the two
    # orders DISAGREE: pinned order keeps 2 tokens, the swapped order keeps 3.
    var comp: List[Float64] = [0.5, 0.25, 0.15, 0.10]

    var pinned = filter_top_p(filter_top_k(comp, 3), 0.80)
    _assert_dist_equal(
        pinned, [0.6666666666666666, 0.3333333333333333, 0.0, 0.0]
    )

    # The swapped order produces a genuinely different distribution — so the
    # golden above pins the ORDER, not just the filters.
    var swapped = filter_top_k(filter_top_p(comp, 0.80), 3)
    _assert_dist_equal(
        swapped,
        [0.5555555555555556, 0.2777777777777778, 0.16666666666666666, 0.0],
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
