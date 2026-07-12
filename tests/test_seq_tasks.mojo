"""Tests for the copy/reverse toy tasks — the encoder-decoder lab's training data.

No oracle: these pin exact structural properties (a reversed target, a
right-shifted teacher-forcing input, values inside the alphabet, BOS one past it)
plus seeded determinism. The teacher-forcing shift is pinned by a hand-computed
example — the off-by-one that would quietly cap a model at the uniform baseline.

The hand-pinned sequences use the s4/s8 helpers rather than inline `[a, b, ...]`
List[Int] literals: on the pinned Mojo (1.0.0b2) a handful of `List[Int]`
literals in one module drives the front-end into a multi-minute compile (a
minimal module with the same imports but no such literals builds in ~2 s), so the
helpers keep this file's compile fast.
"""

from std.testing import assert_true, assert_equal, assert_raises, TestSuite

from llm.lab.tasks import (
    SeqPair,
    bos_id,
    random_source,
    copy_target,
    reverse_target,
    make_pair,
    decoder_input,
    sequences_equal,
    unique_sources,
)
from llm.utils.random import Rng


def s4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    """Build a 4-element List[Int] without a `[...]` literal (see module note).
    """
    var o = List[Int]()
    o.append(a)
    o.append(b)
    o.append(c)
    o.append(d)
    return o^


def s8(
    a: Int, b: Int, c: Int, d: Int, e: Int, f: Int, g: Int, h: Int
) -> List[Int]:
    """Build an 8-element List[Int] without a `[...]` literal."""
    var o = List[Int]()
    o.append(a)
    o.append(b)
    o.append(c)
    o.append(d)
    o.append(e)
    o.append(f)
    o.append(g)
    o.append(h)
    return o^


def test_bos_is_one_past_the_alphabet() raises:
    """BOS = V_data, reserving exactly one id beyond the data symbols [0, V_data).
    """
    assert_equal(bos_id(16), 16)
    assert_equal(bos_id(4), 4)


def test_random_source_values_in_alphabet() raises:
    """Every drawn symbol lies in [0, V_data); BOS is never a data symbol."""
    var rng = Rng(7)
    var v_data = 16
    for _ in range(50):
        var s = random_source(rng, v_data, 8)
        assert_equal(len(s), 8)
        for i in range(len(s)):
            assert_true(0 <= s[i] and s[i] < v_data)


def test_random_source_seeded_determinism() raises:
    """Same seed replays the same source; a different seed almost surely diverges.
    """
    var a = Rng(123)
    var b = Rng(123)
    var sa = random_source(a, 16, 8)
    var sb = random_source(b, 16, 8)
    assert_true(sequences_equal(sa, sb))


def test_random_source_raises_on_degenerate() raises:
    """`random_source` raises on a zero alphabet or zero length."""
    var rng = Rng(1)
    with assert_raises():
        _ = random_source(rng, 0, 8)
    with assert_raises():
        _ = random_source(rng, 16, 0)


def test_copy_target_is_the_source() raises:
    """`copy_target` returns the source unchanged."""
    var src = s8(3, 1, 4, 1, 5, 9, 2, 6)
    var tgt = copy_target(src)
    assert_true(sequences_equal(src, tgt))


def test_reverse_target_mirrors_positions() raises:
    """`reverse_target` maps target[i] to source[T-1-i]."""
    var src = s8(3, 1, 4, 1, 5, 9, 2, 6)
    var tgt = reverse_target(src)
    var t = len(src)
    for i in range(t):
        assert_equal(tgt[i], src[t - 1 - i])
    # A concrete pin: [3,1,4,1,5,9,2,6] reversed is [6,2,9,5,1,4,1,3].
    var expected = s8(6, 2, 9, 5, 1, 4, 1, 3)
    assert_true(sequences_equal(tgt, expected))


def test_make_pair_selects_task() raises:
    """`make_pair` selects copy vs reverse by its flag."""
    var src = s4(0, 1, 2, 3)
    var cp = make_pair(src, False)
    assert_true(sequences_equal(cp.src, src))
    assert_true(sequences_equal(cp.tgt, src))
    var rv = make_pair(src, True)
    assert_true(sequences_equal(rv.src, src))
    assert_true(sequences_equal(rv.tgt, s4(3, 2, 1, 0)))


def test_teacher_forcing_shift_hand_computed() raises:
    """`decoder_input` is [BOS] + tgt[:-1], pinned by a hand-computed example.
    """
    # For the reverse pair src = [3, 1, 4, 2], tgt = [2, 4, 1, 3], BOS = 16, the
    # decoder input is [16, 2, 4, 1]: position 0 sees BOS and predicts tgt[0]=2,
    # position 1 sees tgt[0]=2 and predicts tgt[1]=4, and so on. Same length as
    # tgt; the last target token (3) is only ever a label, never fed back in.
    var src = s4(3, 1, 4, 2)
    var pair = make_pair(src, True)
    assert_true(sequences_equal(pair.tgt, s4(2, 4, 1, 3)))
    var din = decoder_input(pair.tgt, bos_id(16))
    assert_true(sequences_equal(din, s4(16, 2, 4, 1)))
    assert_equal(len(din), len(pair.tgt))


def test_decoder_input_raises_on_empty() raises:
    """`decoder_input` raises on an empty target."""
    with assert_raises():
        _ = decoder_input(List[Int](), 16)


def test_unique_sources_are_distinct() raises:
    """Every returned source differs from every other."""
    # This is what makes a train/held-out split a real generalization test rather
    # than a memorization replay.
    var rng = Rng(99)
    var srcs = unique_sources(rng, 16, 8, 40)
    assert_equal(len(srcs), 40)
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            assert_true(not sequences_equal(srcs[i], srcs[j]))


def test_unique_sources_seeded_determinism() raises:
    """Same seed replays the same unique-source set."""
    var a = Rng(55)
    var b = Rng(55)
    var sa = unique_sources(a, 16, 8, 10)
    var sb = unique_sources(b, 16, 8, 10)
    assert_equal(len(sa), len(sb))
    for i in range(len(sa)):
        assert_true(sequences_equal(sa[i], sb[i]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
