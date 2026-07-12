"""Tests for TokenBatch, the flat [B, T] container of input/target ids.

A batch holds two flat row-major arrays — inputs and targets, each [B, T],
element (b, t) at index b * T + t. These pin that layout (a hand-built batch
returns each value at the coordinate it was placed) and the bounds checks that
keep an off-by-one from silently reading a neighbor's row.
"""

from std.testing import assert_equal, assert_raises, TestSuite

from llm.data import TokenBatch


def test_shape_and_access() raises:
    """A hand-built batch returns each input/target value at its placed coordinate.
    """
    # 2 rows x 3 cols. targets are inputs+100 so the two arrays can't be confused.
    var inputs: List[Int] = [0, 1, 2, 10, 11, 12]
    var targets: List[Int] = [100, 101, 102, 110, 111, 112]
    var batch = TokenBatch(inputs^, targets^, batch_size=2, seq_len=3)
    assert_equal(batch.batch_size, 2)
    assert_equal(batch.seq_len, 3)
    for b in range(2):
        for t in range(3):
            assert_equal(batch.input_at(b, t), b * 10 + t)
            assert_equal(batch.target_at(b, t), 100 + b * 10 + t)


def test_length_mismatch_raises() raises:
    """The constructor raises when a flat length != batch_size * seq_len."""
    var good: List[Int] = [0, 1, 2, 3, 4, 5]
    var short: List[Int] = [0, 1, 2, 3, 4]  # length 5, not 6
    with assert_raises():
        _ = TokenBatch(short.copy(), good.copy(), batch_size=2, seq_len=3)
    with assert_raises():
        _ = TokenBatch(good.copy(), short.copy(), batch_size=2, seq_len=3)


def test_nonpositive_dims_raise() raises:
    """The constructor raises on a non-positive batch_size or seq_len."""
    # Without the check, negative dims can multiply to the right flat length and
    # slip through, leaving accessors with nonsensical ranges like [0, -2).
    var six: List[Int] = [0, 1, 2, 3, 4, 5]
    with assert_raises():
        _ = TokenBatch([], [], batch_size=0, seq_len=0)
    with assert_raises():
        _ = TokenBatch(six.copy(), six.copy(), batch_size=-2, seq_len=-3)


def test_out_of_bounds_raises() raises:
    """`input_at`/target_at raise on out-of-range or negative indices."""
    var inputs: List[Int] = [0, 1, 2, 3, 4, 5]
    var targets: List[Int] = [0, 1, 2, 3, 4, 5]
    var batch = TokenBatch(inputs^, targets^, batch_size=2, seq_len=3)
    with assert_raises():
        _ = batch.input_at(2, 0)  # b == B is out of range
    with assert_raises():
        _ = batch.input_at(0, 3)  # t == T is out of range
    with assert_raises():
        _ = batch.target_at(-1, 0)  # negative index
    with assert_raises():
        _ = batch.target_at(0, -1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
