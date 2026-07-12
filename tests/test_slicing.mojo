"""Tests for the column slice/concat ops — the head split/merge primitives.

slice_cols and concat_cols drive multi-head attention's head split (one [T, 3C]
projection -> H contiguous [T, D] slices) and merge (H heads -> [T, C]). A stride
bug here would masquerade as a model bug three layers up, so they get their own
hand-checked and round-trip tests at the tensor layer. Values are exact integers
cast to Float64 — no oracle needed.
"""

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    TestSuite,
)

from llm.tensor.ops import slice_cols, slice_rows, concat_cols
from llm.tensor.tensor2d import Tensor2D, from_rows


def test_slice_cols_hand_checked() raises:
    """`slice_cols` reads the right column window from each row."""
    # [2, 4] -> columns [1, 3) -> [2, 2]. Pins that it isn't a transposed or offset
    # window.
    var a = from_rows([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]])
    var s = slice_cols(a, 1, 3)  # [2, 4] -> [2, 2]
    assert_equal(s.rows, 2)
    assert_equal(s.cols, 2)
    assert_almost_equal(s[0, 0], 11.0, atol=1e-12)
    assert_almost_equal(s[0, 1], 12.0, atol=1e-12)
    assert_almost_equal(s[1, 0], 21.0, atol=1e-12)
    assert_almost_equal(s[1, 1], 22.0, atol=1e-12)


def test_slice_cols_full_width_is_identity() raises:
    """Slicing the entire width [0, C) returns a copy equal to the input."""
    var a = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var s = slice_cols(a, 0, 3)
    assert_equal(s.rows, 2)
    assert_equal(s.cols, 3)
    for r in range(2):
        for c in range(3):
            assert_almost_equal(s[r, c], a[r, c], atol=1e-12)


def test_split_then_concat_round_trips() raises:
    """Splitting a [2, 6] tensor into three [2, 2] blocks and concatenating is the identity.
    """
    # This is exactly the head split -> merge path MHA relies on, so a round-trip
    # failure here is a head-plumbing bug caught early.
    var a = from_rows(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
    )
    var parts = List[Tensor2D]()
    parts.append(slice_cols(a, 0, 2))
    parts.append(slice_cols(a, 2, 4))
    parts.append(slice_cols(a, 4, 6))
    var merged = concat_cols(parts)
    assert_equal(merged.rows, 2)
    assert_equal(merged.cols, 6)
    for r in range(2):
        for c in range(6):
            assert_almost_equal(merged[r, c], a[r, c], atol=1e-12)


def test_concat_cols_widths_add() raises:
    """`concat_cols` lays parts out left to right, widths adding."""
    # [2,1], [2,3] -> [2,4].
    var p0 = from_rows([[1.0], [4.0]])
    var p1 = from_rows([[2.0, 3.0, 9.0], [5.0, 6.0, 8.0]])
    var parts = List[Tensor2D]()
    parts.append(p0^)
    parts.append(p1^)
    var out = concat_cols(parts)
    assert_equal(out.rows, 2)
    assert_equal(out.cols, 4)
    assert_almost_equal(out[0, 0], 1.0, atol=1e-12)
    assert_almost_equal(out[0, 1], 2.0, atol=1e-12)
    assert_almost_equal(out[0, 3], 9.0, atol=1e-12)
    assert_almost_equal(out[1, 0], 4.0, atol=1e-12)
    assert_almost_equal(out[1, 3], 8.0, atol=1e-12)


def test_slice_cols_bad_range_raises() raises:
    """`slice_cols` raises on inverted, negative, over-wide, or empty ranges."""
    var a = from_rows([[1.0, 2.0, 3.0, 4.0]])
    with assert_raises(contains="slice_cols"):
        _ = slice_cols(a, 3, 1)  # start >= end
    with assert_raises(contains="slice_cols"):
        _ = slice_cols(a, -1, 2)  # start < 0
    with assert_raises(contains="slice_cols"):
        _ = slice_cols(a, 2, 5)  # end > cols
    with assert_raises(contains="slice_cols"):
        _ = slice_cols(a, 2, 2)  # empty range (start == end)


def test_slice_rows_hand_checked() raises:
    """`slice_rows` reads the right row window with every column intact."""
    # [4, 2] -> rows [1, 3) -> [2, 2]. Pins that it isn't a transposed or offset
    # window.
    var a = from_rows([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]])
    var s = slice_rows(a, 1, 3)  # [4, 2] -> [2, 2]
    assert_equal(s.rows, 2)
    assert_equal(s.cols, 2)
    assert_almost_equal(s[0, 0], 20.0, atol=1e-12)
    assert_almost_equal(s[0, 1], 21.0, atol=1e-12)
    assert_almost_equal(s[1, 0], 30.0, atol=1e-12)
    assert_almost_equal(s[1, 1], 31.0, atol=1e-12)


def test_slice_rows_prefix_is_cache_view() raises:
    """`slice_rows` yields the valid-region prefix [0, t) the KV cache leans on.
    """
    # A [5, 3] buffer sliced to its first 2 rows yields exactly those rows,
    # decoupled from the dead rows below.
    var a = from_rows(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    var s = slice_rows(a, 0, 2)
    assert_equal(s.rows, 2)
    assert_equal(s.cols, 3)
    for r in range(2):
        for c in range(3):
            assert_almost_equal(s[r, c], a[r, c], atol=1e-12)


def test_slice_rows_full_height_is_identity() raises:
    """Slicing the entire height [0, R) returns a copy equal to the input."""
    var a = from_rows([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var s = slice_rows(a, 0, 3)
    assert_equal(s.rows, 3)
    assert_equal(s.cols, 2)
    for r in range(3):
        for c in range(2):
            assert_almost_equal(s[r, c], a[r, c], atol=1e-12)


def test_slice_rows_bad_range_raises() raises:
    """`slice_rows` raises on inverted, negative, over-tall, or empty ranges."""
    var a = from_rows([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    with assert_raises(contains="slice_rows"):
        _ = slice_rows(a, 3, 1)  # start >= end
    with assert_raises(contains="slice_rows"):
        _ = slice_rows(a, -1, 2)  # start < 0
    with assert_raises(contains="slice_rows"):
        _ = slice_rows(a, 2, 5)  # end > rows
    with assert_raises(contains="slice_rows"):
        _ = slice_rows(a, 2, 2)  # empty range (start == end)


def test_concat_cols_empty_list_raises() raises:
    """`concat_cols` raises on an empty part list."""
    var parts = List[Tensor2D]()
    with assert_raises(contains="concat_cols"):
        _ = concat_cols(parts)


def test_concat_cols_row_mismatch_raises() raises:
    """`concat_cols` raises when parts have differing row counts."""
    var parts = List[Tensor2D]()
    parts.append(from_rows([[1.0, 2.0], [3.0, 4.0]]))  # 2 rows
    parts.append(from_rows([[5.0, 6.0]]))  # 1 row, mismatch
    with assert_raises(contains="concat_cols"):
        _ = concat_cols(parts)


def test_slice_concat_bit_exact_fractional() raises:
    """`slice_cols`/slice_rows/concat_cols reproduce source bytes bit-for-bit.
    """
    # These copy contiguous rows with memcpy, so use fractional values not exactly
    # representable in binary (0.1, 0.2, …) and assert bit-for-bit equality, so a
    # byte-misaligned, short, or overlapping copy can't hide under almost-equal.
    var src = from_rows(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [2.1, 2.2, 2.3, 2.4, 2.5],
        ]
    )  # [3, 5], fractional

    var sc = slice_cols(src, 1, 4)  # [3, 3], the middle band
    assert_equal(sc.rows, 3)
    assert_equal(sc.cols, 3)
    for r in range(3):
        for c in range(3):
            assert_equal(sc[r, c], src[r, 1 + c])

    var sr = slice_rows(src, 1, 3)  # [2, 5], the bottom two rows
    assert_equal(sr.rows, 2)
    assert_equal(sr.cols, 5)
    for r in range(2):
        for c in range(5):
            assert_equal(sr[r, c], src[1 + r, c])

    # Splitting the columns and concatenating them back must rebuild src exactly.
    var left = slice_cols(src, 0, 2)  # [3, 2]
    var right = slice_cols(src, 2, 5)  # [3, 3]
    var parts = List[Tensor2D]()
    parts.append(left^)
    parts.append(right^)
    var joined = concat_cols(parts)  # [3, 5]
    assert_equal(joined.rows, 3)
    assert_equal(joined.cols, 5)
    for r in range(3):
        for c in range(5):
            assert_equal(joined[r, c], src[r, c])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
