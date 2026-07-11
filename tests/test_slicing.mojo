# Tests for the column slice/concat ops — the head split/merge primitives.
#
# slice_cols and concat_cols are the moving parts of multi-head attention's head
# split (one [T, 3C] projection -> H contiguous [T, D] slices) and merge (H heads
# -> [T, C]). A stride bug here would masquerade as a model bug three layers up,
# so they get their own hand-checked and round-trip tests down at the tensor
# layer. Values are exact integers cast to Float64 — no oracle needed.

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    TestSuite,
)

from llm.tensor.ops import slice_cols, slice_rows, concat_cols
from llm.tensor.tensor2d import Tensor2D, from_rows


def test_slice_cols_hand_checked() raises:
    # [2, 4] -> columns [1, 3) -> [2, 2]. Pins that slice_cols reads the right
    # columns from each row, not a transposed or offset window.
    var a = from_rows([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]])
    var s = slice_cols(a, 1, 3)  # [2, 4] -> [2, 2]
    assert_equal(s.rows, 2)
    assert_equal(s.cols, 2)
    assert_almost_equal(s[0, 0], 11.0, atol=1e-12)
    assert_almost_equal(s[0, 1], 12.0, atol=1e-12)
    assert_almost_equal(s[1, 0], 21.0, atol=1e-12)
    assert_almost_equal(s[1, 1], 22.0, atol=1e-12)


def test_slice_cols_full_width_is_identity() raises:
    # Slicing the entire width [0, C) returns a copy equal to the input.
    var a = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var s = slice_cols(a, 0, 3)
    assert_equal(s.rows, 2)
    assert_equal(s.cols, 3)
    for r in range(2):
        for c in range(3):
            assert_almost_equal(s[r, c], a[r, c], atol=1e-12)


def test_split_then_concat_round_trips() raises:
    # Split a [2, 6] tensor into three contiguous [2, 2] blocks, then concat them
    # back: the identity. This is exactly the head split -> merge path MHA relies
    # on, so a round-trip failure here is a head-plumbing bug caught early.
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
    # Concatenating [2,1], [2,3] -> [2,4], columns laid out left to right in
    # part order.
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
    # [4, 2] -> rows [1, 3) -> [2, 2]. Pins that slice_rows reads the right rows
    # with every column intact, not a transposed or offset window.
    var a = from_rows([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]])
    var s = slice_rows(a, 1, 3)  # [4, 2] -> [2, 2]
    assert_equal(s.rows, 2)
    assert_equal(s.cols, 2)
    assert_almost_equal(s[0, 0], 20.0, atol=1e-12)
    assert_almost_equal(s[0, 1], 21.0, atol=1e-12)
    assert_almost_equal(s[1, 0], 30.0, atol=1e-12)
    assert_almost_equal(s[1, 1], 31.0, atol=1e-12)


def test_slice_rows_prefix_is_cache_view() raises:
    # The valid-region view the KV cache leans on: rows [0, t) of a taller
    # buffer. A [5, 3] buffer sliced to its first 2 rows yields exactly those
    # rows, decoupled from the dead rows below.
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
    # Slicing the entire height [0, R) returns a copy equal to the input.
    var a = from_rows([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var s = slice_rows(a, 0, 3)
    assert_equal(s.rows, 3)
    assert_equal(s.cols, 2)
    for r in range(3):
        for c in range(2):
            assert_almost_equal(s[r, c], a[r, c], atol=1e-12)


def test_slice_rows_bad_range_raises() raises:
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
    var parts = List[Tensor2D]()
    with assert_raises(contains="concat_cols"):
        _ = concat_cols(parts)


def test_concat_cols_row_mismatch_raises() raises:
    var parts = List[Tensor2D]()
    parts.append(from_rows([[1.0, 2.0], [3.0, 4.0]]))  # 2 rows
    parts.append(from_rows([[5.0, 6.0]]))  # 1 row, mismatch
    with assert_raises(contains="concat_cols"):
        _ = concat_cols(parts)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
