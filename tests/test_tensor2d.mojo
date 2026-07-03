# Tests for Tensor2D and its constructors.
#
# Test the abstraction before building on it: shape/size, set-then-get, the
# ones/zeros/full constructors, the checked accessor's out-of-range raise, and
# from_rows rejecting ragged input.

from std.testing import assert_equal, assert_raises, TestSuite

from llm.tensor.tensor2d import Tensor2D, zeros_2d, ones_2d, full_2d, from_rows


def test_zeros_shape() raises:
    var x = zeros_2d(2, 3)
    assert_equal(x.rows, 2)
    assert_equal(x.cols, 3)
    assert_equal(x.size(), 6)
    assert_equal(x[0, 0], 0.0)
    assert_equal(x[1, 2], 0.0)


def test_set_get() raises:
    var x = zeros_2d(2, 3)
    x[1, 2] = 7.5
    assert_equal(x[1, 2], 7.5)


def test_offset() raises:
    var x = zeros_2d(2, 3)
    assert_equal(x.offset(0, 0), 0)
    assert_equal(x.offset(1, 0), 3)
    assert_equal(x.offset(1, 2), 5)


def test_ones() raises:
    var x = ones_2d(2, 2)
    assert_equal(x[0, 0], 1.0)
    assert_equal(x[1, 1], 1.0)


def test_full() raises:
    var x = full_2d(2, 2, -3.0)
    assert_equal(x[0, 1], -3.0)
    assert_equal(x[1, 0], -3.0)


def test_at_out_of_range_raises() raises:
    var x = zeros_2d(2, 2)
    with assert_raises(contains="out of range"):
        _ = x.at(5, 0)


def test_from_rows_values() raises:
    var a = from_rows([[1.0, 2.0], [3.0, 4.0]])
    assert_equal(a.rows, 2)
    assert_equal(a.cols, 2)
    assert_equal(a[0, 1], 2.0)
    assert_equal(a[1, 0], 3.0)


def test_from_rows_rejects_ragged() raises:
    with assert_raises(contains="ragged"):
        _ = from_rows([[1.0, 2.0], [3.0]])


def test_from_rows_rejects_empty() raises:
    var empty = List[List[Float64]]()
    with assert_raises(contains="at least one row"):
        _ = from_rows(empty)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
