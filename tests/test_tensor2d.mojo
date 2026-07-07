# Tests for Tensor2D and its constructors.
#
# Test the abstraction before building on it: shape/size, set-then-get, the
# ones/zeros/full constructors, the checked accessor's out-of-range raise, and
# from_rows rejecting ragged input.

from std.testing import (
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.tensor.tensor2d import Tensor2D, zeros_2d, ones_2d, full_2d, from_rows


def test_zeros_shape() raises:
    var x = zeros_2d(2, 3)
    assert_equal(x.rows, 2)
    assert_equal(x.cols, 3)
    assert_equal(x.size(), 6)
    # Stored floats: use the house tolerance habit even where the values are
    # exact, so the pattern a reader copies is the safe one.
    assert_almost_equal(x[0, 0], 0.0, atol=1e-12)
    assert_almost_equal(x[1, 2], 0.0, atol=1e-12)


def test_set_get() raises:
    var x = zeros_2d(2, 3)
    x[1, 2] = 7.5
    assert_almost_equal(x[1, 2], 7.5, atol=1e-12)


def test_subscript_mutates_in_place() raises:
    # The ref-returning subscript serves read, write, and += through one method:
    # `+=` accumulates into the buffer directly, no separate setter.
    var x = zeros_2d(2, 3)
    x[0, 1] = 5.0
    x[0, 1] += 2.0
    assert_almost_equal(x[0, 1], 7.0, atol=1e-12)


def test_offset() raises:
    var x = zeros_2d(2, 3)
    assert_equal(x.offset(0, 0), 0)
    assert_equal(x.offset(1, 0), 3)
    assert_equal(x.offset(1, 2), 5)


def test_ones() raises:
    var x = ones_2d(2, 2)
    assert_almost_equal(x[0, 0], 1.0, atol=1e-12)
    assert_almost_equal(x[1, 1], 1.0, atol=1e-12)


def test_full() raises:
    var x = full_2d(2, 2, -3.0)
    assert_almost_equal(x[0, 1], -3.0, atol=1e-12)
    assert_almost_equal(x[1, 0], -3.0, atol=1e-12)


def test_at_out_of_range_raises() raises:
    var x = zeros_2d(2, 2)
    with assert_raises(contains="out of range"):
        _ = x.at(5, 0)


def test_from_rows_values() raises:
    var a = from_rows([[1.0, 2.0], [3.0, 4.0]])
    assert_equal(a.rows, 2)
    assert_equal(a.cols, 2)
    assert_almost_equal(a[0, 1], 2.0, atol=1e-12)
    assert_almost_equal(a[1, 0], 3.0, atol=1e-12)


def test_from_rows_rejects_ragged() raises:
    with assert_raises(contains="ragged"):
        _ = from_rows([[1.0, 2.0], [3.0]])


def test_from_rows_rejects_empty() raises:
    var empty = List[List[Float64]]()
    with assert_raises(contains="at least one row"):
        _ = from_rows(empty)


def test_writable_shape_and_values() raises:
    # String.write must carry the shape header and the leading values, and cap
    # the preview: a 3x4 tensor has 12 values, so the `…` truncation marker
    # appears rather than all twelve.
    var x = zeros_2d(3, 4)
    x[0, 0] = 1.5
    x[0, 1] = 2.0
    var s = String.write(x)
    assert_true("Tensor2D[3, 4]" in s)
    assert_true("1.5" in s)
    assert_true("2.0" in s)
    assert_true("…" in s)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
