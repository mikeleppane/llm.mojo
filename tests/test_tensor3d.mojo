# Tests for Tensor3D.
#
# The offset test pins the nested row-major layout itself — offset tests are how
# you catch a transposed stride before it becomes an attention bug. The rest
# cover set-then-get, size, and the checked accessor's out-of-range raise.

from std.testing import (
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.tensor.tensor3d import Tensor3D, zeros_3d


def test_offset_layout() raises:
    var t = zeros_3d(2, 3, 4)
    assert_equal(t.offset(0, 0, 0), 0)
    assert_equal(t.offset(0, 0, 3), 3)  # +1 along channels
    assert_equal(t.offset(0, 1, 0), 4)  # +stride1 = d2
    assert_equal(t.offset(1, 0, 0), 12)  # +stride0 = d1*d2
    assert_equal(t.offset(1, 2, 3), 23)  # last element = size - 1


def test_set_get() raises:
    var t = zeros_3d(2, 3, 4)
    t[1, 2, 3] = 42.0
    assert_almost_equal(t[1, 2, 3], 42.0, atol=1e-12)
    assert_equal(t.size(), 24)


def test_subscript_mutates_in_place() raises:
    # The ref-returning subscript serves read, write, and += through one method:
    # `+=` accumulates into the buffer directly, no separate setter.
    var t = zeros_3d(2, 3, 4)
    t[1, 2, 3] = 5.0
    t[1, 2, 3] += 2.0
    assert_almost_equal(t[1, 2, 3], 7.0, atol=1e-12)


def test_at_out_of_range_raises() raises:
    var t = zeros_3d(2, 2, 2)
    with assert_raises(contains="out of range"):
        _ = t.at(0, 0, 5)


def test_writable_shape_and_values() raises:
    # String.write must carry the shape header and the leading values, and cap
    # the preview: a 2x3x4 tensor has 24 values, so the `…` truncation marker
    # appears rather than all of them.
    var t = zeros_3d(2, 3, 4)
    t[0, 0, 0] = 1.5
    t[0, 0, 1] = 2.0
    var s = String.write(t)
    assert_true("Tensor3D[2, 3, 4]" in s)
    assert_true("1.5" in s)
    assert_true("2.0" in s)
    assert_true("…" in s)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
