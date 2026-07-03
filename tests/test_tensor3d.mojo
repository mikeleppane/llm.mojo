# Tests for Tensor3D.
#
# The offset test pins the nested row-major layout itself — offset tests are how
# you catch a transposed stride before it becomes an attention bug. The rest
# cover set-then-get, size, and the checked accessor's out-of-range raise.

from std.testing import assert_equal, assert_raises, TestSuite

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
    assert_equal(t[1, 2, 3], 42.0)
    assert_equal(t.size(), 24)


def test_at_out_of_range_raises() raises:
    var t = zeros_3d(2, 2, 2)
    with assert_raises(contains="out of range"):
        _ = t.at(0, 0, 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
