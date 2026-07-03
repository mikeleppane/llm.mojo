# Tests for elementwise add/scale and transpose.
#
# The transpose round trip (transpose(transpose(a)) == a) is the pattern to
# memorize: whenever an op has an inverse, the round trip is the cheapest strong
# test. add's shape-mismatch raise is checked too.

from std.testing import (
    assert_equal,
    assert_almost_equal,
    assert_raises,
    TestSuite,
)

from llm.tensor.tensor2d import zeros_2d, from_rows
from llm.tensor.ops import add, scale, transpose


def test_add_elementwise() raises:
    var a = from_rows([[1.0, 2.0], [3.0, 4.0]])
    var b = from_rows([[10.0, 20.0], [30.0, 40.0]])
    var c = add(a, b)
    assert_almost_equal(c[0, 0], 11.0, atol=1e-12)
    assert_almost_equal(c[1, 1], 44.0, atol=1e-12)


def test_add_shape_mismatch_raises() raises:
    var a = zeros_2d(2, 3)
    var b = zeros_2d(3, 2)
    with assert_raises(contains="shape mismatch"):
        _ = add(a, b)


def test_scale() raises:
    var a = from_rows([[1.0, -2.0]])
    var c = scale(a, 0.5)
    assert_almost_equal(c[0, 0], 0.5, atol=1e-12)
    assert_almost_equal(c[0, 1], -1.0, atol=1e-12)


def test_transpose_shape_and_values() raises:
    var a = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var t = transpose(a)
    assert_equal(t.rows, 3)
    assert_equal(t.cols, 2)
    assert_almost_equal(t[2, 0], 3.0, atol=1e-12)
    assert_almost_equal(t[2, 1], 6.0, atol=1e-12)


def test_transpose_round_trip() raises:
    var a = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var back = transpose(transpose(a))
    for i in range(a.rows):
        for j in range(a.cols):
            assert_almost_equal(back[i, j], a[i, j], atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
