# Tests for matmul, matmul_ikj, and matvec.
#
# The hand-computed small case catches wrong bounds, row/column confusion, and
# offset bugs. The orders-agree test proves the two loop orders are numerically
# equivalent (compared with tolerance, since float summation order differs).
# matvec is pinned to a hand computation.

from std.testing import assert_almost_equal, assert_raises, TestSuite

from llm.tensor.tensor2d import zeros_2d, from_rows
from llm.tensor.ops import matmul, matmul_ikj, matvec


def test_matmul_small() raises:
    var a = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    var c = matmul(a, b)
    assert_almost_equal(c[0, 0], 58.0, atol=1e-12)
    assert_almost_equal(c[0, 1], 64.0, atol=1e-12)
    assert_almost_equal(c[1, 0], 139.0, atol=1e-12)
    assert_almost_equal(c[1, 1], 154.0, atol=1e-12)


def test_orders_agree() raises:
    var a = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    var c1 = matmul(a, b)
    var c2 = matmul_ikj(a, b)
    for i in range(c1.rows):
        for j in range(c1.cols):
            assert_almost_equal(c1[i, j], c2[i, j], atol=1e-9)


def test_at_operator_matches_matmul() raises:
    # `a @ b` (Tensor2D.__matmul__, the ikj kernel) must agree elementwise with
    # the clear ijk matmul on a non-square case. Both sum the same products; a
    # non-square shape [2,3] @ [3,2] catches a row/column or bounds slip that a
    # square case could mask. Same loop order, so the sums are bit-identical here
    # — pin at 1e-12.
    var a = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    var c1 = matmul(a, b)
    var c2 = a @ b
    for i in range(c1.rows):
        for j in range(c1.cols):
            assert_almost_equal(c1[i, j], c2[i, j], atol=1e-12)


def test_at_operator_shape_mismatch_raises() raises:
    # The `@` operator carries its own shape guard; pin it so it can't be
    # deleted unnoticed.
    var a = zeros_2d(2, 3)
    var b = zeros_2d(2, 2)  # 3 != 2
    with assert_raises(contains="shape mismatch"):
        _ = a @ b


def test_matmul_shape_mismatch_raises() raises:
    var a = zeros_2d(2, 3)
    var b = zeros_2d(2, 2)  # 3 != 2
    with assert_raises(contains="shape mismatch"):
        _ = matmul(a, b)


def test_matmul_ikj_shape_mismatch_raises() raises:
    # The ikj path has its own guard; pin it so it can't be deleted unnoticed.
    var a = zeros_2d(2, 3)
    var b = zeros_2d(2, 2)  # 3 != 2
    with assert_raises(contains="shape mismatch"):
        _ = matmul_ikj(a, b)


def test_matvec_hand_computed() raises:
    var a = from_rows([[1.0, 2.0], [3.0, 4.0]])
    var y = matvec(a, [10.0, 100.0])
    assert_almost_equal(y[0], 210.0, atol=1e-12)  # 1*10 + 2*100
    assert_almost_equal(y[1], 430.0, atol=1e-12)  # 3*10 + 4*100


def test_matvec_shape_mismatch_raises() raises:
    var a = zeros_2d(2, 3)
    with assert_raises(contains="shape mismatch"):
        _ = matvec(a, [1.0, 2.0])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
