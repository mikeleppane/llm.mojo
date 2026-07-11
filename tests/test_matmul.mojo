# Tests for matmul, matmul_ikj, and matvec.
#
# The hand-computed small case catches wrong bounds, row/column confusion, and
# offset bugs. The orders-agree test proves the two loop orders are numerically
# equivalent (compared with tolerance, since float summation order differs).
# matvec is pinned to a hand computation.

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    TestSuite,
)

from llm.tensor.tensor2d import Tensor2D, zeros_2d, from_rows
from llm.tensor.ops import (
    matmul,
    matmul_ikj,
    matmul_transpose_b,
    matvec,
    transpose,
)
from llm.utils.random import Rng


def _random_tensor(mut rng: Rng, rows: Int, cols: Int) -> Tensor2D:
    # A [rows, cols] tensor of seeded normal draws — a shared builder for the
    # matmul_transpose_b exact-equality tests.
    var t = zeros_2d(rows, cols)
    for r in range(rows):
        for c in range(cols):
            t[r, c] = rng.normal(0.0, 1.0)
    return t^


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
    # the clear ijk matmul on a non-square case. The two use different loop
    # nesting (ijk vs ikj), but each output element still accumulates over k in
    # the same increasing order, so the per-element sums come out bit-identical,
    # not merely close. A non-square shape [2,3] @ [3,2] catches a row/column or
    # bounds slip that a square case could mask. Pin at 1e-12.
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


def test_matmul_transpose_b_hand_computed() raises:
    # c = a @ b^T with c[i, j] = sum_k a[i, k] * b[j, k]. a [2, 3], b [2, 3] so
    # b^T is [3, 2] and c is [2, 2]. Each row of a dotted with each row of b:
    #   c[0,0] = 1*7 + 2*8 + 3*9  = 50    c[0,1] = 1*10 + 2*11 + 3*12 = 68
    #   c[1,0] = 4*7 + 5*8 + 6*9  = 122   c[1,1] = 4*10 + 5*11 + 6*12 = 167
    var a = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = from_rows([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    var c = matmul_transpose_b(a, b)
    assert_equal(c.rows, 2)
    assert_equal(c.cols, 2)
    assert_almost_equal(c[0, 0], 50.0, atol=1e-12)
    assert_almost_equal(c[0, 1], 68.0, atol=1e-12)
    assert_almost_equal(c[1, 0], 122.0, atol=1e-12)
    assert_almost_equal(c[1, 1], 167.0, atol=1e-12)


def test_matmul_transpose_b_equals_composed_spelling() raises:
    # The load-bearing claim: matmul_transpose_b(a, b) is BIT-IDENTICAL to
    # matmul(a, transpose(b)). transpose only relabels indices, so each output
    # cell accumulates over k in the same ascending order — the results must be
    # EXACTLY equal, not merely close (assert_equal, no tolerance). This is the
    # exactness the KV-cache tied head inherits: the decode step scores against
    # the [V, C] table with this op, the batch path with the composed spelling,
    # and the two logits rows must match to the bit. Several seeded shapes,
    # including a [1, k] decode row.
    var rng = Rng(20260711)
    var shapes = [(3, 4, 5), (1, 7, 6), (4, 4, 4), (2, 1, 3), (6, 3, 1)]
    for s in range(len(shapes)):
        var m = shapes[s][0]
        var k = shapes[s][1]
        var n = shapes[s][2]
        var a = _random_tensor(rng, m, k)  # [M, K]
        var b = _random_tensor(rng, n, k)  # [N, K], stored un-transposed
        var direct = matmul_transpose_b(a, b)  # a @ b^T
        var composed = matmul(a, transpose(b))  # a @ (b^T), materialized
        assert_equal(direct.rows, m)
        assert_equal(direct.cols, n)
        for i in range(m):
            for j in range(n):
                # Exact equality on purpose: the summation orders are identical,
                # so any drift is a real reordering bug, not float noise.
                assert_equal(direct[i, j], composed[i, j])


def test_matmul_transpose_b_shape_mismatch_raises() raises:
    # The contraction width is the COLUMN count of BOTH operands (b is stored
    # un-transposed), so a.cols must equal b.cols. A [2, 3] and [2, 4] mismatch.
    var a = zeros_2d(2, 3)
    var b = zeros_2d(2, 4)  # 3 != 4 on the shared contraction width
    with assert_raises(contains="shape mismatch"):
        _ = matmul_transpose_b(a, b)


def test_matvec_hand_computed() raises:
    var a = from_rows([[1.0, 2.0], [3.0, 4.0]])
    var x = [10.0, 100.0]
    var y = matvec(a, x)
    assert_almost_equal(y[0], 210.0, atol=1e-12)  # 1*10 + 2*100
    assert_almost_equal(y[1], 430.0, atol=1e-12)  # 3*10 + 4*100


def test_matvec_shape_mismatch_raises() raises:
    var a = zeros_2d(2, 3)
    var x = [1.0, 2.0]
    with assert_raises(contains="shape mismatch"):
        _ = matvec(a, x)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
