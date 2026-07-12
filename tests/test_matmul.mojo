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
    matmul_transpose_a,
    matmul_transpose_b,
    matvec,
    transpose,
)
from llm.utils.random import Rng


def _random_tensor(mut rng: Rng, rows: Int, cols: Int) -> Tensor2D:
    # A [rows, cols] tensor of seeded normal draws — a shared builder for the
    # matmul_transpose_b / matvec kernel-vs-reference tests.
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


def _scalar_ikj(a: Tensor2D, b: Tensor2D) raises -> Tensor2D:
    # The plain scalar ikj matmul — the reference the SIMD-over-j `@` operator
    # must match BIT-FOR-BIT. Same loop order as `@`, no vectorization: each
    # out[i, j] accumulates over k ascending via out[i, j] += a[i, k]*b[k, j].
    var out = zeros_2d(a.rows, b.cols)
    for i in range(a.rows):
        for k in range(a.cols):
            var a_ik = a[i, k]
            for j in range(b.cols):
                out[i, j] += a_ik * b[k, j]
    return out^


def test_at_operator_bit_identical_to_scalar_ikj() raises:
    # `@` vectorizes over the OUTPUT columns j (Class A): each cell still sums k in
    # the same ascending order as the scalar ikj loop, so the two must be EXACTLY
    # equal (assert_equal, no tolerance) — SIMD-over-j reorders nothing that gets
    # summed. Any drift here is a real bug (a mis-handled tail, an off-by-one in
    # the vector step), not float noise. Seeded shapes span sizes 1 and 2 and
    # output widths n that are NOT multiples of the f64 SIMD width 4 (1, 2, 3, 5,
    # 7, 13) so the scalar remainder loop is exercised beside the full vectors.
    var rng = Rng(20260713)
    var shapes = [
        (1, 1, 1),
        (2, 3, 2),
        (3, 4, 5),
        (2, 5, 7),
        (4, 8, 13),
        (1, 768, 3),
        (5, 64, 64),
        (64, 128, 512),  # crosses the @ threading threshold (work ~4M, m>=32)
        (
            40,
            100,
            101,
        ),  # threaded, m not divisible by the block count, ragged n
    ]
    for s in range(len(shapes)):
        var m = shapes[s][0]
        var k = shapes[s][1]
        var n = shapes[s][2]
        var a = _random_tensor(rng, m, k)
        var b = _random_tensor(rng, k, n)
        var fast = a @ b
        var reference = _scalar_ikj(a, b)
        assert_equal(fast.rows, m)
        assert_equal(fast.cols, n)
        for i in range(m):
            for j in range(n):
                assert_equal(fast[i, j], reference[i, j])


def test_at_operator_threaded_is_deterministic() raises:
    # The @ operator threads over output rows above the work threshold; that
    # static partition must be run-to-run bit-identical. Two calls on the same
    # threaded-size inputs (work ~8M, m=64) must agree EXACTLY.
    var rng = Rng(20260715)
    var a = _random_tensor(rng, 64, 256)
    var b = _random_tensor(rng, 256, 512)
    var c1 = a @ b
    var c2 = a @ b
    for i in range(c1.rows):
        for j in range(c1.cols):
            assert_equal(c1[i, j], c2[i, j])


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


def _assert_close_rel(got: Float64, want: Float64, tag: String) raises:
    # Genuinely 1e-12 RELATIVE: |got - want| <= 1e-12*|want| + 1e-13. The rtol
    # term is the Class B contract — matmul_transpose_b's SIMD dot reassociates
    # the k-sum, so it agrees with the scalar spelling only to ~k*eps. Measured
    # max relative error over these shapes is ~2.3e-13 at k=4 and <1e-13 at
    # k=768/3072, so 1e-12 relative is rigorous with ~4x margin and NOT dominated
    # by the floor. The 1e-13 atol only catches an exact-zero want (never produced
    # by these seeded dots); it sits below the measured absolute error, so the
    # relative term governs every real cell. A real tail/reorder bug (errors
    # orders larger) is caught; legitimate reassociation passes.
    var tol = 1e-12 * abs(want) + 1e-13
    if abs(got - want) > tol:
        raise Error(
            "not close ("
            + tag
            + "): got "
            + String(got)
            + " want "
            + String(want)
            + " |diff| "
            + String(abs(got - want))
            + " > tol "
            + String(tol)
        )


def test_matmul_transpose_b_matches_composed_spelling() raises:
    # matmul_transpose_b(a, b) == matmul(a, transpose(b)) to 1e-12 RELATIVE. It is
    # a Class B kernel: a multi-accumulator SIMD dot over the contraction k, which
    # reassociates the sum vs the scalar spelling's left-to-right order (float
    # addition is not associative), so the two agree within ~k*eps, NOT bit-for-bit.
    # `matmul` stays the scalar ijk reference oracle — this is a fast-vs-naive
    # comparison, the only kind that proves the fast kernel correct. Shapes span the
    # real model contraction widths (k=768 c_attn/scores/tiedhead, k=3072 mlp_down),
    # [1, k] decode rows, and RAGGED tails k not divisible by the f64 SIMD width
    # (4 here) — 5, 7, 13, 769, 3073 — to exercise the scalar remainder loop.
    var rng = Rng(20260711)
    var shapes = [
        (3, 4, 5),
        (1, 7, 6),
        (4, 4, 4),
        (2, 1, 3),
        (6, 3, 1),
        (1, 5, 8),
        (2, 13, 4),
        (1, 768, 7),
        (3, 768, 5),
        (1, 3072, 2),
        (2, 769, 3),
        (1, 3073, 4),
        (2, 768, 2048),  # crosses the threading threshold (work ~3M, n>=32)
        (1, 768, 4096),  # threaded, m=1 decode-shaped, n a multiple of blocks
        (3, 512, 2000),  # threaded, n not divisible by the block count
    ]
    for s in range(len(shapes)):
        var m = shapes[s][0]
        var k = shapes[s][1]
        var n = shapes[s][2]
        var a = _random_tensor(rng, m, k)  # [M, K]
        var b = _random_tensor(rng, n, k)  # [N, K], stored un-transposed
        var direct = matmul_transpose_b(a, b)  # a @ b^T, SIMD dot
        var composed = matmul(a, transpose(b))  # a @ (b^T), scalar reference
        assert_equal(direct.rows, m)
        assert_equal(direct.cols, n)
        for i in range(m):
            for j in range(n):
                _assert_close_rel(
                    direct[i, j], composed[i, j], "mtb k=" + String(k)
                )


def test_matmul_transpose_b_threaded_is_deterministic() raises:
    # The threaded path partitions output columns statically across workers, so it
    # must be run-to-run bit-identical (no shared accumulators, no scheduling
    # dependence). Two calls on the same threaded-size inputs must agree EXACTLY.
    # A shape well over the threshold (work ~6M, n=4096) guarantees the parallel
    # branch runs.
    var rng = Rng(20260714)
    var a = _random_tensor(rng, 2, 768)
    var b = _random_tensor(rng, 4096, 768)
    var c1 = matmul_transpose_b(a, b)
    var c2 = matmul_transpose_b(a, b)
    assert_equal(c1.rows, 2)
    assert_equal(c1.cols, 4096)
    for i in range(c1.rows):
        for j in range(c1.cols):
            assert_equal(c1[i, j], c2[i, j])


def test_matmul_transpose_b_threaded_matches_serial() raises:
    # The threading claim is Class A: the parallel path must be bit-identical to
    # the single-threaded _simd_dot, not merely deterministic. matvec(b, a.row(i))
    # dots each row of b against a[i, :] through the SAME _simd_dot, serially — so
    # matmul_transpose_b(a, b)[i, j] must EXACTLY equal matvec(b, a.row(i))[j].
    # A threaded-size b (n=4096, work > 1M) forces the parallel branch.
    var rng = Rng(20260717)
    var a = _random_tensor(rng, 3, 768)  # [M, K]
    var b = _random_tensor(rng, 4096, 768)  # [N, K]
    var threaded = matmul_transpose_b(a, b)  # parallel over output columns
    assert_equal(threaded.rows, 3)
    assert_equal(threaded.cols, 4096)
    for i in range(3):
        var serial = matvec(b, a.row(i))  # serial _simd_dot per output column
        for j in range(4096):
            assert_equal(threaded[i, j], serial[j])


def test_matmul_transpose_a_threaded_is_deterministic() raises:
    # matmul_transpose_a threads over output rows above the work threshold; that
    # static partition must be run-to-run bit-identical. Two calls on the same
    # threaded-size inputs (work ~6M, output rows = 512) must agree EXACTLY.
    var rng = Rng(20260718)
    var a = _random_tensor(rng, 64, 512)  # [N, I] -> 512 output rows
    var b = _random_tensor(rng, 64, 200)  # [N, J]
    var c1 = matmul_transpose_a(a, b)
    var c2 = matmul_transpose_a(a, b)
    assert_equal(c1.rows, 512)
    for i in range(c1.rows):
        for j in range(c1.cols):
            assert_equal(c1[i, j], c2[i, j])


def test_matmul_transpose_b_shape_mismatch_raises() raises:
    # The contraction width is the COLUMN count of BOTH operands (b is stored
    # un-transposed), so a.cols must equal b.cols. A [2, 3] and [2, 4] mismatch.
    var a = zeros_2d(2, 3)
    var b = zeros_2d(2, 4)  # 3 != 4 on the shared contraction width
    with assert_raises(contains="shape mismatch"):
        _ = matmul_transpose_b(a, b)


def test_matmul_transpose_a_matches_composed_spelling() raises:
    # matmul_transpose_a(a, b) = a^T @ b, computed without materializing a^T. It
    # must be BIT-IDENTICAL to transpose(a) @ b (assert_equal, no tolerance): it
    # is a Class A change (order-preserving), accumulating over the shared row
    # dimension n in the SAME ascending order as the composed spelling, only
    # skipping the transpose allocation and vectorizing over the output columns.
    # Any drift is a real reorder/tail bug. Shapes span sizes 1 and 2, output
    # widths not divisible by the SIMD width, the real 124M backward widths
    # (N=64 rows, out/in up to 3072), and a threaded-size case (work > 1M).
    var rng = Rng(20260716)
    var shapes = [
        (1, 1, 1),
        (3, 2, 4),
        (2, 5, 7),
        (4, 8, 13),
        (64, 384, 128),  # c_attn backward: d_out^T @ x
        (64, 512, 128),  # mlp_fc backward
        (64, 256, 128),  # tied head backward: d_logits^T @ h
        (64, 3072, 768),  # 124M mlp_fc backward — threaded (work > 1M)
    ]
    for s in range(len(shapes)):
        var nrows = shapes[s][0]  # shared contraction dimension N
        var idim = shapes[s][1]  # a has idim columns -> out rows
        var jdim = shapes[s][2]  # b has jdim columns -> out cols
        var a = _random_tensor(rng, nrows, idim)  # [N, I]
        var b = _random_tensor(rng, nrows, jdim)  # [N, J]
        var direct = matmul_transpose_a(a, b)  # a^T @ b
        # Compare against the SCALAR matmul oracle (ijk), not the SIMD `@`: both
        # matmul_transpose_a and `@` vectorize over columns, so `@` would be a
        # fast-vs-fast check. matmul is the untouched scalar reference, and both
        # accumulate the shared row dim ascending, so they are bit-identical.
        var composed = matmul(transpose(a), b)  # (a^T) @ b, scalar reference
        assert_equal(direct.rows, idim)
        assert_equal(direct.cols, jdim)
        for i in range(idim):
            for j in range(jdim):
                assert_equal(direct[i, j], composed[i, j])


def test_matmul_transpose_a_shape_mismatch_raises() raises:
    # The contraction is the shared ROW count (a^T @ b needs a.rows == b.rows).
    var a = zeros_2d(3, 2)
    var b = zeros_2d(4, 2)  # 3 != 4 shared rows
    with assert_raises(contains="shape mismatch"):
        _ = matmul_transpose_a(a, b)


def test_matvec_hand_computed() raises:
    var a = from_rows([[1.0, 2.0], [3.0, 4.0]])
    var x = [10.0, 100.0]
    var y = matvec(a, x)
    assert_almost_equal(y[0], 210.0, atol=1e-12)  # 1*10 + 2*100
    assert_almost_equal(y[1], 430.0, atol=1e-12)  # 3*10 + 4*100


def _scalar_dot(a: Tensor2D, row: Int, x: List[Float64]) -> Float64:
    # Left-to-right scalar dot of a[row, :] with x — the naive reference the
    # SIMD matvec must match to 1e-12 relative.
    var acc = 0.0
    for k in range(a.cols):
        acc += a[row, k] * x[k]
    return acc


def test_matvec_matches_scalar_dot() raises:
    # matvec is the same Class B SIMD dot as matmul_transpose_b, in vector form:
    # y[i] = dot(a[i, :], x). Pin it against the scalar left-to-right dot at the
    # real decode contraction widths and ragged tails (k not a multiple of the
    # f64 SIMD width 4), so the reassociated result agrees to ~k*eps.
    var rng = Rng(20260712)
    var ks = [1, 3, 5, 7, 13, 768, 769, 3072, 3073]
    for s in range(len(ks)):
        var k = ks[s]
        var m = 4
        var a = _random_tensor(rng, m, k)  # [m, k]
        var x = List[Float64](capacity=k)
        for _ in range(k):
            x.append(rng.normal(0.0, 1.0))
        var y = matvec(a, x)  # SIMD dot per row
        assert_equal(len(y), m)
        for i in range(m):
            _assert_close_rel(
                y[i], _scalar_dot(a, i, x), "matvec k=" + String(k)
            )


def test_matvec_shape_mismatch_raises() raises:
    var a = zeros_2d(2, 3)
    var x = [1.0, 2.0]
    with assert_raises(contains="shape mismatch"):
        _ = matvec(a, x)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
