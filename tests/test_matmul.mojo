"""Tests for matmul, matmul_ikj, and matvec.

The hand-computed small cases catch wrong bounds, row/column confusion, and
offset bugs; the fast SIMD/threaded kernels are pinned against scalar reference
spellings, bit-for-bit where the summation order is preserved and to a mixed
tolerance where the SIMD dot reassociates the contraction.
"""

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
    """Build a [rows, cols] tensor of seeded normal draws.

    Args:
        rng: Seeded generator, advanced in place.
        rows: Row count.
        cols: Column count.

    Returns:
        A [rows, cols] tensor. Allocates.
    """
    var t = zeros_2d(rows, cols)
    for r in range(rows):
        for c in range(cols):
            t[r, c] = rng.normal(0.0, 1.0)
    return t^


def test_matmul_small() raises:
    """`matmul` on a hand-computed [2,3] @ [3,2] case."""
    var a = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    var c = matmul(a, b)
    assert_almost_equal(c[0, 0], 58.0, atol=1e-12)
    assert_almost_equal(c[0, 1], 64.0, atol=1e-12)
    assert_almost_equal(c[1, 0], 139.0, atol=1e-12)
    assert_almost_equal(c[1, 1], 154.0, atol=1e-12)


def test_orders_agree() raises:
    """`matmul` (ijk) and matmul_ikj agree elementwise on a small case."""
    var a = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    var c1 = matmul(a, b)
    var c2 = matmul_ikj(a, b)
    for i in range(c1.rows):
        for j in range(c1.cols):
            assert_almost_equal(c1[i, j], c2[i, j], atol=1e-9)


def test_at_operator_matches_matmul() raises:
    """`a @ b` matches the clear ijk matmul bit-identically on a non-square case.
    """
    # Both accumulate over k in the same ascending order, so the per-element sums
    # come out bit-identical, not merely close. A non-square [2,3] @ [3,2] catches
    # a row/column or bounds slip a square case could mask.
    var a = from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    var c1 = matmul(a, b)
    var c2 = a @ b
    for i in range(c1.rows):
        for j in range(c1.cols):
            assert_almost_equal(c1[i, j], c2[i, j], atol=1e-12)


def _scalar_ikj(a: Tensor2D, b: Tensor2D) raises -> Tensor2D:
    """Plain scalar ikj matmul, the reference the SIMD-over-j `@` must match.

    Same loop order as `@`, no vectorization: each out[i, j] accumulates over k
    ascending.

    Args:
        a: Left operand [M, K].
        b: Right operand [K, N].

    Returns:
        out [M, N]. Allocates.
    """
    var out = zeros_2d(a.rows, b.cols)
    for i in range(a.rows):
        for k in range(a.cols):
            var a_ik = a[i, k]
            for j in range(b.cols):
                out[i, j] += a_ik * b[k, j]
    return out^


def test_at_operator_bit_identical_to_scalar_ikj() raises:
    """`@` (SIMD over output columns) is bit-identical to the scalar ikj loop.
    """
    # Each cell still sums k in the same ascending order, so the two must be
    # EXACTLY equal — SIMD-over-j reorders nothing that gets summed. Output widths
    # n that are NOT multiples of the f64 SIMD width 4 exercise the remainder loop.
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
        ),  # single-threaded (work 404k < 1M), ragged n exercises the tail
        (
            64,
            128,
            129,
        ),  # THREADED (work ~1.06M >= 1M, m>=2) AND ragged n (129 % 4, % 8 != 0):
        # exercises the remainder loop inside a parallelize worker, where the tail
        # must write through the raw pointer, not the mut `result` capture
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
    """The threaded `@` path is run-to-run bit-identical on the same inputs."""
    # The static row partition above the work threshold must not depend on
    # scheduling. Two calls on threaded-size inputs (work ~8M, m=64) agree EXACTLY.
    var rng = Rng(20260715)
    var a = _random_tensor(rng, 64, 256)
    var b = _random_tensor(rng, 256, 512)
    var c1 = a @ b
    var c2 = a @ b
    for i in range(c1.rows):
        for j in range(c1.cols):
            assert_equal(c1[i, j], c2[i, j])


def test_at_operator_shape_mismatch_raises() raises:
    """`@` raises on a shape mismatch (its own guard)."""
    var a = zeros_2d(2, 3)
    var b = zeros_2d(2, 2)  # 3 != 2
    with assert_raises(contains="shape mismatch"):
        _ = a @ b


def test_matmul_shape_mismatch_raises() raises:
    """`matmul` raises on a shape mismatch."""
    var a = zeros_2d(2, 3)
    var b = zeros_2d(2, 2)  # 3 != 2
    with assert_raises(contains="shape mismatch"):
        _ = matmul(a, b)


def test_matmul_ikj_shape_mismatch_raises() raises:
    """`matmul_ikj` raises on a shape mismatch (its own guard)."""
    var a = zeros_2d(2, 3)
    var b = zeros_2d(2, 2)  # 3 != 2
    with assert_raises(contains="shape mismatch"):
        _ = matmul_ikj(a, b)


def test_matmul_transpose_b_hand_computed() raises:
    """`matmul_transpose_b` computes a @ b^T on a hand-computed [2,3]x[2,3] case.
    """
    # c[i, j] = sum_k a[i, k] * b[j, k]. Each row of a dotted with each row of b:
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
    """Assert |got - want| <= 1e-12*|want| + 1e-13 (relative tolerance).

    The rtol term covers the SIMD dot reassociating the k-sum, so it agrees with
    the scalar spelling only to ~k*eps (measured max ~2.3e-13). A real
    tail/reorder bug (errors orders larger) is caught; legitimate reassociation
    passes.

    Args:
        got: Value under test.
        want: Reference value.
        tag: Label included in the failure message.

    Raises:
        Error: If the values are not close.
    """
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
    """`matmul_transpose_b`(a, b) matches matmul(a, transpose(b)) to 1e-12 relative.
    """
    # The multi-accumulator SIMD dot reassociates the contraction vs the scalar
    # left-to-right order, so the two agree within ~k*eps, NOT bit-for-bit; matmul
    # stays the scalar ijk oracle. Shapes span the real contraction widths (k=768,
    # k=3072), [1, k] decode rows, and ragged tails (k not a multiple of 4).
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
    """The threaded matmul_transpose_b is run-to-run bit-identical."""
    # Output columns partition statically across workers (no shared accumulators),
    # so two calls on the same threaded-size inputs (work ~6M, n=4096) agree
    # EXACTLY.
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
    """The threaded matmul_transpose_b is bit-identical to serial _simd_dot."""
    # matvec(b, a.row(i)) dots each row of b against a[i, :] through the SAME
    # _simd_dot, serially, so matmul_transpose_b(a, b)[i, j] must EXACTLY equal
    # matvec(b, a.row(i))[j]. A threaded-size b (n=4096) forces the parallel branch.
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
    """The threaded matmul_transpose_a is run-to-run bit-identical."""
    # The static output-row partition above the work threshold must not depend on
    # scheduling. Two calls on the same threaded-size inputs (work ~6M, output rows
    # = 512) agree EXACTLY.
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
    """`matmul_transpose_b` raises when the shared contraction widths differ."""
    # b is stored un-transposed, so a.cols must equal b.cols.
    var a = zeros_2d(2, 3)
    var b = zeros_2d(2, 4)  # 3 != 4 on the shared contraction width
    with assert_raises(contains="shape mismatch"):
        _ = matmul_transpose_b(a, b)


def test_matmul_transpose_a_matches_composed_spelling() raises:
    """`matmul_transpose_a`(a, b) is bit-identical to transpose(a) @ b."""
    # It accumulates over the shared row dimension n in the SAME ascending order as
    # the composed spelling, only skipping the transpose allocation, so any drift
    # is a real reorder/tail bug. Shapes span the real backward widths and a
    # threaded-size case.
    var rng = Rng(20260716)
    var shapes = [
        (1, 1, 1),
        (3, 2, 4),
        (2, 5, 7),
        (4, 8, 13),
        (64, 384, 128),  # c_attn backward: d_out^T @ x
        (64, 512, 128),  # mlp_fc backward
        (64, 256, 128),  # tied head backward: d_logits^T @ h
        (64, 3072, 768),  # mlp_fc backward — threaded (work > 1M)
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
    """`matmul_transpose_a` raises when the shared row counts differ."""
    # a^T @ b needs a.rows == b.rows.
    var a = zeros_2d(3, 2)
    var b = zeros_2d(4, 2)  # 3 != 4 shared rows
    with assert_raises(contains="shape mismatch"):
        _ = matmul_transpose_a(a, b)


def test_matvec_hand_computed() raises:
    """`matvec` on a hand-computed [2,2] @ [2] case."""
    var a = from_rows([[1.0, 2.0], [3.0, 4.0]])
    var x = [10.0, 100.0]
    var y = matvec(a, x)
    assert_almost_equal(y[0], 210.0, atol=1e-12)  # 1*10 + 2*100
    assert_almost_equal(y[1], 430.0, atol=1e-12)  # 3*10 + 4*100


def _scalar_dot(a: Tensor2D, row: Int, x: List[Float64]) -> Float64:
    """Left-to-right scalar dot of a[row, :] with x — the naive reference.

    Args:
        a: Tensor whose row is dotted.
        row: Row index into a.
        x: Vector of the same length as a's columns.

    Returns:
        The dot product.
    """
    var acc = 0.0
    for k in range(a.cols):
        acc += a[row, k] * x[k]
    return acc


def test_matvec_matches_scalar_dot() raises:
    """`matvec` matches the scalar left-to-right dot to 1e-12 relative."""
    # matvec is the SIMD dot in vector form: y[i] = dot(a[i, :], x). Real decode
    # contraction widths and ragged tails (k not a multiple of 4) exercise the
    # reassociation and the remainder loop.
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
    """`matvec` raises when the vector length does not match the column count.
    """
    var a = zeros_2d(2, 3)
    var x = [1.0, 2.0]
    with assert_raises(contains="shape mismatch"):
        _ = matvec(a, x)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
