"""Tensor and vector operations built on Tensor2D.

Read-only vector inputs are borrowed `Span[Float64, _]` views — a Tensor2D row
(via `.row()`) or a List both coerce in, so no per-row copy on the hot path;
owned results are fresh `List[Float64]`. Covers elementwise add/scale, transpose,
row/column slicing and column concat, four matmul variants (naive ijk, the
optimized ikj `@` delegate, and the two transposed-operand kernels) plus matvec,
numerically stable softmax and cross-entropy, and argmax. Softmax and logsumexp
subtract the row max before exponentiating so large logits never overflow.
"""

from std.algorithm import parallelize
from std.collections import List
from std.math import exp, log
from std.memory import memcpy
from std.sys import simd_width_of

from llm.tensor.tensor2d import Tensor2D, zeros_2d

# Below this many multiply-adds (m*n*k), matmul_transpose_b runs single-threaded:
# the parallelize dispatch costs more than it saves on the small kernels
# (c_proj's [1,768].[768,768]^T is ~0.6M and stays serial; c_attn/mlp/tied head
# clear it). Threading independent output columns is bit-identical to serial
# (each cell computed by exactly one worker in the same _simd_dot order), so this
# threshold trades only speed, never numbers — a smaller machine can lower it.
comptime _MTB_THREAD_MIN_WORK = 1_000_000


# --- elementwise ---


def add(a: Tensor2D, b: Tensor2D) raises -> Tensor2D:
    """Elementwise a + b. Allocates the result.

    Raises:
        Error: On a shape mismatch.
    """
    if a.rows != b.rows or a.cols != b.cols:
        raise Error("add shape mismatch")
    var out = zeros_2d(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            out[i, j] = a[i, j] + b[i, j]
    return out^


def scale(a: Tensor2D, s: Float64) -> Tensor2D:
    """Elementwise a * s. Allocates the result."""
    var out = zeros_2d(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            out[i, j] = a[i, j] * s
    return out^


def transpose(a: Tensor2D) -> Tensor2D:
    """Physical transpose: a new contiguous [cols, rows] tensor. Allocates."""
    var out = zeros_2d(a.cols, a.rows)
    for i in range(a.rows):
        for j in range(a.cols):
            out[j, i] = a[i, j]
    return out^


# --- column slice / concat ---


def slice_cols(a: Tensor2D, start: Int, end: Int) raises -> Tensor2D:
    """Extract the contiguous column band [start, end) from every row.

    [R, C] -> [R, end - start]. The head-split primitive: one projection [T, 3C]
    carved into contiguous [T, D] slices. Allocates the result.

    Raises:
        Error: Unless 0 <= start < end <= cols.
    """
    if not (0 <= start and start < end and end <= a.cols):
        raise Error(
            "slice_cols: need 0 <= start < end <= cols, got start="
            + String(start)
            + " end="
            + String(end)
            + " cols="
            + String(a.cols)
        )
    var width = end - start
    var out = zeros_2d(a.rows, width)
    # The [start, end) band of each row is contiguous in a's flat storage, so
    # copy it in one shot instead of element by element.
    var pa = a.data.unsafe_ptr()
    var po = out.data.unsafe_ptr()
    for r in range(a.rows):
        memcpy(dest=po + r * width, src=pa + r * a.cols + start, count=width)
    return out^


def slice_rows(a: Tensor2D, start: Int, end: Int) raises -> Tensor2D:
    """Extract the contiguous row band [start, end) with every column.

    [R, C] -> [end - start, C]. The row mirror of slice_cols; the KV-cache decode
    path uses it to view the filled prefix [0, length) of a capacity-sized buffer
    as a [t, C] tensor. Allocates the result.

    Raises:
        Error: Unless 0 <= start < end <= rows.
    """
    if not (0 <= start and start < end and end <= a.rows):
        raise Error(
            "slice_rows: need 0 <= start < end <= rows, got start="
            + String(start)
            + " end="
            + String(end)
            + " rows="
            + String(a.rows)
        )
    var height = end - start
    var out = zeros_2d(height, a.cols)
    # Each output row is a whole contiguous row of a — copy the full [start, end)
    # band of rows in one memcpy per row.
    var w = a.cols
    var pa = a.data.unsafe_ptr()
    var po = out.data.unsafe_ptr()
    for r in range(height):
        memcpy(dest=po + r * w, src=pa + (start + r) * w, count=w)
    return out^


def concat_cols(parts: List[Tensor2D]) raises -> Tensor2D:
    """Join tensors side by side along columns, in list order.

    k tensors [R, C_i] -> [R, sum(C_i)]. The head-merge primitive: H head outputs
    [T, D] concatenated back to [T, C]. Allocates the result.

    Raises:
        Error: On an empty list (no row count to adopt) or a row-count mismatch.
    """
    if len(parts) == 0:
        raise Error("concat_cols: parts is empty")
    var rows = parts[0].rows
    var total_cols = 0
    for i in range(len(parts)):
        if parts[i].rows != rows:
            raise Error(
                "concat_cols: row-count mismatch, part 0 has "
                + String(rows)
                + " rows but part "
                + String(i)
                + " has "
                + String(parts[i].rows)
            )
        total_cols += parts[i].cols
    var out = zeros_2d(rows, total_cols)
    var po = out.data.unsafe_ptr()
    var col_offset = 0
    for i in range(len(parts)):
        var part_cols = parts[i].cols
        # Each part's row is contiguous; drop it into its column band with one
        # memcpy per row.
        var pp = parts[i].data.unsafe_ptr()
        for r in range(rows):
            memcpy(
                dest=po + r * total_cols + col_offset,
                src=pp + r * part_cols,
                count=part_cols,
            )
        col_offset += part_cols
    return out^


# --- SIMD dot kernel (Class B: reassociates the k-sum) ---


def _simd_dot(a: Span[Float64, _], b: Span[Float64, _]) -> Float64:
    """Vectorized dot product of two contiguous equal-length f64 spans.

    The one reduction kernel matvec and matmul_transpose_b share. It keeps 4 * W
    independent SIMD accumulators, so the additions are regrouped relative to a
    strict left-to-right scalar sum; since float addition is not associative, the
    result differs by ~k*eps relative (~7e-13 at k=3072) — the price of the ~25x
    single-thread speedup. The regrouping is fixed by the length alone, so a given
    operand pair yields identical bits no matter who calls it.
    """
    comptime W = simd_width_of[DType.float64]()
    var n = len(a)
    var pa = a.unsafe_ptr()
    var pb = b.unsafe_ptr()
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = SIMD[DType.float64, W](0.0)
    var acc2 = SIMD[DType.float64, W](0.0)
    var acc3 = SIMD[DType.float64, W](0.0)
    var k = 0
    var step = 4 * W
    # Main body: four vectors of width W per iteration, four accumulators to hide
    # the FMA latency.
    while k + step <= n:
        acc0 = pa.load[width=W](k) * pb.load[width=W](k) + acc0
        acc1 = pa.load[width=W](k + W) * pb.load[width=W](k + W) + acc1
        acc2 = pa.load[width=W](k + 2 * W) * pb.load[width=W](k + 2 * W) + acc2
        acc3 = pa.load[width=W](k + 3 * W) * pb.load[width=W](k + 3 * W) + acc3
        k += step
    # Remaining whole vectors.
    while k + W <= n:
        acc0 = pa.load[width=W](k) * pb.load[width=W](k) + acc0
        k += W
    var acc = (acc0 + acc1 + acc2 + acc3).reduce_add()
    # Scalar tail: the ragged 0..W-1 elements that don't fill a vector.
    while k < n:
        acc += a[k] * b[k]
        k += 1
    return acc


# --- matmul family ---


def matmul(a: Tensor2D, b: Tensor2D) raises -> Tensor2D:
    """Clear ijk-order matmul [M, K] @ [K, N] -> [M, N].

    The inner loop walks b[k, j] down a column (strided). The readable/oracle
    spelling; the cache-friendly kernel is the `@` operator. Allocates.

    Raises:
        Error: On a shape mismatch.
    """
    if a.cols != b.rows:
        raise Error("matmul shape mismatch")
    var out = zeros_2d(a.rows, b.cols)
    for i in range(a.rows):
        for j in range(b.cols):
            var acc = 0.0
            for k in range(a.cols):
                acc += a[i, k] * b[k, j]
            out[i, j] = acc
    return out^


def matmul_ikj(a: Tensor2D, b: Tensor2D) raises -> Tensor2D:
    """Cache-friendly ikj-order matmul, same math as matmul but often faster.

    A thin delegate to the `@` operator (where the kernel lives to avoid an
    import cycle), kept as a named entry point for tests and benchmarks.

    Raises:
        Error: On a shape mismatch.
    """
    return a @ b


def matvec(a: Tensor2D, x: Span[Float64, _]) raises -> List[Float64]:
    """Matrix-vector product [M, K] @ [K] -> [M].

    The case that dominates single-token decoding: each output is one row of a
    dotted with x via the SIMD dot (so ~k*eps reassociation vs a scalar loop, see
    _simd_dot). Allocates the result.

    Raises:
        Error: On a shape mismatch.
    """
    if a.cols != len(x):
        raise Error("matvec shape mismatch")
    var out = List[Float64](capacity=a.rows)  # one row entry per output
    for i in range(a.rows):
        out.append(_simd_dot(a.row(i), x))
    return out^


def matmul_transpose_b(a: Tensor2D, b: Tensor2D) raises -> Tensor2D:
    """Compute c = a @ b^T without materializing b^T.

    [M, K] @ [N, K]^T -> [M, N] with c[i, j] = dot(a[i, :], b[j, :]). Both share
    the contraction width K as their column count (b is stored un-transposed), so
    the guard is a.cols == b.cols. Each cell is a SIMD dot that regroups the
    k-sum, agreeing with matmul(a, transpose(b)) within ~k*eps (a test pins 1e-12
    relative), not bit-for-bit. The tied LM head uses it to score one decode row
    against the [V, C] embedding table without a ~309 MB per-token transpose copy.
    Allocates the result.

    Raises:
        Error: On a contraction-width mismatch (a.cols != b.cols).
    """
    if a.cols != b.cols:
        raise Error(
            "matmul_transpose_b shape mismatch: a.cols="
            + String(a.cols)
            + " must equal b.cols="
            + String(b.cols)
        )
    var m = a.rows
    var n = b.rows
    var out = zeros_2d(m, n)
    if m * n * a.cols < _MTB_THREAD_MIN_WORK or n < 2:
        for i in range(m):
            var arow = a.row(i)  # contiguous [K] view, reused across all j
            for j in range(n):
                out[i, j] = _simd_dot(arow, b.row(j))
        return out^

    # Threaded: partition the OUTPUT columns j (= rows of b) into contiguous
    # blocks, one worker per block. Every out[i, j] is written by exactly one
    # worker, computed by the same _simd_dot as the serial path — no shared
    # accumulators, no atomics, a static partition — so the result is
    # bit-identical to single-threaded and stable run to run. The output is
    # written through a raw pointer to disjoint cells; a and b are shared
    # immutably.
    var pout = out.data.unsafe_ptr()
    var nblocks = 32 if n >= 32 else n
    var block = (n + nblocks - 1) // nblocks

    @parameter
    def col_block(t: Int):
        var j0 = t * block
        var j1 = j0 + block
        if j1 > n:
            j1 = n
        for i in range(m):
            var arow = a.row(i)
            for j in range(j0, j1):
                pout[i * n + j] = _simd_dot(arow, b.row(j))

    parallelize[col_block](nblocks)
    return out^


def matmul_transpose_a(a: Tensor2D, b: Tensor2D) raises -> Tensor2D:
    """Compute c = a^T @ b without materializing a^T.

    [N, I]^T @ [N, J] -> [I, J] with c[i, j] = sum_n a[n, i] * b[n, j]. Both share
    the contraction dimension N as their row count, so the guard is
    a.rows == b.rows. The mirror of matmul_transpose_b for the backward pass
    (Linear's dW, attention's dV/dK/dQ, the tied head's d_table). Order-preserving:
    it accumulates over n ascending in exactly transpose(a) @ b (ikj) order,
    vectorizing only the independent columns j, so it is bit-identical to the
    composed spelling. Skips the [I, N] transpose allocation (28-65% of each
    backward product once the matmul was vectorized). Threads over output rows i.
    Allocates.

    Raises:
        Error: On a row-count mismatch (a.rows != b.rows).
    """
    if a.rows != b.rows:
        raise Error(
            "matmul_transpose_a shape mismatch: a.rows="
            + String(a.rows)
            + " must equal b.rows="
            + String(b.rows)
        )
    comptime W = simd_width_of[DType.float64]()
    var nrows = a.rows  # shared contraction N
    var idim = a.cols  # output rows
    var jdim = b.cols  # output cols
    var out = zeros_2d(idim, jdim)
    var pout = out.data.unsafe_ptr()
    var pa = a.data.unsafe_ptr()
    var pb = b.data.unsafe_ptr()

    # One output row block: for each output row i, accumulate a[n, i] * b[n, :]
    # over n ascending (SIMD over j), matching transpose(a) @ b's ikj order.
    @parameter
    def row_block(i0: Int, i1: Int):
        for i in range(i0, i1):
            var ibase = i * jdim
            for nn in range(nrows):
                var a_ni = pa[nn * idim + i]  # a[n, i]
                var a_vec = SIMD[DType.float64, W](a_ni)
                var obase = nn * jdim
                var j = 0
                while j + W <= jdim:
                    var acc = pout.load[width=W](ibase + j) + a_vec * pb.load[
                        width=W
                    ](obase + j)
                    pout.store(ibase + j, acc)
                    j += W
                while j < jdim:
                    pout[ibase + j] += a_ni * pb[obase + j]
                    j += 1

    if idim * nrows * jdim < _MTB_THREAD_MIN_WORK or idim < 2:
        row_block(0, idim)
        return out^

    var nblocks = 32 if idim >= 32 else idim
    var block = (idim + nblocks - 1) // nblocks

    @parameter
    def worker(t: Int):
        var i0 = t * block
        var i1 = i0 + block
        if i1 > idim:
            i1 = idim
        row_block(i0, i1)

    parallelize[worker](nblocks)
    return out^


# --- softmax ---


def softmax_row(input: Span[Float64, _]) -> List[Float64]:
    """Numerically stable softmax over one vector.

    Subtracts the row max before exponentiating so the largest exponent is
    exp(0) = 1 and nothing overflows. An empty input yields an empty output.
    Allocates the result.
    """
    var n = len(input)
    var out = List[Float64](
        capacity=n
    )  # one exp() per input, reserved up front
    if n == 0:
        return out^

    var max_value = input[0]
    for i in range(1, n):
        if input[i] > max_value:
            max_value = input[i]

    var denom = 0.0
    for i in range(n):
        var e = exp(input[i] - max_value)
        out.append(e)
        denom += e

    for i in range(n):
        out[i] = out[i] / denom
    return out^


def softmax_rows(scores: Tensor2D) -> Tensor2D:
    """Row-wise stable softmax over a Tensor2D — the shape attention uses.

    Each row is normalized independently. A zero-column input has no logits to
    normalize, so it returns unchanged rather than reading out of bounds.
    Allocates the result.
    """
    var out = zeros_2d(scores.rows, scores.cols)
    if scores.cols == 0:
        return out^
    for r in range(scores.rows):
        var max_value = scores[r, 0]
        for c in range(1, scores.cols):
            if scores[r, c] > max_value:
                max_value = scores[r, c]
        var denom = 0.0
        for c in range(scores.cols):
            var e = exp(scores[r, c] - max_value)
            out[r, c] = e
            denom += e
        for c in range(scores.cols):
            out[r, c] = out[r, c] / denom
    return out^


def softmax_rows_backward(
    weights: Tensor2D, d_weights: Tensor2D
) raises -> Tensor2D:
    """VJP of softmax_rows: given W = softmax_rows(scores) (the output, not scores)
    and d_weights = dL/dW, return dL/dscores.

    Per row with p = softmax(s), the row Jacobian dp_i/ds_j = p_i (delta_ij - p_j)
    contracts to dS = W * (dW - rowsum(dW * W)). The subtracted rowsum is one
    scalar shared across the row, so a blocked entry (W ~ 0) both contributes ~0
    and receives ~0 back — why masked positions leak no gradient. Shapes
    [R, C] -> [R, C]. Allocates the result.

    Raises:
        Error: On a shape mismatch.
    """
    if weights.rows != d_weights.rows or weights.cols != d_weights.cols:
        raise Error("softmax_rows_backward shape mismatch")
    var out = zeros_2d(weights.rows, weights.cols)
    for r in range(weights.rows):
        var dot = 0.0
        for c in range(weights.cols):
            dot += d_weights[r, c] * weights[r, c]
        for c in range(weights.cols):
            out[r, c] = weights[r, c] * (d_weights[r, c] - dot)
    return out^


def softmax_row_temperature(
    input: Span[Float64, _], temperature: Float64
) raises -> List[Float64]:
    """Softmax of input / temperature. T < 1 sharpens, T > 1 flattens.

    Does NOT divide first then delegate to softmax_row: dividing first would
    overflow (a near-zero T sends a large logit to +inf before any max
    subtraction). Instead it subtracts the row max and divides the difference,
    exp((x_i - max) / T), which is algebraically identical but keeps every
    exponent <= 0. An empty input yields an empty output.

    Raises:
        Error: If temperature <= 0.
    """
    if temperature <= 0.0:
        raise Error("temperature must be positive")
    var n = len(input)
    var out = List[Float64](
        capacity=n
    )  # one exp() per input, reserved up front
    if n == 0:
        return out^

    var max_value = input[0]
    for i in range(1, n):
        if input[i] > max_value:
            max_value = input[i]

    var denom = 0.0
    for i in range(n):
        var e = exp((input[i] - max_value) / temperature)
        out.append(e)
        denom += e

    for i in range(n):
        out[i] = out[i] / denom
    return out^


# --- cross-entropy ---


def logsumexp(x: Span[Float64, _]) -> Float64:
    """Stable log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x)))).

    Assumes a non-empty x.
    """
    var n = len(x)
    var m = x[0]
    for i in range(1, n):
        if x[i] > m:
            m = x[i]
    var s = 0.0
    for i in range(n):
        s += exp(x[i] - m)
    return m + log(s)


def cross_entropy_one(logits: Span[Float64, _], target: Int) raises -> Float64:
    """Cross-entropy loss for one position: logsumexp(logits) - logits[target].

    The stable form of -log(softmax(logits)[target]).

    Raises:
        Error: If target is out of range.
    """
    if target < 0 or target >= len(logits):
        raise Error("target out of range")
    return logsumexp(logits) - logits[target]


def cross_entropy_grad(
    logits: Span[Float64, _], target: Int
) raises -> List[Float64]:
    """Gradient of cross_entropy_one wrt logits: softmax(logits) - onehot(target).

    That is p_i - y_i, the backbone of every training step. Allocates the result.

    Raises:
        Error: On an out-of-range target — the same guard as cross_entropy_one, so
            loss and gradient reject bad targets symmetrically.
    """
    if target < 0 or target >= len(logits):
        raise Error("target out of range")
    var p = softmax_row(logits)
    p[target] = p[target] - 1.0
    return p^


def cross_entropy_rows(logits: Tensor2D, targets: List[Int]) raises -> Float64:
    """Mean cross-entropy over N rows: (1/N) * sum_i CE(logits[i, :], targets[i]).

    logits [N, V]; targets has length N (one target id per row). The mean, not the
    sum, so the gradient scale is independent of batch size. Borrows each row as a
    Span (no per-row copy).

    Raises:
        Error: On a length mismatch, an empty batch (no mean is defined), or an
            out-of-range target.
    """
    var n = logits.rows
    if len(targets) != n:
        raise Error(
            "cross_entropy_rows: targets length "
            + String(len(targets))
            + " must equal logits rows "
            + String(n)
        )
    if n == 0:
        raise Error("cross_entropy_rows: empty batch has no mean loss")
    var total = 0.0
    for i in range(n):
        total += cross_entropy_one(logits.row(i), targets[i])
    return total / Float64(n)


def cross_entropy_rows_backward(
    logits: Tensor2D, targets: List[Int]
) raises -> Tensor2D:
    """Gradient of cross_entropy_rows wrt logits: (softmax(logits) - onehot) / N.

    logits [N, V] -> [N, V], row by row. Each row is cross_entropy_grad (sums to
    0) scaled by the mean's 1/N factor, so every row still sums to 0. Borrows each
    row as a Span (no per-row copy); allocates the result.

    Raises:
        Error: On a length mismatch, empty batch, or out-of-range target — the
            same guards as the loss.
    """
    var n = logits.rows
    if len(targets) != n:
        raise Error(
            "cross_entropy_rows_backward: targets length "
            + String(len(targets))
            + " must equal logits rows "
            + String(n)
        )
    if n == 0:
        raise Error("cross_entropy_rows_backward: empty batch has no gradient")
    var inv_n = 1.0 / Float64(n)
    var out = zeros_2d(n, logits.cols)
    for i in range(n):
        var g = cross_entropy_grad(
            logits.row(i), targets[i]
        )  # softmax - onehot
        for j in range(logits.cols):
            out[i, j] = g[j] * inv_n
    return out^


# --- argmax ---


def argmax(values: Span[Float64, _]) -> Int:
    """Index of the maximum value, first-wins on ties (strict >).

    The tie rule keeps greedy decoding deterministic. Assumes a non-empty input.
    """
    var best_idx = 0
    var best_value = values[0]
    for i in range(1, len(values)):
        if values[i] > best_value:
            best_value = values[i]
            best_idx = i
    return best_idx
