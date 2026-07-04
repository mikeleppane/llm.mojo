# Tensor and vector operations built on Tensor2D and List[Float64].
#
# Everything a small LLM leans on before attention: elementwise add/scale, a
# physical transpose, matmul in two loop orders (ijk clear, ikj cache-friendly)
# plus the matvec special case, numerically stable softmax (row, rows, and
# temperature variants), cross-entropy via logsumexp with its p-y gradient, and
# argmax for greedy decoding.
#
# Numerical discipline throughout: softmax and logsumexp subtract the row max
# before exponentiating so large logits never overflow, and floats are compared
# with tolerances (never ==) by the tests that exercise these.

from std.collections import List
from std.math import exp, log

from llm.tensor.tensor2d import Tensor2D, zeros_2d


# --- elementwise ---


def add(a: Tensor2D, b: Tensor2D) raises -> Tensor2D:
    # Elementwise a + b. Raises on a shape mismatch (a caller error worth
    # surfacing). Allocates the result.
    if a.rows != b.rows or a.cols != b.cols:
        raise Error("add shape mismatch")
    var out = zeros_2d(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            out[i, j] = a[i, j] + b[i, j]
    return out^


def scale(a: Tensor2D, s: Float64) -> Tensor2D:
    # Elementwise a * s. Cannot fail, so non-raising. Allocates the result.
    var out = zeros_2d(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            out[i, j] = a[i, j] * s
    return out^


def transpose(a: Tensor2D) -> Tensor2D:
    # Physical transpose: a new contiguous [cols, rows] tensor. The stride-swap
    # alternative (a view) waits for the performance chapters where the tradeoff
    # can be measured. Allocates the result.
    var out = zeros_2d(a.cols, a.rows)
    for i in range(a.rows):
        for j in range(a.cols):
            out[j, i] = a[i, j]
    return out^


# --- column slice / concat ---


def slice_cols(a: Tensor2D, start: Int, end: Int) raises -> Tensor2D:
    # Extract the contiguous column band [start, end) from every row:
    # [R, C] -> [R, end - start]. This is the head-split primitive — one
    # projection [T, 3C] carved into contiguous [T, D] slices. Raises unless
    # 0 <= start < end <= C (an empty or reversed range is a caller bug, not a
    # silent zero-width result). Allocates the result.
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
    for r in range(a.rows):
        for c in range(width):
            out[r, c] = a[r, start + c]
    return out^


def concat_cols(parts: List[Tensor2D]) raises -> Tensor2D:
    # Join tensors side by side along columns, in list order:
    # k tensors [R, C_i] -> [R, sum(C_i)]. The head-merge primitive — H head
    # outputs [T, D] concatenated back to [T, C]. Raises on an empty list (no
    # row count to adopt) or any row-count mismatch (a ragged concat is a caller
    # bug). Allocates the result.
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
    var col_offset = 0
    for i in range(len(parts)):
        var part_cols = parts[i].cols
        for r in range(rows):
            for c in range(part_cols):
                out[r, col_offset + c] = parts[i][r, c]
        col_offset += part_cols
    return out^


# --- matmul family ---


def matmul(a: Tensor2D, b: Tensor2D) raises -> Tensor2D:
    # Clear ijk-order matmul [M, K] @ [K, N] -> [M, N]. The inner loop walks
    # b[k, j] down a column (strided). Raises on a shape mismatch.
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
    # Same math as matmul, ikj loop order: the inner loop walks b[k, *] and
    # out[i, *] contiguously along rows, which is cache-friendly and often
    # several times faster on non-trivial sizes. Raises on a shape mismatch.
    if a.cols != b.rows:
        raise Error("matmul shape mismatch")
    var out = zeros_2d(a.rows, b.cols)
    for i in range(a.rows):
        for k in range(a.cols):
            # a[i, k] is constant across j; hoist it out of the inner loop.
            var a_ik = a[i, k]
            for j in range(b.cols):
                out[i, j] += a_ik * b[k, j]
    return out^


def matvec(a: Tensor2D, x: List[Float64]) raises -> List[Float64]:
    # Matrix-vector product [M, K] @ [K] -> [M]. The special case that dominates
    # single-token decoding. The inner loop walks a[i, *] contiguously. Raises on
    # a shape mismatch; allocates the result.
    if a.cols != len(x):
        raise Error("matvec shape mismatch")
    var out = List[Float64]()
    for i in range(a.rows):
        var acc = 0.0
        for k in range(a.cols):
            acc += a[i, k] * x[k]
        out.append(acc)
    return out^


# --- softmax ---


def softmax_row(input: List[Float64]) -> List[Float64]:
    # Numerically stable softmax over one vector: subtract the row max before
    # exponentiating so the largest exponent is exp(0) = 1 and nothing overflows.
    # An empty input yields an empty output. Allocates the result.
    var n = len(input)
    var out = List[Float64]()
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
    # Row-wise stable softmax over a Tensor2D — the shape attention uses. Each
    # row is normalized independently. Allocates the result. A zero-column input
    # has no logits to normalize, so it returns unchanged (mirrors softmax_row's
    # empty-input handling) rather than reading scores[r, 0] out of bounds.
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


def softmax_row_temperature(
    input: List[Float64], temperature: Float64
) raises -> List[Float64]:
    # Softmax of input / temperature. T < 1 sharpens, T > 1 flattens. Raises if
    # temperature <= 0; an empty input yields an empty output.
    #
    # Stability note: this does NOT delegate to softmax_row(input / temperature).
    # Dividing first would overflow — a near-zero T sends a large logit to +inf
    # before any max subtraction can help, and inf - inf is NaN. Instead we
    # subtract the row max first and divide the *difference*:
    #     exp((x_i - max) / T)
    # which is algebraically identical (exp(m/T) cancels in the ratio) but keeps
    # every exponent <= 0, so the argmax term is exp(0) = 1 and the rest underflow
    # safely to 0. The chapter's "thin wrapper" form is subtly wrong at extreme T;
    # this is the honest stable version.
    if temperature <= 0.0:
        raise Error("temperature must be positive")
    var n = len(input)
    var out = List[Float64]()
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


def logsumexp(x: List[Float64]) -> Float64:
    # Stable log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x)))). Assumes a
    # non-empty x.
    var n = len(x)
    var m = x[0]
    for i in range(1, n):
        if x[i] > m:
            m = x[i]
    var s = 0.0
    for i in range(n):
        s += exp(x[i] - m)
    return m + log(s)


def cross_entropy_one(logits: List[Float64], target: Int) raises -> Float64:
    # Cross-entropy loss for one position: logsumexp(logits) - logits[target],
    # the stable form of -log(softmax(logits)[target]). Raises if target is out
    # of range.
    if target < 0 or target >= len(logits):
        raise Error("target out of range")
    return logsumexp(logits) - logits[target]


def cross_entropy_grad(
    logits: List[Float64], target: Int
) raises -> List[Float64]:
    # Gradient of cross_entropy_one wrt logits: softmax(logits) - onehot(target),
    # i.e. p_i - y_i. Cheap, bounded, and the backbone of every training step.
    # Allocates the result. Raises on an out-of-range target — the same guard
    # cross_entropy_one uses, so the loss and its gradient reject bad targets
    # symmetrically instead of silently writing out of bounds.
    if target < 0 or target >= len(logits):
        raise Error("target out of range")
    var p = softmax_row(logits)
    p[target] = p[target] - 1.0
    return p^


# --- argmax ---


def argmax(values: List[Float64]) -> Int:
    # Index of the maximum value, first-wins on ties (strict >). The tie rule is
    # deliberate: greedy decoding must be deterministic, and two logits can
    # genuinely tie. Assumes a non-empty input.
    var best_idx = 0
    var best_value = values[0]
    for i in range(1, len(values)):
        if values[i] > best_value:
            best_value = values[i]
            best_idx = i
    return best_idx
