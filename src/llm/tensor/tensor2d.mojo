# A small educational 2-D tensor over flat row-major Float64 storage.
#
# This is not the final high-performance tensor — it is the readable abstraction
# we reason about shapes, indexing, and gradients on before optimizing. Storage
# is a single `List[Float64]` in row-major order: element (row, col) lives at
# `row * cols + col`.
#
# Access comes in two flavors on purpose. `[i, j]` (one ref-returning
# __getitem__) is unchecked and cheap — the hot path. It hands back a *reference*
# into the flat buffer, tagged with that buffer as its origin, so the single
# subscript serves reads, writes (`t[i, j] = v`), and in-place updates
# (`t[i, j] += v`) alike — no separate setter to keep in sync. `.at(i, j)`
# bounds-checks and therefore raises — the debugging/test path. Fast by default,
# checked on demand.

from std.collections import List
from std.sys import simd_width_of


@fieldwise_init
struct Tensor2D(Copyable, Movable, Writable):
    var rows: Int
    var cols: Int
    var data: List[Float64]  # flat row-major [rows, cols]

    def size(self) -> Int:
        return self.rows * self.cols

    def offset(self, row: Int, col: Int) -> Int:
        # Row-major flat index. No bounds check — callers that need one use at().
        return row * self.cols + col

    def __getitem__(ref self, row: Int, col: Int) -> ref[self.data] Float64:
        # Unchecked ref access (hot path): one method serves read, write, and +=.
        # The returned reference borrows self.data as its origin, so assigning or
        # accumulating through the subscript mutates the buffer directly — no
        # separate setter. (Origin must name the field, not self.)
        return self.data[self.offset(row, col)]

    def at(self, row: Int, col: Int) raises -> Float64:
        # Bounds-checked read. Raises if (row, col) is outside the shape.
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise Error("Tensor2D index out of range")
        return self.data[self.offset(row, col)]

    def row(ref self, r: Int) -> Span[Float64, origin_of(self.data)]:
        # A borrowed [cols] view of row r — no copy, no allocation. The Span
        # borrows self.data as its origin, so it reads (and writes) straight
        # through the flat buffer for as long as the view is live; the compiler
        # forbids mutating the tensor while the view is held. Unchecked like
        # __getitem__ (the hot path): a bad r indexes the buffer, it does not
        # raise. What it teaches: an origin-tagged view replacing a per-row copy,
        # at the cost of the borrow — the tensor is pinned immutable underneath.
        var start = r * self.cols
        return Span(self.data)[start : start + self.cols]

    def fill(mut self, value: Float64):
        # Overwrite every element with `value`. Mutates in place.
        for i in range(self.size()):
            self.data[i] = value

    def write_to(self, mut writer: Some[Writer]):
        # Printable form: a shape header plus a capped preview of the buffer,
        # e.g. `Tensor2D[3, 4](1.5, 2.0, 0.0, …)`. The preview stops at 8 values
        # (a trailing `…` marks the truncation) so printing a big table — a
        # [50257, 768] embedding — stays one short line, not millions of floats.
        writer.write("Tensor2D[", self.rows, ", ", self.cols, "](")
        var n = self.size()
        var limit = n if n < 8 else 8
        for i in range(limit):
            if i > 0:
                writer.write(", ")
            writer.write(self.data[i])
        if n > 8:
            writer.write(", …")
        writer.write(")")

    def __matmul__(self, other: Tensor2D) raises -> Tensor2D:
        # Matrix multiply via the `@` operator: [M, K] @ [K, N] -> [M, N] in ikj
        # loop order. The inner loop walks other[k, *] and result[i, *]
        # contiguously along rows, which is cache-friendly and vectorizes: the W
        # output columns result[i, j..j+W] are updated in one SIMD fused step
        # result[i, j:] += a_ik * other[k, j:], with a scalar tail for the ragged
        # remainder. Raises on a shape mismatch; allocates the result. (The
        # teaching contrast, the plain ijk kernel, lives in ops.matmul.)
        #
        # This is a Class A (order-preserving) vectorization: each output cell
        # result[i, j] still accumulates over k in the SAME ascending order as the
        # scalar loop — SIMD only groups the independent j columns, never the k
        # reduction — so every cell is bit-identical to the scalar ikj result (a
        # test pins exact equality against a scalar reference). Contrast
        # _simd_dot / matmul_transpose_b, which DO regroup the k-sum (Class B).
        if self.cols != other.rows:
            raise Error("matmul shape mismatch")
        comptime W = simd_width_of[DType.float64]()
        var n = other.cols
        var result = zeros_2d(self.rows, n)
        var pres = result.data.unsafe_ptr()
        var poth = other.data.unsafe_ptr()
        for i in range(self.rows):
            var ibase = i * n
            for k in range(self.cols):
                # self[i, k] is constant across j; hoist it and broadcast it.
                var a_ik = self[i, k]
                var a_vec = SIMD[DType.float64, W](a_ik)
                var obase = k * n
                var j = 0
                while j + W <= n:
                    var acc = pres.load[width=W](ibase + j) + a_vec * poth.load[
                        width=W
                    ](obase + j)
                    pres.store(ibase + j, acc)
                    j += W
                while j < n:
                    result[i, j] += a_ik * other[k, j]
                    j += 1
        return result^


def zeros_2d(rows: Int, cols: Int) -> Tensor2D:
    # A rows x cols tensor of zeros. Allocates the flat buffer in one shot:
    # length= sizes it and fill= zeros it, no append loop to grow it element by
    # element (which would reallocate as it outgrows its capacity).
    var data = List[Float64](length=rows * cols, fill=0.0)
    return Tensor2D(rows, cols, data^)


def full_2d(rows: Int, cols: Int, value: Float64) -> Tensor2D:
    # A rows x cols tensor filled with `value`.
    var t = zeros_2d(rows, cols)
    t.fill(value)
    return t^


def ones_2d(rows: Int, cols: Int) -> Tensor2D:
    # A rows x cols tensor of ones.
    return full_2d(rows, cols, 1.0)


def from_rows(values: List[List[Float64]]) raises -> Tensor2D:
    # Build a tensor from a list of equal-length rows. Raises on zero rows or
    # ragged input, so a mis-shaped literal fails at construction.
    var rows = len(values)
    if rows == 0:
        raise Error("from_rows needs at least one row")
    var cols = len(values[0])
    var t = zeros_2d(rows, cols)
    for r in range(rows):
        if len(values[r]) != cols:
            raise Error("ragged rows are not allowed")
        for c in range(cols):
            t[r, c] = values[r][c]
    return t^
