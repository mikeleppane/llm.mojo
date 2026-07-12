"""A small educational 2-D tensor over flat row-major Float64 storage.

The readable abstraction for reasoning about shapes, indexing, and gradients
before optimizing. Element (row, col) lives at `row * cols + col`. Access comes
in two flavors: `[i, j]` is unchecked, cheap, and returns a mutable reference
(the hot path); `.at(i, j)` bounds-checks and raises (the debugging/test path).
"""

from std.algorithm import parallelize
from std.collections import List
from std.sys import simd_width_of

# Below this many multiply-adds (m*n*k), the @ matmul runs single-threaded: the
# parallelize dispatch costs more than it saves on the small products (decode's
# attention weights @ v is tiny and stays serial). Threading independent output
# ROWS is bit-identical to serial — each row computed by one worker in the same
# order — so this threshold trades only speed, never numbers.
comptime _MATMUL_THREAD_MIN_WORK = 1_000_000


@fieldwise_init
struct Tensor2D(Copyable, Movable, Writable):
    """A [rows, cols] tensor over flat row-major Float64 storage."""

    var rows: Int
    var cols: Int
    var data: List[Float64]  # flat row-major [rows, cols]

    def size(self) -> Int:
        """Total element count rows * cols."""
        return self.rows * self.cols

    def offset(self, row: Int, col: Int) -> Int:
        """Row-major flat index. No bounds check — callers that need one use at().
        """
        return row * self.cols + col

    def __getitem__(ref self, row: Int, col: Int) -> ref[self.data] Float64:
        """Unchecked ref access (hot path): one method serves read, write, and +=.

        The returned reference borrows self.data as its origin, so assigning or
        accumulating through the subscript mutates the buffer directly.
        """
        return self.data[self.offset(row, col)]

    def at(self, row: Int, col: Int) raises -> Float64:
        """Bounds-checked read.

        Raises:
            Error: If (row, col) is outside the shape.
        """
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise Error("Tensor2D index out of range")
        return self.data[self.offset(row, col)]

    def row(ref self, r: Int) -> Span[Float64, origin_of(self.data)]:
        """Return a borrowed [cols] view of row r — no copy, no allocation.

        The Span borrows self.data as its origin, so it reads and writes straight
        through the flat buffer while live; the compiler forbids mutating the
        tensor meanwhile. Unchecked like __getitem__: a bad r indexes the buffer,
        it does not raise.
        """
        var start = r * self.cols
        return Span(self.data)[start : start + self.cols]

    def fill(mut self, value: Float64):
        """Overwrite every element with `value`. Mutates in place."""
        for i in range(self.size()):
            self.data[i] = value

    def write_to(self, mut writer: Some[Writer]):
        """Write a shape header plus a capped preview, e.g. `Tensor2D[3, 4](...)`.

        The preview stops at 8 values (trailing `…` marks truncation) so printing
        a big table — a [50257, 768] embedding — stays one short line.
        """
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
        """Matrix multiply via `@`: [M, K] @ [K, N] -> [M, N] in ikj loop order.

        The inner loop walks other[k, *] and result[i, *] contiguously, which is
        cache-friendly and vectorizes: the W columns result[i, j..j+W] update in
        one SIMD fused step, with a scalar tail. Order-preserving: each cell
        accumulates over k in the same ascending order as the scalar loop (SIMD
        only groups the independent j columns), so it is bit-identical to a scalar
        ikj reference. Threading partitions the independent output rows, which
        keeps it bit-identical to serial and stable run to run.

        Args:
            other: Right operand, shape [K, N].

        Returns:
            The product [M, N]. Allocates the result.

        Raises:
            Error: On a shape mismatch (self.cols != other.rows).
        """
        if self.cols != other.rows:
            raise Error("matmul shape mismatch")
        comptime W = simd_width_of[DType.float64]()
        var m = self.rows
        var kdim = self.cols
        var n = other.cols
        var result = zeros_2d(m, n)
        var pres = result.data.unsafe_ptr()
        var poth = other.data.unsafe_ptr()

        @parameter
        def row_block(i0: Int, i1: Int):
            for i in range(i0, i1):
                var ibase = i * n
                for k in range(kdim):
                    # self[i, k] is constant across j; hoist and broadcast it.
                    var a_ik = self[i, k]
                    var a_vec = SIMD[DType.float64, W](a_ik)
                    var obase = k * n
                    var j = 0
                    while j + W <= n:
                        var acc = pres.load[width=W](
                            ibase + j
                        ) + a_vec * poth.load[width=W](obase + j)
                        pres.store(ibase + j, acc)
                        j += W
                    while j < n:
                        # Write the ragged tail through the raw pointer too, not
                        # the `result` capture: this runs inside the parallelize
                        # worker, and the project's rule is parallel output goes
                        # through unsafe_ptr, never a mut capture.
                        pres[ibase + j] += a_ik * poth[obase + j]
                        j += 1

        if m * kdim * n < _MATMUL_THREAD_MIN_WORK or m < 2:
            row_block(0, m)
            return result^

        var nblocks = 32 if m >= 32 else m
        var block = (m + nblocks - 1) // nblocks

        @parameter
        def worker(t: Int):
            var i0 = t * block
            var i1 = i0 + block
            if i1 > m:
                i1 = m
            row_block(i0, i1)

        parallelize[worker](nblocks)
        return result^


def zeros_2d(rows: Int, cols: Int) -> Tensor2D:
    """Return a rows x cols tensor of zeros, allocated in one shot."""
    var data = List[Float64](length=rows * cols, fill=0.0)
    return Tensor2D(rows, cols, data^)


def full_2d(rows: Int, cols: Int, value: Float64) -> Tensor2D:
    """Return a rows x cols tensor filled with `value`."""
    var t = zeros_2d(rows, cols)
    t.fill(value)
    return t^


def ones_2d(rows: Int, cols: Int) -> Tensor2D:
    """Return a rows x cols tensor of ones."""
    return full_2d(rows, cols, 1.0)


def from_rows(values: List[List[Float64]]) raises -> Tensor2D:
    """Build a tensor from a list of equal-length rows.

    Raises:
        Error: On zero rows or ragged input, so a mis-shaped literal fails at
            construction.
    """
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
