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


@fieldwise_init
struct Tensor2D(Copyable, Movable):
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

    def fill(mut self, value: Float64):
        # Overwrite every element with `value`. Mutates in place.
        for i in range(self.size()):
            self.data[i] = value


def zeros_2d(rows: Int, cols: Int) -> Tensor2D:
    # A rows x cols tensor of zeros. Allocates the flat buffer.
    var data = List[Float64]()
    for _ in range(rows * cols):
        data.append(0.0)
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
