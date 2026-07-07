# A small educational 3-D tensor — the shape of batched sequences [B, T, C].
#
# Same idea as Tensor2D, one dimension higher, over flat nested row-major
# storage. For shape [d0, d1, d2] the strides are (d1*d2, d2, 1), so
#
#     offset(i, j, k) = (i * d1 + j) * d2 + k
#
# Pinning that offset arithmetic with a test is how you catch a transposed
# stride before it becomes an attention bug. Access mirrors Tensor2D: one
# ref-returning `[i, j, k]` on the hot path — a reference into the flat buffer
# (its origin), so the same subscript reads, writes, and does `+=` with no
# separate setter — and checked `.at(i, j, k)` (raising) for tests.

from std.collections import List


@fieldwise_init
struct Tensor3D(Copyable, Movable):
    var d0: Int
    var d1: Int
    var d2: Int
    var data: List[Float64]  # flat nested row-major [d0, d1, d2]

    def size(self) -> Int:
        return self.d0 * self.d1 * self.d2

    def offset(self, i: Int, j: Int, k: Int) -> Int:
        # Nested row-major flat index: (i*d1 + j)*d2 + k. No bounds check.
        return (i * self.d1 + j) * self.d2 + k

    def __getitem__(ref self, i: Int, j: Int, k: Int) -> ref[self.data] Float64:
        # Unchecked ref access (hot path): one method serves read, write, and +=.
        # The returned reference borrows self.data as its origin, so assigning or
        # accumulating through the subscript mutates the buffer directly — no
        # separate setter. (Origin must name the field, not self.)
        return self.data[self.offset(i, j, k)]

    def at(self, i: Int, j: Int, k: Int) raises -> Float64:
        # Bounds-checked read. Raises if any index is outside its dimension.
        if (
            i < 0
            or i >= self.d0
            or j < 0
            or j >= self.d1
            or k < 0
            or k >= self.d2
        ):
            raise Error("Tensor3D index out of range")
        return self.data[self.offset(i, j, k)]


def zeros_3d(d0: Int, d1: Int, d2: Int) -> Tensor3D:
    # A d0 x d1 x d2 tensor of zeros. Allocates the flat buffer.
    var data = List[Float64]()
    for _ in range(d0 * d1 * d2):
        data.append(0.0)
    return Tensor3D(d0, d1, d2, data^)
