"""A small educational 3-D tensor — the shape of batched sequences [B, T, C].

Same idea as Tensor2D, one dimension higher, over flat nested row-major storage.
For shape [d0, d1, d2] the strides are (d1*d2, d2, 1), so
offset(i, j, k) = (i * d1 + j) * d2 + k.
"""

from std.collections import List


@fieldwise_init
struct Tensor3D(Copyable, Movable, Writable):
    """Batched activations [d0, d1, d2] over flat nested row-major Float64 storage.
    """

    var d0: Int
    var d1: Int
    var d2: Int
    var data: List[Float64]  # flat nested row-major [d0, d1, d2]

    def size(self) -> Int:
        """Total element count d0 * d1 * d2."""
        return self.d0 * self.d1 * self.d2

    def offset(self, i: Int, j: Int, k: Int) -> Int:
        """Nested row-major flat index (i*d1 + j)*d2 + k. No bounds check."""
        return (i * self.d1 + j) * self.d2 + k

    def __getitem__(ref self, i: Int, j: Int, k: Int) -> ref[self.data] Float64:
        """Unchecked ref access (hot path): one method serves read, write, and +=.

        The returned reference borrows self.data as its origin, so assigning or
        accumulating through the subscript mutates the buffer directly.
        """
        return self.data[self.offset(i, j, k)]

    def write_to(self, mut writer: Some[Writer]):
        """Write a shape header plus a capped preview, e.g. `Tensor3D[2, 3, 4](...)`.

        The preview stops at 8 values (trailing `…` marks truncation) so printing
        a big batched activation stays one short line.
        """
        writer.write("Tensor3D[", self.d0, ", ", self.d1, ", ", self.d2, "](")
        var n = self.size()
        var limit = n if n < 8 else 8
        for i in range(limit):
            if i > 0:
                writer.write(", ")
            writer.write(self.data[i])
        if n > 8:
            writer.write(", …")
        writer.write(")")

    def at(self, i: Int, j: Int, k: Int) raises -> Float64:
        """Bounds-checked read.

        Raises:
            Error: If any index is outside its dimension.
        """
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
    """Return a d0 x d1 x d2 tensor of zeros, allocated in one shot."""
    var data = List[Float64](length=d0 * d1 * d2, fill=0.0)
    return Tensor3D(d0, d1, d2, data^)
