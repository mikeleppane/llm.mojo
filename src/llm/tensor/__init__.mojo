"""Tensor types and operations: Tensor2D, Tensor3D, and the ops built on them."""

from .tensor2d import Tensor2D, zeros_2d, full_2d, ones_2d, from_rows
from .tensor3d import Tensor3D, zeros_3d
from .ops import (
    add,
    scale,
    transpose,
    slice_cols,
    slice_rows,
    concat_cols,
    matmul,
    matmul_ikj,
    matmul_transpose_a,
    matmul_transpose_b,
    matvec,
    softmax_row,
    softmax_rows,
    softmax_rows_backward,
    softmax_row_temperature,
    logsumexp,
    cross_entropy_one,
    cross_entropy_grad,
    cross_entropy_rows,
    cross_entropy_rows_backward,
    argmax,
)
from .init_weights import xavier_2d
