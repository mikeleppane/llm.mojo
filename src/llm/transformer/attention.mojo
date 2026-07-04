# Scaled-dot-product attention — the one core that serves both self- and
# cross-attention.
#
# Because q is a separate argument from k/v, each with its own length, the same
# function *is* the cross-attention core: self-attention passes q, k, v all
# derived from one sequence; cross-attention passes q from the decoder and k/v
# from the encoder. Nothing in here knows or cares which — the shape [T_q, D] vs
# [T_k, D] is the whole story. This is why the tests exercise T_q != T_k now,
# before any cross-attention layer exists to need it.
#
# The pipeline reuses the tested tensor ops (matmul, transpose, scale, add,
# softmax_rows) rather than open-coding a second matmul or softmax. The order is
# load-bearing and pinned in the docstring below.

from std.math import sqrt

from llm.tensor.ops import add, matmul, scale, softmax_rows, transpose
from llm.tensor.tensor2d import Tensor2D


struct AttentionResult(Copyable, Movable):
    # The core returns the weights alongside the output on purpose: the weights
    # are how tests *prove* causality (every entry above the diagonal is 0 after
    # softmax) and how the chapter visualizes what each query attends to.
    # Recomputing them in a second code path would test a copy, not the code.
    var output: Tensor2D  # [T_q, D_v]
    var weights: Tensor2D  # [T_q, T_k] — post-softmax, each row sums to 1

    def __init__(out self, var output: Tensor2D, var weights: Tensor2D):
        self.output = output^
        self.weights = weights^


def scaled_dot_product_attention(
    q: Tensor2D, k: Tensor2D, v: Tensor2D, mask: Tensor2D
) raises -> AttentionResult:
    # Attention(q, k, v) with an additive mask. Shapes:
    #   q    [T_q, D]      queries
    #   k    [T_k, D]      keys   (same feature width D as q)
    #   v    [T_k, D_v]    values (same count T_k as k, own width D_v)
    #   mask [T_q, T_k]    additive: 0 attends, large-negative blocks
    # Returns output [T_q, D_v] and weights [T_q, T_k].
    #
    # Pinned order (a wrong order is a classic attention bug):
    #   1. scores  = q @ k^T                     [T_q, T_k]
    #   2. scale   = scores * (1 / sqrt(D))      D = q.cols = d_head, NOT d_model
    #   3. add     = scale + mask                additive mask, AFTER the scale
    #   4. weights = softmax_rows(add)           row-wise, stable
    #   5. output  = weights @ v                 [T_q, D_v]
    # Scaling before the mask keeps a blocked score decisively negative; the mask
    # is designed as an additive term on already-scaled scores. Reads its args
    # only; allocates the intermediates and result; raises on any shape mismatch.
    var t_q = q.rows
    var d = q.cols
    var t_k = k.rows
    if d == 0:
        raise Error("scaled_dot_product_attention: q has zero feature width D")
    if k.cols != d:
        raise Error(
            "scaled_dot_product_attention: q/k feature-width mismatch, q has D="
            + String(d)
            + " but k has "
            + String(k.cols)
        )
    if v.rows != t_k:
        raise Error(
            "scaled_dot_product_attention: v/k length mismatch, k has T_k="
            + String(t_k)
            + " but v has "
            + String(v.rows)
        )
    if mask.rows != t_q or mask.cols != t_k:
        raise Error(
            "scaled_dot_product_attention: mask must be [T_q, T_k] = ["
            + String(t_q)
            + ", "
            + String(t_k)
            + "], got ["
            + String(mask.rows)
            + ", "
            + String(mask.cols)
            + "]"
        )

    var scores = matmul(q, transpose(k))  # [T_q, T_k]
    var scaled = scale(scores, 1.0 / sqrt(Float64(d)))  # 1/sqrt(d_head)
    var biased = add(scaled, mask)  # additive mask, after the scale
    var weights = softmax_rows(biased)  # [T_q, T_k], rows sum to 1
    var output = matmul(weights, v)  # [T_q, D_v]
    return AttentionResult(output^, weights^)
