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

from llm.nn.linear import Linear
from llm.tensor.ops import (
    add,
    concat_cols,
    matmul,
    scale,
    slice_cols,
    softmax_rows,
    transpose,
)
from llm.tensor.tensor2d import Tensor2D
from llm.utils.random import Rng


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


@fieldwise_init
struct MultiHeadAttention(Copyable, Movable):
    # GPT-2's multi-head self-attention with the FUSED QKV projection.
    #
    # One Linear(C -> 3C) (GPT-2's `c_attn`) produces Q, K, V for every head in a
    # single matmul; one Linear(C -> C) (`c_proj`) mixes the concatenated head
    # outputs. This is the checkpoint layout weight loading will fill, and its
    # parameter cost (3C^2 + 3C for qkv, C^2 + C for proj) is exactly the
    # attention arithmetic the architecture's parameter count already commits to.
    #
    # Head layout is CONTIGUOUS, matching GPT-2: the [T, 3C] projection splits
    # into three [T, C] thirds (Q, K, V), and each third splits into H contiguous
    # [T, D] head slices with D = C / H — head h owns columns [h*D, (h+1)*D), not
    # an interleaved stride. The [B, H, T, D] shape you see in framework code is
    # loop discipline here: B is the block's batch loop (a later part), H is the
    # loop below, and the tile each head works on is [T, D].
    var qkv: Linear  # C -> 3C, fused c_attn
    var proj: Linear  # C -> C, c_proj
    var n_heads: Int

    @staticmethod
    def init_random(
        mut rng: Rng, d_model: Int, n_heads: Int
    ) raises -> MultiHeadAttention:
        # An MHA with both Linears drawn from GPT-2's normal(0, 0.02) scheme and
        # zero biases (qkv weights are drawn first, then proj). Mutates rng
        # (advances its state); allocates both layers; deterministic given the
        # generator's state. Raises unless n_heads > 0 and d_model is a positive
        # multiple of n_heads — otherwise the head width D = C / H is undefined
        # or degenerate.
        if n_heads <= 0:
            raise Error(
                "MultiHeadAttention.init_random: n_heads must be positive, got "
                + String(n_heads)
            )
        if d_model <= 0:
            raise Error(
                "MultiHeadAttention.init_random: d_model must be positive, got "
                + String(d_model)
            )
        if d_model % n_heads != 0:
            raise Error(
                "MultiHeadAttention.init_random: d_model ("
                + String(d_model)
                + ") must be divisible by n_heads ("
                + String(n_heads)
                + ")"
            )
        var qkv = Linear.init_random(rng, d_model, 3 * d_model)  # C -> 3C
        var proj = Linear.init_random(rng, d_model, d_model)  # C -> C
        return MultiHeadAttention(qkv^, proj^, n_heads)

    def forward(self, x: Tensor2D, mask: Tensor2D) raises -> Tensor2D:
        # Self-attention over one sequence: [T, C] + mask [T, T] -> [T, C]. Q, K,
        # V all derive from x. Reads self only; allocates the projections, the
        # per-head slices, and the result; raises on a feature-count mismatch
        # (via the qkv Linear) or an indivisible width. The caller's mask is
        # applied unchanged to every head.
        var c = self.qkv.weight.value.cols  # in_features of c_attn = C
        if c % self.n_heads != 0:
            raise Error(
                "MultiHeadAttention.forward: C ("
                + String(c)
                + ") not divisible by n_heads ("
                + String(self.n_heads)
                + ")"
            )
        var d_head = c // self.n_heads  # per-head width D = C / H

        var qkv = self.qkv.forward(x)  # [T, C] -> [T, 3C]
        # Split the fused projection into the Q, K, V thirds (each [T, C]).
        var q_all = slice_cols(qkv, 0, c)  # [T, C]
        var k_all = slice_cols(qkv, c, 2 * c)  # [T, C]
        var v_all = slice_cols(qkv, 2 * c, 3 * c)  # [T, C]

        # Run the core once per head over its contiguous [T, D] slice, then
        # concatenate the head outputs back to [T, C].
        var head_outputs = List[Tensor2D]()
        for h in range(self.n_heads):
            var lo = h * d_head
            var hi = lo + d_head
            var q_h = slice_cols(q_all, lo, hi)  # [T, D]
            var k_h = slice_cols(k_all, lo, hi)  # [T, D]
            var v_h = slice_cols(v_all, lo, hi)  # [T, D]
            var result = scaled_dot_product_attention(q_h, k_h, v_h, mask)
            # Copy the field out: a single struct field cannot be transferred
            # (`^`) from a live value ("destroyed out of the middle of a value").
            head_outputs.append(result.output.copy())  # [T, D]
        var concatenated = concat_cols(head_outputs)  # [T, C]

        return self.proj.forward(concatenated)  # [T, C] -> [T, C]
