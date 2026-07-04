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

from llm.nn.dropout import DropoutResult, dropout_backward, dropout_cached
from llm.nn.linear import Linear, LinearCache
from llm.tensor.ops import (
    add,
    concat_cols,
    matmul,
    scale,
    slice_cols,
    softmax_rows,
    softmax_rows_backward,
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
struct AttentionCache(Copyable, Movable):
    # What the core's backward needs: q, k, v and the post-softmax weights. The
    # mask is not stored — it is constant additive data with no gradient, and its
    # effect already lives in `weights` (blocked entries are ~0, so the softmax
    # backward kills their gradient). Valid only for the forward call that
    # produced it.
    var q: Tensor2D  # [T_q, D]
    var k: Tensor2D  # [T_k, D]
    var v: Tensor2D  # [T_k, D_v]
    var weights: Tensor2D  # [T_q, T_k], post-softmax


@fieldwise_init
struct AttentionForward(Copyable, Movable):
    # scaled_dot_product_attention_cached's output plus the cache its backward
    # consumes.
    var output: Tensor2D  # [T_q, D_v]
    var cache: AttentionCache


@fieldwise_init
struct AttentionGrads(Copyable, Movable):
    # The core backward's three input gradients, one per differentiable input.
    var d_q: Tensor2D  # [T_q, D]
    var d_k: Tensor2D  # [T_k, D]
    var d_v: Tensor2D  # [T_k, D_v]


def scaled_dot_product_attention_cached(
    q: Tensor2D, k: Tensor2D, v: Tensor2D, mask: Tensor2D
) raises -> AttentionForward:
    # Same forward as scaled_dot_product_attention, additionally returning the
    # cache its backward needs (q, k, v, weights). Reads its args; allocates the
    # output and the cache; raises on the same shape mismatches the forward does.
    var result = scaled_dot_product_attention(q, k, v, mask)
    var cache = AttentionCache(
        q.copy(), k.copy(), v.copy(), result.weights.copy()
    )
    return AttentionForward(result.output.copy(), cache^)


def scaled_dot_product_attention_backward(
    cache: AttentionCache, d_out: Tensor2D
) raises -> AttentionGrads:
    # Reverse of the pinned forward order. Let s = 1/sqrt(D), D = q.cols = d_head
    # (the SAME scale the forward used, applied once). Forward: S = qk^T, scaled
    # by s, + mask, softmax -> W, output = Wv. So with dO = d_out [T_q, D_v]:
    #   dV = W^T @ dO                          [T_k, T_q] @ [T_q, D_v] -> [T_k, D_v]
    #   dW = dO @ V^T                          [T_q, D_v] @ [D_v, T_k] -> [T_q, T_k]
    #   dScaled = softmax_rows_backward(W, dW)                          -> [T_q, T_k]
    #   dScores = dScaled                      (mask is constant; add passes dW through)
    #   dQ = (dScores @ K) * s                 [T_q, T_k] @ [T_k, D]   -> [T_q, D]
    #   dK = (dScores^T @ Q) * s               [T_k, T_q] @ [T_q, D]   -> [T_k, D]
    # The scale s folds into dQ/dK once (dScores = s·(qk^T-backward)); it does NOT
    # apply to dV, which comes off the value matmul, not the scaled scores. The
    # mask leaks no gradient: softmax_rows_backward multiplies by W, and a blocked
    # entry's W is ~0. Reads the cache and d_out; allocates the three gradients;
    # raises on a shape mismatch or a zero feature width.
    var t_q = cache.q.rows
    var d = cache.q.cols
    var t_k = cache.k.rows
    if d == 0:
        raise Error(
            "scaled_dot_product_attention_backward: q has zero feature width D"
        )
    if cache.weights.rows != t_q or cache.weights.cols != t_k:
        raise Error(
            "scaled_dot_product_attention_backward: weights must be [T_q, T_k]"
        )
    if d_out.rows != t_q or d_out.cols != cache.v.cols:
        raise Error(
            "scaled_dot_product_attention_backward: d_out must be [T_q, D_v]"
        )
    if cache.v.rows != t_k:
        raise Error(
            "scaled_dot_product_attention_backward: v/k length mismatch"
        )
    var s = 1.0 / sqrt(Float64(d))
    var d_v = matmul(transpose(cache.weights), d_out)  # [T_k, D_v]
    var d_w = matmul(d_out, transpose(cache.v))  # [T_q, T_k]
    var d_scores = softmax_rows_backward(cache.weights, d_w)  # [T_q, T_k]
    var d_q = scale(matmul(d_scores, cache.k), s)  # [T_q, D]
    var d_k = scale(matmul(transpose(d_scores), cache.q), s)  # [T_k, D]
    return AttentionGrads(d_q^, d_k^, d_v^)


# ===================== Attention-weight dropout (train path) =====================
#
# GPT-2 drops the post-softmax attention weights before they weight the values:
# `output = dropout(W) @ v`. This is an ADDITIVE path over the frozen core above —
# the existing forward/backward are untouched, so every earlier caller keeps
# compiling against them. With training = False (or p = 0) `dropout_cached` returns
# an all-ones mask, unit scale, and consumes NO rng, so this path degenerates to
# the proven one exactly — a test pins output and gradient equality.


@fieldwise_init
struct AttentionTrainCache(Copyable, Movable):
    # What the train-path backward needs: the core cache (q, k, v, the PRE-dropout
    # post-softmax weights) plus the dropout mask actually applied to those weights
    # and the scale the forward used. `weights` is the pre-dropout W — what the
    # softmax produced, and what softmax_rows_backward must be fed; `drop_mask` and
    # `inv_keep` reconstruct the DROPPED weights the value-matmul saw. Valid only
    # for the forward call that produced it.
    var q: Tensor2D  # [T_q, D]
    var k: Tensor2D  # [T_k, D]
    var v: Tensor2D  # [T_k, D_v]
    var weights: Tensor2D  # [T_q, T_k], PRE-dropout post-softmax
    var drop_mask: Tensor2D  # [T_q, T_k], entries in {0, 1}
    var inv_keep: Float64  # applied dropout scale: 1/(1-p) train, 1.0 eval/p==0


@fieldwise_init
struct AttentionTrainForward(Copyable, Movable):
    # scaled_dot_product_attention_train's output plus the cache its backward
    # consumes.
    var output: Tensor2D  # [T_q, D_v]
    var cache: AttentionTrainCache


def scaled_dot_product_attention_train(
    q: Tensor2D,
    k: Tensor2D,
    v: Tensor2D,
    mask: Tensor2D,
    p: Float64,
    training: Bool,
    mut rng: Rng,
) raises -> AttentionTrainForward:
    # The core forward with GPT-2's attention-weight dropout spliced into the
    # pinned order:
    #   scores -> scale -> + mask -> softmax -> W -> dropout_cached(W) -> dropped_W
    #   output = dropped_W @ v
    # Shapes exactly as scaled_dot_product_attention (q [T_q, D], k [T_k, D],
    # v [T_k, D_v], mask [T_q, T_k] -> output [T_q, D_v]). Reads its args; allocates
    # the weights, the mask, and the output; mutates rng only in the training/p>0
    # branch (one uniform per weight entry, row-major); raises on the same shape
    # mismatches the core does or an out-of-range p.
    #
    # The base call recomputes the un-dropped W @ v and discards it — a deliberate
    # simplicity-over-speed choice that keeps the pinned order in ONE place (the
    # core) rather than re-spelling scores->softmax here; a later performance pass
    # can fold the throwaway matmul away. Only `base.weights` is used.
    var base = scaled_dot_product_attention(q, k, v, mask)  # base.weights = W
    var drop = dropout_cached(base.weights, p, training, rng)  # dropped_W, mask
    var output = matmul(drop.output, v)  # dropped_W @ v -> [T_q, D_v]
    var cache = AttentionTrainCache(
        q.copy(),
        k.copy(),
        v.copy(),
        base.weights.copy(),  # PRE-dropout W, for softmax_rows_backward
        drop.mask.copy(),
        drop.inv_keep,
    )
    return AttentionTrainForward(output^, cache^)


def scaled_dot_product_attention_train_backward(
    cache: AttentionTrainCache, d_out: Tensor2D
) raises -> AttentionGrads:
    # Reverse of the train forward — composition of the frozen core backward with
    # the dropout backward, no new math. Let dropped_W = W ⊙ mask · inv_keep be the
    # weights the value-matmul actually used. With dO = d_out [T_q, D_v]:
    #   dV        = dropped_W^T @ dO              (the value matmul saw the DROPPED W)
    #   d_droppedW = dO @ V^T
    #   dW        = dropout_backward(mask, inv_keep, d_droppedW)   (undo the drop)
    #   dS        = softmax_rows_backward(W, dW)   (on the PRE-dropout W — what softmax made)
    #   dQ = (dS @ K)·s,  dK = (dS^T @ Q)·s,  s = 1/sqrt(D)   (exactly as the core)
    # The two weight tensors are distinct on purpose: dV is fed the DROPPED weights
    # (that is what multiplied v in the forward), while softmax_rows_backward is fed
    # the PRE-dropout W (that is what the softmax produced). Swapping them is the
    # classic attention-dropout backward bug the finite-diff catches. Reads the
    # cache and d_out; allocates the three gradients; raises on a shape mismatch or
    # a zero feature width.
    var d = cache.q.cols
    if d == 0:
        raise Error(
            "scaled_dot_product_attention_train_backward: q has zero feature"
            " width D"
        )
    # dropout_backward applies the diagonal (mask · inv_keep) map, which is its own
    # transpose — so feeding it the pre-dropout weights reproduces exactly the
    # dropped weights the forward's value-matmul consumed.
    var dropped_w = dropout_backward(
        cache.drop_mask, cache.inv_keep, cache.weights
    )  # [T_q, T_k]
    var d_v = matmul(transpose(dropped_w), d_out)  # [T_k, D_v]
    var d_dropped_w = matmul(d_out, transpose(cache.v))  # [T_q, T_k]
    var d_w = dropout_backward(
        cache.drop_mask, cache.inv_keep, d_dropped_w
    )  # [T_q, T_k]
    var d_scores = softmax_rows_backward(cache.weights, d_w)  # PRE-dropout W
    var s = 1.0 / sqrt(Float64(d))
    var d_q = scale(matmul(d_scores, cache.k), s)  # [T_q, D]
    var d_k = scale(matmul(transpose(d_scores), cache.q), s)  # [T_k, D]
    return AttentionGrads(d_q^, d_k^, d_v^)


@fieldwise_init
struct MHACache(Copyable, Movable):
    # What MultiHeadAttention.backward needs, mirroring the forward plumbing: the
    # fused qkv Linear's cache (its input x), one AttentionCache per head, and the
    # output proj Linear's cache (its input, the concatenated head outputs). Valid
    # only for the forward call that produced it.
    var qkv_cache: LinearCache  # holds x                    [T, C]
    var head_caches: List[AttentionCache]  # per head: q,k,v,weights
    var proj_cache: LinearCache  # holds concat(head outputs) [T, C]


@fieldwise_init
struct MHAForward(Copyable, Movable):
    # forward_cached's output plus the cache its backward consumes.
    var output: Tensor2D  # [T, C]
    var cache: MHACache


@fieldwise_init
struct MHATrainCache(Copyable, Movable):
    # What MultiHeadAttention.backward_train needs — the train-path twin of
    # MHACache: the fused qkv Linear's cache (its input x), one AttentionTrainCache
    # per head (each carrying that head's attention-weight dropout mask), and the
    # output proj Linear's cache. Valid only for the forward call that produced it.
    var qkv_cache: LinearCache  # holds x                          [T, C]
    var head_caches: List[
        AttentionTrainCache
    ]  # per head: q,k,v,W,mask,inv_keep
    var proj_cache: LinearCache  # holds concat(head outputs)      [T, C]


@fieldwise_init
struct MHATrainForward(Copyable, Movable):
    # forward_cached_train's output plus the cache its backward consumes.
    var output: Tensor2D  # [T, C]
    var cache: MHATrainCache


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
        # @fieldwise_init lets a caller build this struct directly, bypassing
        # init_random's guard, so re-check n_heads here rather than trap on a
        # modulo-by-zero below.
        if self.n_heads <= 0:
            raise Error(
                "MultiHeadAttention.forward: n_heads must be positive, got "
                + String(self.n_heads)
            )
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

    def forward_cached(self, x: Tensor2D, mask: Tensor2D) raises -> MHAForward:
        # Self-attention over one sequence with the cache backward needs: [T, C] +
        # mask [T, T] -> MHAForward. Same computation as forward, capturing the
        # fused-qkv cache, one AttentionCache per head, and the proj cache. Reads
        # self; allocates the projections, per-head slices, caches, and result;
        # raises on a feature-count mismatch or an indivisible width. The cache is
        # valid only for this call.
        if self.n_heads <= 0:
            raise Error(
                "MultiHeadAttention.forward_cached: n_heads must be positive,"
                " got "
                + String(self.n_heads)
            )
        var c = self.qkv.weight.value.cols  # in_features of c_attn = C
        if c % self.n_heads != 0:
            raise Error(
                "MultiHeadAttention.forward_cached: C ("
                + String(c)
                + ") not divisible by n_heads ("
                + String(self.n_heads)
                + ")"
            )
        var d_head = c // self.n_heads  # per-head width D = C / H

        var qkv_fwd = self.qkv.forward_cached(x)  # output [T, 3C], cache = x
        var q_all = slice_cols(qkv_fwd.output, 0, c)  # [T, C]
        var k_all = slice_cols(qkv_fwd.output, c, 2 * c)  # [T, C]
        var v_all = slice_cols(qkv_fwd.output, 2 * c, 3 * c)  # [T, C]

        var head_outputs = List[Tensor2D]()
        var head_caches = List[AttentionCache]()
        for h in range(self.n_heads):
            var lo = h * d_head
            var hi = lo + d_head
            var q_h = slice_cols(q_all, lo, hi)  # [T, D]
            var k_h = slice_cols(k_all, lo, hi)  # [T, D]
            var v_h = slice_cols(v_all, lo, hi)  # [T, D]
            var head = scaled_dot_product_attention_cached(q_h, k_h, v_h, mask)
            head_outputs.append(head.output.copy())  # [T, D]
            head_caches.append(head.cache.copy())
        var concatenated = concat_cols(head_outputs)  # [T, C]
        var proj_fwd = self.proj.forward_cached(concatenated)  # [T, C]

        var cache = MHACache(
            qkv_fwd.cache.copy(), head_caches^, proj_fwd.cache.copy()
        )
        return MHAForward(proj_fwd.output.copy(), cache^)

    def backward(mut self, cache: MHACache, d_out: Tensor2D) raises -> Tensor2D:
        # Reverse the forward plumbing exactly, right to left. [T, C] d_out ->
        # [T, C] d_x, accumulating the qkv and proj parameter grads (+=).
        #   1. d_concat = proj.backward(proj_cache, d_out)   [T, C]
        #   2. split d_concat into H contiguous [T, D] head slices; per head run
        #      the core backward to get d_q_h, d_k_h, d_v_h.
        #   3. concat the per-head d_q's (and d_k's, d_v's) back to [T, C] — the
        #      exact inverse of the forward's head split.
        #   4. d_qkv = [d_q_all | d_k_all | d_v_all]         [T, 3C] — the inverse
        #      of the forward's Q/K/V third-split.
        #   5. d_x = qkv.backward(qkv_cache, d_qkv)          [T, C]
        # slice_cols and concat_cols are exact inverses (pinned in Part X), so the
        # column bookkeeping round-trips. Mutates self.proj and self.qkv parameter
        # grads; allocates and returns d_x; raises on a shape/config mismatch.
        if self.n_heads <= 0:
            raise Error(
                "MultiHeadAttention.backward: n_heads must be positive, got "
                + String(self.n_heads)
            )
        var c = self.qkv.weight.value.cols
        if c % self.n_heads != 0:
            raise Error(
                "MultiHeadAttention.backward: C ("
                + String(c)
                + ") not divisible by n_heads ("
                + String(self.n_heads)
                + ")"
            )
        if len(cache.head_caches) != self.n_heads:
            raise Error(
                "MultiHeadAttention.backward: cache has "
                + String(len(cache.head_caches))
                + " head caches but n_heads is "
                + String(self.n_heads)
            )
        var d_head = c // self.n_heads

        var d_concat = self.proj.backward(cache.proj_cache, d_out)  # [T, C]

        var dq_heads = List[Tensor2D]()
        var dk_heads = List[Tensor2D]()
        var dv_heads = List[Tensor2D]()
        for h in range(self.n_heads):
            var lo = h * d_head
            var hi = lo + d_head
            var d_head_out = slice_cols(d_concat, lo, hi)  # [T, D]
            var grads = scaled_dot_product_attention_backward(
                cache.head_caches[h], d_head_out
            )
            dq_heads.append(grads.d_q.copy())  # [T, D]
            dk_heads.append(grads.d_k.copy())
            dv_heads.append(grads.d_v.copy())
        var d_q_all = concat_cols(dq_heads)  # [T, C]
        var d_k_all = concat_cols(dk_heads)  # [T, C]
        var d_v_all = concat_cols(dv_heads)  # [T, C]

        # Reassemble the fused-qkv gradient in the same Q|K|V column order the
        # forward's projection produced.
        var qkv_parts = List[Tensor2D]()
        qkv_parts.append(d_q_all^)
        qkv_parts.append(d_k_all^)
        qkv_parts.append(d_v_all^)
        var d_qkv = concat_cols(qkv_parts)  # [T, 3C]

        return self.qkv.backward(cache.qkv_cache, d_qkv)  # [T, C]

    def forward_cached_train(
        self,
        x: Tensor2D,
        mask: Tensor2D,
        p: Float64,
        training: Bool,
        mut rng: Rng,
    ) raises -> MHATrainForward:
        # The train-path twin of forward_cached: identical plumbing (fused qkv,
        # per-head split, concat, proj), but each head runs the attention-weight
        # dropout core. [T, C] + mask [T, T] -> MHATrainForward. rng is threaded
        # through the heads in order, so each head draws its own [T, T] weight mask;
        # with training = False (or p = 0) no head draws and the result equals
        # forward_cached exactly. Reads self; allocates the projections, per-head
        # slices, caches, and result; mutates rng only in the training/p>0 branch;
        # raises on a feature-count mismatch, an indivisible width, or an
        # out-of-range p. The cache is valid only for this call.
        if self.n_heads <= 0:
            raise Error(
                "MultiHeadAttention.forward_cached_train: n_heads must be"
                " positive, got "
                + String(self.n_heads)
            )
        var c = self.qkv.weight.value.cols  # in_features of c_attn = C
        if c % self.n_heads != 0:
            raise Error(
                "MultiHeadAttention.forward_cached_train: C ("
                + String(c)
                + ") not divisible by n_heads ("
                + String(self.n_heads)
                + ")"
            )
        var d_head = c // self.n_heads  # per-head width D = C / H

        var qkv_fwd = self.qkv.forward_cached(x)  # output [T, 3C], cache = x
        var q_all = slice_cols(qkv_fwd.output, 0, c)  # [T, C]
        var k_all = slice_cols(qkv_fwd.output, c, 2 * c)  # [T, C]
        var v_all = slice_cols(qkv_fwd.output, 2 * c, 3 * c)  # [T, C]

        var head_outputs = List[Tensor2D]()
        var head_caches = List[AttentionTrainCache]()
        for h in range(self.n_heads):
            var lo = h * d_head
            var hi = lo + d_head
            var q_h = slice_cols(q_all, lo, hi)  # [T, D]
            var k_h = slice_cols(k_all, lo, hi)  # [T, D]
            var v_h = slice_cols(v_all, lo, hi)  # [T, D]
            var head = scaled_dot_product_attention_train(
                q_h, k_h, v_h, mask, p, training, rng
            )
            head_outputs.append(head.output.copy())  # [T, D]
            head_caches.append(head.cache.copy())
        var concatenated = concat_cols(head_outputs)  # [T, C]
        var proj_fwd = self.proj.forward_cached(concatenated)  # [T, C]

        var cache = MHATrainCache(
            qkv_fwd.cache.copy(), head_caches^, proj_fwd.cache.copy()
        )
        return MHATrainForward(proj_fwd.output.copy(), cache^)

    def backward_train(
        mut self, cache: MHATrainCache, d_out: Tensor2D
    ) raises -> Tensor2D:
        # The train-path twin of backward: reverse the same plumbing right to left,
        # but each head runs the attention-weight-dropout core backward. [T, C]
        # d_out -> [T, C] d_x, accumulating qkv and proj parameter grads (+=). With
        # training = False (or p = 0) each head cache carries an all-ones mask and
        # unit scale, so this returns exactly what backward returns — a test pins
        # the equality. Mutates self.proj and self.qkv parameter grads; allocates
        # and returns d_x; raises on a shape/config mismatch.
        if self.n_heads <= 0:
            raise Error(
                "MultiHeadAttention.backward_train: n_heads must be positive,"
                " got "
                + String(self.n_heads)
            )
        var c = self.qkv.weight.value.cols
        if c % self.n_heads != 0:
            raise Error(
                "MultiHeadAttention.backward_train: C ("
                + String(c)
                + ") not divisible by n_heads ("
                + String(self.n_heads)
                + ")"
            )
        if len(cache.head_caches) != self.n_heads:
            raise Error(
                "MultiHeadAttention.backward_train: cache has "
                + String(len(cache.head_caches))
                + " head caches but n_heads is "
                + String(self.n_heads)
            )
        var d_head = c // self.n_heads

        var d_concat = self.proj.backward(cache.proj_cache, d_out)  # [T, C]

        var dq_heads = List[Tensor2D]()
        var dk_heads = List[Tensor2D]()
        var dv_heads = List[Tensor2D]()
        for h in range(self.n_heads):
            var lo = h * d_head
            var hi = lo + d_head
            var d_head_out = slice_cols(d_concat, lo, hi)  # [T, D]
            var grads = scaled_dot_product_attention_train_backward(
                cache.head_caches[h], d_head_out
            )
            dq_heads.append(grads.d_q.copy())  # [T, D]
            dk_heads.append(grads.d_k.copy())
            dv_heads.append(grads.d_v.copy())
        var d_q_all = concat_cols(dq_heads)  # [T, C]
        var d_k_all = concat_cols(dk_heads)  # [T, C]
        var d_v_all = concat_cols(dv_heads)  # [T, C]

        var qkv_parts = List[Tensor2D]()
        qkv_parts.append(d_q_all^)
        qkv_parts.append(d_k_all^)
        qkv_parts.append(d_v_all^)
        var d_qkv = concat_cols(qkv_parts)  # [T, 3C]

        return self.qkv.backward(cache.qkv_cache, d_qkv)  # [T, C]
