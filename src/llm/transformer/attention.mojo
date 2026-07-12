"""Scaled-dot-product attention: the one core serving self- and cross-attention.

Because q is a separate argument from k/v, each with its own length, the same
function is the cross-attention core: self-attention passes q, k, v from one
sequence; cross-attention passes q from the decoder and k/v from the encoder.
Nothing here cares which — the shapes [T_q, D] vs [T_k, D] are the whole story.
The pipeline reuses the tested tensor ops (matmul_transpose_b, the `@` operator,
scale, add, softmax_rows); the order is load-bearing and pinned below.
"""

from std.math import sqrt

from llm.nn.dropout import DropoutResult, dropout_backward, dropout_cached
from llm.nn.linear import Linear, LinearCache
from llm.tensor.ops import (
    add,
    concat_cols,
    matmul_transpose_a,
    matmul_transpose_b,
    scale,
    slice_cols,
    slice_rows,
    softmax_rows,
    softmax_rows_backward,
    transpose,
)
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.utils.random import Rng


struct AttentionResult(Copyable, Movable):
    """The attention output alongside the post-softmax weights.

    The weights are returned so tests can prove causality (every above-diagonal
    entry is 0) and callers can visualize what each query attends to, without
    recomputing them in a second code path.
    """

    var output: Tensor2D  # [T_q, D_v]
    var weights: Tensor2D  # [T_q, T_k] — post-softmax, each row sums to 1

    def __init__(out self, var output: Tensor2D, var weights: Tensor2D):
        self.output = output^
        self.weights = weights^

    def take_output(deinit self) -> Tensor2D:
        """Consume this result and return just the output, dropping the weights.

        The inference path needs the output only, so this moves it out instead of
        copying from a live struct.

        Returns:
            The output [T_q, D_v], moved out.
        """
        return self.output^

    def split(deinit self, mut weights_slot: Tensor2D) -> Tensor2D:
        """Consume this result: move the weights into `weights_slot`, return the output.

        A caching forward keeps both pieces; this moves each out instead of
        copying, since a field can't be transferred with `^` while the struct is
        still live.

        Args:
            weights_slot: Slot the weights are moved into.

        Returns:
            The output [T_q, D_v], moved out.
        """
        weights_slot = self.weights^
        return self.output^


def scaled_dot_product_attention(
    q: Tensor2D, k: Tensor2D, v: Tensor2D, mask: Tensor2D
) raises -> AttentionResult:
    """Attention(q, k, v) with an additive mask, in the pinned order.

    Pinned order (a wrong order is a classic attention bug):
        1. scores  = q @ k^T                     [T_q, T_k]
        2. scale   = scores * (1 / sqrt(D))      D = q.cols = d_head, not d_model
        3. add     = scale + mask                additive mask, after the scale
        4. weights = softmax_rows(add)           row-wise, stable
        5. output  = weights @ v                 [T_q, D_v]
    Scaling before the mask keeps a blocked score decisively negative.

    Args:
        q: Queries, shape [T_q, D].
        k: Keys, shape [T_k, D] (same feature width D as q).
        v: Values, shape [T_k, D_v] (same count T_k as k, own width D_v).
        mask: Additive mask, shape [T_q, T_k]; 0 attends, large-negative blocks.

    Returns:
        The output [T_q, D_v] and weights [T_q, T_k]. Reads its args only;
        allocates.

    Raises:
        Error: On any shape mismatch or a zero feature width.
    """
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

    # scores[i, j] = sum_d q[i, d] * k[j, d] = q @ k^T, computed directly (no
    # [D, T_k] transpose copy of k). step and batch scoring share this SIMD
    # kernel, so their scores stay bit-identical.
    var scores = matmul_transpose_b(q, k)  # [T_q, T_k]
    var scaled = scale(scores, 1.0 / sqrt(Float64(d)))  # 1/sqrt(d_head)
    var biased = add(scaled, mask)  # additive mask, after the scale
    var weights = softmax_rows(biased)  # [T_q, T_k], rows sum to 1
    var output = weights @ v  # [T_q, D_v]
    return AttentionResult(output^, weights^)


@fieldwise_init
struct AttentionCache(Copyable, Movable):
    """What the core's backward needs: q, k, v and the post-softmax weights.

    The mask is not stored — it is constant additive data with no gradient, and
    its effect already lives in `weights` (blocked entries are ~0). Valid only
    for the forward call that produced it.
    """

    var q: Tensor2D  # [T_q, D]
    var k: Tensor2D  # [T_k, D]
    var v: Tensor2D  # [T_k, D_v]
    var weights: Tensor2D  # [T_q, T_k], post-softmax


@fieldwise_init
struct AttentionForward(Copyable, Movable):
    """Bundle scaled_dot_product_attention_cached's output with its backward cache.
    """

    var output: Tensor2D  # [T_q, D_v]
    var cache: AttentionCache

    def split(deinit self, mut cache_slot: AttentionCache) -> Tensor2D:
        """Consume this forward: move the cache into `cache_slot`, return the output.

        Lets the head loop move a head's output and cache into their lists
        instead of copying each out of a live struct.

        Args:
            cache_slot: Slot the cache is moved into.

        Returns:
            The output [T_q, D_v], moved out.
        """
        cache_slot = self.cache^
        return self.output^


@fieldwise_init
struct AttentionGrads(Copyable, Movable):
    """The core backward's three input gradients, one per differentiable input.
    """

    var d_q: Tensor2D  # [T_q, D]
    var d_k: Tensor2D  # [T_k, D]
    var d_v: Tensor2D  # [T_k, D_v]


def scaled_dot_product_attention_cached(
    q: Tensor2D, k: Tensor2D, v: Tensor2D, mask: Tensor2D
) raises -> AttentionForward:
    """Same forward as scaled_dot_product_attention, plus the backward cache.

    Args:
        q: Queries, shape [T_q, D].
        k: Keys, shape [T_k, D].
        v: Values, shape [T_k, D_v].
        mask: Additive mask, shape [T_q, T_k].

    Returns:
        The output and the cache (q, k, v, weights) its backward needs.
        Allocates.

    Raises:
        Error: On the same shape mismatches the forward does.
    """
    var result = scaled_dot_product_attention(q, k, v, mask)
    # q/k/v are borrowed inputs the cache must own, so they copy; the output and
    # weights are owned by `result`, so move them out instead of copying.
    var weights = zeros_2d(0, 0)  # placeholder, replaced by the move
    var output = result^.split(weights)
    var cache = AttentionCache(q.copy(), k.copy(), v.copy(), weights^)
    return AttentionForward(output^, cache^)


def scaled_dot_product_attention_backward(
    cache: AttentionCache, d_out: Tensor2D
) raises -> AttentionGrads:
    """Reverse of the pinned forward order. Let s = 1/sqrt(D), D = q.cols = d_head.

    With dO = d_out [T_q, D_v]:
        dV = W^T @ dO                                       -> [T_k, D_v]
        dW = dO @ V^T                                       -> [T_q, T_k]
        dScaled = softmax_rows_backward(W, dW)              -> [T_q, T_k]
        dScores = dScaled           (mask is constant; add passes dW through)
        dQ = (dScores @ K) * s                              -> [T_q, D]
        dK = (dScores^T @ Q) * s                            -> [T_k, D]
    The scale s folds into dQ/dK once; it does not apply to dV. The mask leaks
    no gradient (softmax_rows_backward multiplies by W, and a blocked W is ~0).

    Args:
        cache: The forward cache (q, k, v, weights).
        d_out: Upstream gradient, shape [T_q, D_v].

    Returns:
        The gradients d_q [T_q, D], d_k [T_k, D], d_v [T_k, D_v]. Allocates.

    Raises:
        Error: On a shape mismatch or a zero feature width.
    """
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
    var d_v = matmul_transpose_a(cache.weights, d_out)  # W^T @ dO [T_k, D_v]
    var d_w = d_out @ transpose(cache.v)  # dO @ V^T [T_q, T_k] (v tiny, kept)
    var d_scores = softmax_rows_backward(cache.weights, d_w)  # [T_q, T_k]
    var d_q = scale(d_scores @ cache.k, s)  # [T_q, D]
    var d_k = scale(
        matmul_transpose_a(d_scores, cache.q), s
    )  # dS^T @ Q [T_k, D]
    return AttentionGrads(d_q^, d_k^, d_v^)


# ===================== Attention-weight dropout (train path) =====================
#
# GPT-2 drops the post-softmax attention weights before they weight the values:
# `output = dropout(W) @ v`. This is an additive path over the frozen core above,
# leaving the existing forward/backward untouched. With training = False (or
# p = 0) dropout_cached returns an all-ones mask, unit scale, and draws no rng, so
# this path degenerates exactly to the proven one.


@fieldwise_init
struct AttentionTrainCache(Copyable, Movable):
    """The train-path backward's inputs: the core cache plus the dropout mask.

    `weights` is the pre-dropout W (what softmax produced, fed to
    softmax_rows_backward); `drop_mask` and `inv_keep` reconstruct the dropped
    weights the value-matmul saw. Valid only for the forward call that produced it.
    """

    var q: Tensor2D  # [T_q, D]
    var k: Tensor2D  # [T_k, D]
    var v: Tensor2D  # [T_k, D_v]
    var weights: Tensor2D  # [T_q, T_k], PRE-dropout post-softmax
    var drop_mask: Tensor2D  # [T_q, T_k], entries in {0, 1}
    var inv_keep: Float64  # applied dropout scale: 1/(1-p) train, 1.0 eval/p==0


@fieldwise_init
struct AttentionTrainForward(Copyable, Movable):
    """Bundle scaled_dot_product_attention_train's output with its backward cache.
    """

    var output: Tensor2D  # [T_q, D_v]
    var cache: AttentionTrainCache

    def split(deinit self, mut cache_slot: AttentionTrainCache) -> Tensor2D:
        """Consume this forward: move the cache into `cache_slot`, return the output.

        Lets the train-path head loop move a head's output and cache into their
        lists instead of copying each out.

        Args:
            cache_slot: Slot the cache is moved into.

        Returns:
            The output [T_q, D_v], moved out.
        """
        cache_slot = self.cache^
        return self.output^


def scaled_dot_product_attention_train(
    q: Tensor2D,
    k: Tensor2D,
    v: Tensor2D,
    mask: Tensor2D,
    p: Float64,
    training: Bool,
    mut rng: Rng,
) raises -> AttentionTrainForward:
    """The core forward with GPT-2's attention-weight dropout spliced in.

    Pinned order:
        scores -> scale -> + mask -> softmax -> W -> dropout_cached(W) -> dropped_W
        output = dropped_W @ v

    Args:
        q: Queries, shape [T_q, D].
        k: Keys, shape [T_k, D].
        v: Values, shape [T_k, D_v].
        mask: Additive mask, shape [T_q, T_k].
        p: Dropout probability.
        training: Whether to apply dropout and draw from rng.
        rng: Random generator; mutated only in the training/p>0 branch.

    Returns:
        The output [T_q, D_v] and the backward cache. Allocates.

    Raises:
        Error: On the same shape mismatches the core does, or an out-of-range p.
    """
    # The base call recomputes the un-dropped W @ v and discards it, keeping the
    # pinned order in one place (the core) rather than re-spelling it here; only
    # base.weights is used.
    var base = scaled_dot_product_attention(q, k, v, mask)  # base.weights = W
    var drop = dropout_cached(base.weights, p, training, rng)  # dropped_W, mask
    var output = drop.output @ v  # dropped_W @ v -> [T_q, D_v]
    # base.weights (the PRE-dropout W) is owned by base and no longer read after
    # the dropout call, so move it into the cache; base's recomputed output is
    # unused and dropped. q/k/v are borrowed inputs, so they still copy, and
    # DropoutResult's mask field can't move out.
    var weights_pre = zeros_2d(0, 0)  # placeholder, replaced by the move
    _ = base^.split(weights_pre)
    var cache = AttentionTrainCache(
        q.copy(),
        k.copy(),
        v.copy(),
        weights_pre^,  # PRE-dropout W, for softmax_rows_backward
        drop.mask.copy(),
        drop.inv_keep,
    )
    return AttentionTrainForward(output^, cache^)


def scaled_dot_product_attention_train_backward(
    cache: AttentionTrainCache, d_out: Tensor2D
) raises -> AttentionGrads:
    """Reverse of the train forward: the core backward composed with dropout backward.

    Let dropped_W = W (elementwise) mask * inv_keep. With dO = d_out [T_q, D_v]:
        dV        = dropped_W^T @ dO      (the value matmul saw the dropped W)
        d_droppedW = dO @ V^T
        dW        = dropout_backward(mask, inv_keep, d_droppedW)
        dS        = softmax_rows_backward(W, dW)   (on the pre-dropout W)
        dQ = (dS @ K)*s,  dK = (dS^T @ Q)*s,  s = 1/sqrt(D)
    dV is fed the dropped weights (what multiplied v), while softmax_rows_backward
    is fed the pre-dropout W (what softmax produced); swapping them is the classic
    attention-dropout backward bug.

    Args:
        cache: The train forward cache.
        d_out: Upstream gradient, shape [T_q, D_v].

    Returns:
        The gradients d_q, d_k, d_v. Allocates.

    Raises:
        Error: On a shape mismatch or a zero feature width.
    """
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
    var d_v = matmul_transpose_a(
        dropped_w, d_out
    )  # dropped_W^T @ dO [T_k, D_v]
    var d_dropped_w = d_out @ transpose(cache.v)  # dO @ V^T [T_q, T_k] (v tiny)
    var d_w = dropout_backward(
        cache.drop_mask, cache.inv_keep, d_dropped_w
    )  # [T_q, T_k]
    var d_scores = softmax_rows_backward(cache.weights, d_w)  # PRE-dropout W
    var s = 1.0 / sqrt(Float64(d))
    var d_q = scale(d_scores @ cache.k, s)  # [T_q, D]
    var d_k = scale(
        matmul_transpose_a(d_scores, cache.q), s
    )  # dS^T @ Q [T_k, D]
    return AttentionGrads(d_q^, d_k^, d_v^)


@fieldwise_init
struct MHACache(Copyable, Movable):
    """What MultiHeadAttention.backward needs, mirroring the forward plumbing.

    The fused qkv Linear's cache (its input x), one AttentionCache per head, and
    the output proj Linear's cache. Valid only for the forward that produced it.
    """

    var qkv_cache: LinearCache  # holds x                    [T, C]
    var head_caches: List[AttentionCache]  # per head: q,k,v,weights
    var proj_cache: LinearCache  # holds concat(head outputs) [T, C]


@fieldwise_init
struct MHAForward(Copyable, Movable):
    """Bundle forward_cached's output with the cache its backward consumes."""

    var output: Tensor2D  # [T, C]
    var cache: MHACache

    def split(deinit self, mut cache_slot: MHACache) -> Tensor2D:
        """Consume this forward: move the cache into `cache_slot`, return the output.

        Lets a block move the (large) attention cache into its own cache instead
        of deep-copying it out of a live struct.

        Args:
            cache_slot: Slot the cache is moved into.

        Returns:
            The output [T, C], moved out.
        """
        cache_slot = self.cache^
        return self.output^


@fieldwise_init
struct MHATrainCache(Copyable, Movable):
    """The train-path twin of MHACache.

    The fused qkv Linear's cache (its input x), one AttentionTrainCache per head
    (each carrying that head's attention-weight dropout mask), and the output
    proj Linear's cache. Valid only for the forward that produced it.
    """

    var qkv_cache: LinearCache  # holds x                          [T, C]
    var head_caches: List[
        AttentionTrainCache
    ]  # per head: q,k,v,W,mask,inv_keep
    var proj_cache: LinearCache  # holds concat(head outputs)      [T, C]


@fieldwise_init
struct MHATrainForward(Copyable, Movable):
    """Bundle forward_cached_train's output with the cache its backward consumes.
    """

    var output: Tensor2D  # [T, C]
    var cache: MHATrainCache

    def split(deinit self, mut cache_slot: MHATrainCache) -> Tensor2D:
        """Consume this forward: move the cache into `cache_slot`, return the output.

        Lets a block move the (large) attention cache into its own cache instead
        of deep-copying it out of a live struct.

        Args:
            cache_slot: Slot the cache is moved into.

        Returns:
            The output [T, C], moved out.
        """
        cache_slot = self.cache^
        return self.output^


@fieldwise_init
struct MultiHeadAttention(Copyable, Movable):
    """GPT-2's multi-head self-attention with the fused QKV projection.

    One Linear(C -> 3C) (GPT-2's `c_attn`) produces Q, K, V for every head in a
    single matmul; one Linear(C -> C) (`c_proj`) mixes the concatenated head
    outputs. Head layout is contiguous, matching GPT-2: the [T, 3C] projection
    splits into three [T, C] thirds (Q, K, V), and each third into H contiguous
    [T, D] head slices with D = C / H — head h owns columns [h*D, (h+1)*D).
    """

    var qkv: Linear  # C -> 3C, fused c_attn
    var proj: Linear  # C -> C, c_proj
    var n_heads: Int

    @staticmethod
    def init_random(
        mut rng: Rng, d_model: Int, n_heads: Int
    ) raises -> MultiHeadAttention:
        """Build an MHA with both Linears drawn from GPT-2's normal(0, 0.02).

        Zero biases; qkv weights are drawn first, then proj, so a generator state
        reproduces the same layer.

        Args:
            rng: Random generator; advanced by the draws.
            d_model: Model width C; must be a positive multiple of n_heads.
            n_heads: Number of heads; must be positive.

        Returns:
            A fresh MHA. Allocates both layers.

        Raises:
            Error: Unless n_heads > 0 and d_model is a positive multiple of it
                (else the head width D = C / H is undefined or degenerate).
        """
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
        """Self-attention over one sequence, with Q, K, V all derived from x.

        The caller's mask is applied unchanged to every head.

        Args:
            x: Input sequence, shape [T, C].
            mask: Additive attention mask, shape [T, T].

        Returns:
            The output [T, C]. Reads self only; allocates.

        Raises:
            Error: On a feature-count mismatch (via qkv) or an indivisible width.
        """
        # @fieldwise_init lets a caller build this struct directly, bypassing
        # init_random's guard, so re-check n_heads rather than trap on a
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
            # Move the output out (the weights are not needed here) rather than
            # copying a field out of a live struct.
            head_outputs.append(result^.take_output())  # [T, D]
        var concatenated = concat_cols(head_outputs)  # [T, C]

        return self.proj.forward(concatenated)  # [T, C] -> [T, C]

    def step(
        self,
        x: Tensor2D,
        mut k_cache: Tensor2D,
        mut v_cache: Tensor2D,
        pos: Int,
    ) raises -> Tensor2D:
        """KV-cached single-token self-attention: one query row against the cached past.

        `forward` one row wide, reusing the same core with T_q = 1 and
        T_k = pos + 1. Writes this position's K/V thirds into cache row `pos`,
        views the valid region rows 0..pos, and per head attends with an
        all-zeros [1, t] mask — causality is enforced by what is in the cache, not
        by masking (the batch causal mask's last row is all zeros for the same
        reason).

        Args:
            x: The single query row, shape [1, C].
            k_cache: This layer's [capacity, C] key buffer, written at row `pos`.
            v_cache: This layer's [capacity, C] value buffer, written at row `pos`.
            pos: The newest position (the cache's length on entry).

        Returns:
            The output [1, C]. Reads self; mutates the cache buffers; allocates.

        Raises:
            Error: On a bad shape/config or an out-of-range pos.
        """
        if self.n_heads <= 0:
            raise Error(
                "MultiHeadAttention.step: n_heads must be positive, got "
                + String(self.n_heads)
            )
        var c = self.qkv.weight.value.cols  # in_features of c_attn = C
        if c % self.n_heads != 0:
            raise Error(
                "MultiHeadAttention.step: C ("
                + String(c)
                + ") not divisible by n_heads ("
                + String(self.n_heads)
                + ")"
            )
        if x.rows != 1:
            raise Error(
                "MultiHeadAttention.step: expected a single query row [1, C],"
                " got rows="
                + String(x.rows)
            )
        if k_cache.cols != c or v_cache.cols != c:
            raise Error(
                "MultiHeadAttention.step: cache width must be C="
                + String(c)
                + ", got k_cache.cols="
                + String(k_cache.cols)
                + " v_cache.cols="
                + String(v_cache.cols)
            )
        if k_cache.rows != v_cache.rows:
            raise Error(
                "MultiHeadAttention.step: k/v cache height mismatch, "
                + String(k_cache.rows)
                + " vs "
                + String(v_cache.rows)
            )
        if pos < 0 or pos >= k_cache.rows:
            raise Error(
                "MultiHeadAttention.step: pos "
                + String(pos)
                + " out of range for cache capacity "
                + String(k_cache.rows)
            )
        var d_head = c // self.n_heads  # per-head width D = C / H

        var qkv_row = self.qkv.forward(x)  # [1, C] -> [1, 3C]
        var q_row = slice_cols(qkv_row, 0, c)  # [1, C]
        var k_row = slice_cols(qkv_row, c, 2 * c)  # [1, C]
        var v_row = slice_cols(qkv_row, 2 * c, 3 * c)  # [1, C]

        # Write this position's K and V (the full pre-head-split rows) into the
        # cache at row `pos`, matching the batch path's k_all/v_all layout.
        for j in range(c):
            k_cache[pos, j] = k_row[0, j]
            v_cache[pos, j] = v_row[0, j]

        # The valid region is rows 0..pos: t = pos + 1 cached positions.
        var t = pos + 1
        var k_valid = slice_rows(k_cache, 0, t)  # [t, C]
        var v_valid = slice_rows(v_cache, 0, t)  # [t, C]

        var head_outputs = List[Tensor2D]()
        for h in range(self.n_heads):
            var lo = h * d_head
            var hi = lo + d_head
            var q_h = slice_cols(q_row, lo, hi)  # [1, D]
            var k_h = slice_cols(k_valid, lo, hi)  # [t, D]
            var v_h = slice_cols(v_valid, lo, hi)  # [t, D]
            # All-zeros [1, t] mask: the newest query attends to every cached key.
            var result = scaled_dot_product_attention(
                q_h, k_h, v_h, zeros_2d(1, t)
            )
            head_outputs.append(result^.take_output())  # [1, D]
        var concatenated = concat_cols(head_outputs)  # [1, C]

        return self.proj.forward(concatenated)  # [1, C] -> [1, C]

    def forward_cached(
        self, var x: Tensor2D, mask: Tensor2D
    ) raises -> MHAForward:
        """Self-attention over one sequence, capturing the cache its backward needs.

        Same computation as forward, capturing the fused-qkv cache, one
        AttentionCache per head, and the proj cache. Takes x by value and moves
        it into the qkv cache; each stage's forward is split into (output, cache)
        so both pieces move on with no [T, *] copy.

        Args:
            x: Input sequence, shape [T, C]; moved into the qkv cache.
            mask: Additive attention mask, shape [T, T].

        Returns:
            The output [T, C] and the cache, valid only for this call. Allocates.

        Raises:
            Error: On a feature-count mismatch or an indivisible width.
        """
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

        var qkv_fwd = self.qkv.forward_cached(x^)  # output [T, 3C], cache = x
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
            var head_cache = AttentionCache(
                zeros_2d(0, 0), zeros_2d(0, 0), zeros_2d(0, 0), zeros_2d(0, 0)
            )  # placeholder, replaced by the move
            var out_h = head^.split(head_cache)  # [T, D]; cache -> head_cache
            head_outputs.append(out_h^)
            head_caches.append(head_cache^)
        var concatenated = concat_cols(head_outputs)  # [T, C]
        var proj_fwd = self.proj.forward_cached(concatenated^)  # [T, C]

        # Move the qkv and proj caches into the MHA cache (the qkv output was
        # already sliced into q/k/v, so split's returned output is dropped).
        var qkv_cache = LinearCache(zeros_2d(0, 0))  # placeholder
        _ = qkv_fwd^.split(qkv_cache)
        var proj_cache = LinearCache(zeros_2d(0, 0))  # placeholder
        var output = proj_fwd^.split(proj_cache)  # [T, C]
        return MHAForward(
            output^, MHACache(qkv_cache^, head_caches^, proj_cache^)
        )

    def backward(mut self, cache: MHACache, d_out: Tensor2D) raises -> Tensor2D:
        """Reverse the forward plumbing exactly, right to left.

        proj.backward, split into H head slices, run the core backward per head,
        concat the per-head d_q/d_k/d_v back to [T, C], reassemble the fused
        d_qkv [T, 3C], then qkv.backward. slice_cols and concat_cols are exact
        inverses, so the column bookkeeping round-trips.

        Args:
            cache: The forward cache from forward_cached.
            d_out: Upstream gradient, shape [T, C].

        Returns:
            d_x [T, C]. Mutates self.proj and self.qkv parameter grads (+=);
            allocates.

        Raises:
            Error: On a shape/config mismatch.
        """
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
        var x: Tensor2D,
        mask: Tensor2D,
        p: Float64,
        training: Bool,
        mut rng: Rng,
    ) raises -> MHATrainForward:
        """The train-path twin of forward_cached, with per-head attention-weight dropout.

        Identical plumbing (fused qkv, per-head split, concat, proj), but each
        head runs the dropout core. rng is threaded through the heads in order, so
        each head draws its own [T, T] weight mask; with training = False (or
        p = 0) no head draws and the result equals forward_cached.

        Args:
            x: Input sequence, shape [T, C]; moved into the qkv cache.
            mask: Additive attention mask, shape [T, T].
            p: Dropout probability.
            training: Whether to apply dropout and draw from rng.
            rng: Random generator; mutated only in the training/p>0 branch.

        Returns:
            The output [T, C] and the cache, valid only for this call. Allocates.

        Raises:
            Error: On a feature-count mismatch, an indivisible width, or a bad p.
        """
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

        var qkv_fwd = self.qkv.forward_cached(x^)  # output [T, 3C], cache = x
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
            var head_cache = AttentionTrainCache(
                zeros_2d(0, 0),
                zeros_2d(0, 0),
                zeros_2d(0, 0),
                zeros_2d(0, 0),
                zeros_2d(0, 0),
                0.0,
            )  # placeholder, replaced by the move
            var out_h = head^.split(head_cache)  # [T, D]; cache -> head_cache
            head_outputs.append(out_h^)
            head_caches.append(head_cache^)
        var concatenated = concat_cols(head_outputs)  # [T, C]
        var proj_fwd = self.proj.forward_cached(concatenated^)  # [T, C]

        # Move the qkv and proj caches into the MHA cache (the qkv output was
        # already sliced into q/k/v, so split's returned output is dropped).
        var qkv_cache = LinearCache(zeros_2d(0, 0))  # placeholder
        _ = qkv_fwd^.split(qkv_cache)
        var proj_cache = LinearCache(zeros_2d(0, 0))  # placeholder
        var output = proj_fwd^.split(proj_cache)  # [T, C]
        return MHATrainForward(
            output^, MHATrainCache(qkv_cache^, head_caches^, proj_cache^)
        )

    def backward_train(
        mut self, cache: MHATrainCache, d_out: Tensor2D
    ) raises -> Tensor2D:
        """The train-path twin of backward, with per-head dropout backward.

        Reverse the same plumbing right to left, each head running the
        attention-weight-dropout core backward. With training = False (or p = 0)
        each head cache carries an all-ones mask and unit scale, so this returns
        exactly what backward returns.

        Args:
            cache: The train forward cache from forward_cached_train.
            d_out: Upstream gradient, shape [T, C].

        Returns:
            d_x [T, C]. Mutates self.proj and self.qkv parameter grads (+=);
            allocates.

        Raises:
            Error: On a shape/config mismatch.
        """
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
