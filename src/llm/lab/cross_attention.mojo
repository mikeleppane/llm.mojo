"""Cross-multi-head attention: queries from the decoder, keys/values from memory.

Because Q derives from the decoder stream and K, V from the encoder memory
(different inputs, different lengths), the projections cannot fully fuse the way
GPT-2 self-attention does. Only K and V share a matmul:

    q:    Linear(C -> C)    on x       (queries)
    kv:   Linear(C -> 2C)   on memory  (keys and values, fused)
    proj: Linear(C -> C)    on the concatenated head outputs

The per-head core reuses scaled_dot_product_attention unchanged (it already
accepts T_q != T_k); backward returns two input gradients, d_x through the q
projection and d_memory through the fused kv projection.
"""

from llm.nn.linear import Linear, LinearCache
from llm.tensor.ops import concat_cols, slice_cols
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.attention import (
    AttentionCache,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_cached,
)
from llm.training.optimizer import sgd_step
from llm.utils.random import Rng


@fieldwise_init
struct CrossMHACache(Copyable, Movable):
    """Everything CrossMultiHeadAttention.backward needs from a forward pass.

    Mirrors the forward plumbing: the q Linear cache (input x), the kv Linear
    cache (input memory), one AttentionCache per head, and the proj Linear cache
    (its input, the concatenated head outputs). Valid only for the forward call
    that produced it.
    """

    var q_cache: LinearCache  # holds x                     [T_q, C]
    var kv_cache: LinearCache  # holds memory               [T_k, C]
    var head_caches: List[AttentionCache]  # per head: q,k,v,weights
    var proj_cache: LinearCache  # holds concat(head outputs) [T_q, C]


@fieldwise_init
struct CrossMHAForward(Copyable, Movable):
    """Output of forward_cached plus the cache its backward consumes."""

    var output: Tensor2D  # [T_q, C]
    var cache: CrossMHACache

    def split(deinit self, mut cache_slot: CrossMHACache) -> Tensor2D:
        """Consume this forward, moving the cache into a slot and returning output.

        Lets a decoder block take the large cross-attention cache without
        deep-copying it out of a live struct.

        Args:
            cache_slot: Destination the cache moves into.

        Returns:
            The output tensor, shape [T_q, C].
        """
        cache_slot = self.cache^
        return self.output^


@fieldwise_init
struct CrossMHAGrads(Copyable, Movable):
    """The two input gradients cross-attention produces.

    d_x flows back through the q projection into the decoder stream; d_memory
    flows back through the fused kv projection into the encoder output. The mask
    is constant additive data with no gradient.
    """

    var d_x: Tensor2D  # [T_q, C]  through the q projection
    var d_memory: Tensor2D  # [T_k, C]  through the kv projection


@fieldwise_init
struct CrossMultiHeadAttention(Copyable, Movable):
    var q: Linear  # C -> C,  queries from the decoder stream x
    var kv: Linear  # C -> 2C, keys AND values from memory (fused)
    var proj: Linear  # C -> C,  mixes the concatenated head outputs
    var n_heads: Int

    @staticmethod
    def init_random(
        mut rng: Rng, d_model: Int, n_heads: Int
    ) raises -> CrossMultiHeadAttention:
        """Build a seeded cross-MHA from GPT-2's normal(0, 0.02) init.

        Draw order is q, then kv, then proj (field order), so a given generator
        state reproduces the same layer.

        Args:
            rng: Random generator; its state is advanced.
            d_model: Model width C; must be a positive multiple of n_heads.
            n_heads: Number of heads; must be positive.

        Returns:
            A new layer. Allocates the three Linears.

        Raises:
            Error: If n_heads <= 0 or d_model is not a positive multiple of
                n_heads, leaving the head width C / H undefined or degenerate.
        """
        if n_heads <= 0:
            raise Error(
                "CrossMultiHeadAttention.init_random: n_heads must be positive,"
                " got "
                + String(n_heads)
            )
        if d_model <= 0:
            raise Error(
                "CrossMultiHeadAttention.init_random: d_model must be positive,"
                " got "
                + String(d_model)
            )
        if d_model % n_heads != 0:
            raise Error(
                "CrossMultiHeadAttention.init_random: d_model ("
                + String(d_model)
                + ") must be divisible by n_heads ("
                + String(n_heads)
                + ")"
            )
        var q = Linear.init_random(rng, d_model, d_model)  # C -> C
        var kv = Linear.init_random(rng, d_model, 2 * d_model)  # C -> 2C
        var proj = Linear.init_random(rng, d_model, d_model)  # C -> C
        return CrossMultiHeadAttention(q^, kv^, proj^, n_heads)

    def _check_config(self, c: Int) raises:
        """Re-check head config, since @fieldwise_init bypasses init_random's guard.

        Args:
            c: Model width C to validate against n_heads.

        Raises:
            Error: If n_heads <= 0 or C is not divisible by n_heads.
        """
        if self.n_heads <= 0:
            raise Error(
                "CrossMultiHeadAttention: n_heads must be positive, got "
                + String(self.n_heads)
            )
        if c % self.n_heads != 0:
            raise Error(
                "CrossMultiHeadAttention: C ("
                + String(c)
                + ") not divisible by n_heads ("
                + String(self.n_heads)
                + ")"
            )

    def forward(
        self, x: Tensor2D, memory: Tensor2D, mask: Tensor2D
    ) raises -> Tensor2D:
        """Compute cross-attention: Q from x, K and V from memory.

        The caller's mask is applied unchanged to every head.

        Args:
            x: Decoder queries, shape [T_q, C].
            memory: Encoder keys/values source, shape [T_k, C].
            mask: Additive attention mask, shape [T_q, T_k].

        Returns:
            Attention output, shape [T_q, C]. Reads self only; allocates.

        Raises:
            Error: On a feature-count mismatch, an indivisible width, or a mask
                shape mismatch.
        """
        var c = self.q.weight.value.cols  # in_features of q = C
        self._check_config(c)
        var d_head = c // self.n_heads  # per-head width D = C / H

        var q_all = self.q.forward(x)  # [T_q, C]
        var kv_all = self.kv.forward(memory)  # [T_k, 2C]
        var k_all = slice_cols(kv_all, 0, c)  # [T_k, C]
        var v_all = slice_cols(kv_all, c, 2 * c)  # [T_k, C]

        var head_outputs = List[Tensor2D]()
        for h in range(self.n_heads):
            var lo = h * d_head
            var hi = lo + d_head
            var q_h = slice_cols(q_all, lo, hi)  # [T_q, D]
            var k_h = slice_cols(k_all, lo, hi)  # [T_k, D]
            var v_h = slice_cols(v_all, lo, hi)  # [T_k, D]
            var result = scaled_dot_product_attention(q_h, k_h, v_h, mask)
            # Move the output out (the weights are not needed here) rather than
            # copying a field out of a live struct.
            head_outputs.append(result^.take_output())  # [T_q, D]
        var concatenated = concat_cols(head_outputs)  # [T_q, C]

        return self.proj.forward(concatenated)  # [T_q, C]

    def forward_cached(
        self, var x: Tensor2D, var memory: Tensor2D, mask: Tensor2D
    ) raises -> CrossMHAForward:
        """Compute cross-attention and capture the cache backward needs.

        Takes x and memory by value and moves them into the q and kv caches; each
        stage's forward is split into (output, cache) to avoid a [T, *] copy.

        Args:
            x: Decoder queries, shape [T_q, C]; moved into the q cache.
            memory: Encoder keys/values, shape [T_k, C]; moved into the kv cache.
            mask: Additive attention mask, shape [T_q, T_k].

        Returns:
            Output [T_q, C] plus a cache valid only for this call. Allocates.

        Raises:
            Error: On the same mismatches forward raises.
        """
        var c = self.q.weight.value.cols
        self._check_config(c)
        var d_head = c // self.n_heads

        var q_fwd = self.q.forward_cached(x^)  # output [T_q, C], cache = x
        var kv_fwd = self.kv.forward_cached(
            memory^
        )  # output [T_k, 2C], cache = memory
        var k_all = slice_cols(kv_fwd.output, 0, c)  # [T_k, C]
        var v_all = slice_cols(kv_fwd.output, c, 2 * c)  # [T_k, C]

        var head_outputs = List[Tensor2D]()
        var head_caches = List[AttentionCache]()
        for h in range(self.n_heads):
            var lo = h * d_head
            var hi = lo + d_head
            var q_h = slice_cols(q_fwd.output, lo, hi)  # [T_q, D]
            var k_h = slice_cols(k_all, lo, hi)  # [T_k, D]
            var v_h = slice_cols(v_all, lo, hi)  # [T_k, D]
            var head = scaled_dot_product_attention_cached(q_h, k_h, v_h, mask)
            var head_cache = AttentionCache(
                zeros_2d(0, 0), zeros_2d(0, 0), zeros_2d(0, 0), zeros_2d(0, 0)
            )  # placeholder, replaced by the move
            var out_h = head^.split(head_cache)  # [T_q, D]; cache -> head_cache
            head_outputs.append(out_h^)
            head_caches.append(head_cache^)
        var concatenated = concat_cols(head_outputs)  # [T_q, C]
        var proj_fwd = self.proj.forward_cached(concatenated^)  # [T_q, C]

        # Move the q, kv, and proj caches into the cross-attention cache (the q and
        # kv outputs were already sliced, so those splits' outputs are dropped).
        var q_cache = LinearCache(zeros_2d(0, 0))  # placeholder
        _ = q_fwd^.split(q_cache)
        var kv_cache = LinearCache(zeros_2d(0, 0))  # placeholder
        _ = kv_fwd^.split(kv_cache)
        var proj_cache = LinearCache(zeros_2d(0, 0))  # placeholder
        var output = proj_fwd^.split(proj_cache)  # [T_q, C]
        return CrossMHAForward(
            output^,
            CrossMHACache(q_cache^, kv_cache^, head_caches^, proj_cache^),
        )

    def backward(
        mut self, cache: CrossMHACache, d_out: Tensor2D
    ) raises -> CrossMHAGrads:
        """Reverse the forward plumbing, splitting d_out into d_x and d_memory.

        Backprop runs proj, then per-head core, then reassembles the fused-kv
        gradient as [d_k | d_v] in the same K|V column order the forward
        projection produced (or d_memory and the kv grads are silently wrong),
        then the q and kv projections. Mutates self.proj, self.q, self.kv
        parameter grads (+=).

        Args:
            cache: The cache from the matching forward_cached call.
            d_out: Upstream gradient, shape [T_q, C].

        Returns:
            d_x [T_q, C] and d_memory [T_k, C]. Allocates.

        Raises:
            Error: On a shape/config mismatch or a head-count mismatch.
        """
        var c = self.q.weight.value.cols
        self._check_config(c)
        if len(cache.head_caches) != self.n_heads:
            raise Error(
                "CrossMultiHeadAttention.backward: cache has "
                + String(len(cache.head_caches))
                + " head caches but n_heads is "
                + String(self.n_heads)
            )
        var d_head = c // self.n_heads

        var d_concat = self.proj.backward(cache.proj_cache, d_out)  # [T_q, C]

        var dq_heads = List[Tensor2D]()
        var dk_heads = List[Tensor2D]()
        var dv_heads = List[Tensor2D]()
        for h in range(self.n_heads):
            var lo = h * d_head
            var hi = lo + d_head
            var d_head_out = slice_cols(d_concat, lo, hi)  # [T_q, D]
            var grads = scaled_dot_product_attention_backward(
                cache.head_caches[h], d_head_out
            )
            dq_heads.append(grads.d_q.copy())  # [T_q, D]
            dk_heads.append(grads.d_k.copy())  # [T_k, D]
            dv_heads.append(grads.d_v.copy())  # [T_k, D]
        var d_q_all = concat_cols(dq_heads)  # [T_q, C]
        var d_k_all = concat_cols(dk_heads)  # [T_k, C]
        var d_v_all = concat_cols(dv_heads)  # [T_k, C]

        # Reassemble the fused-kv gradient in the SAME K|V column order the
        # forward's projection produced (k first, then v).
        var kv_parts = List[Tensor2D]()
        kv_parts.append(d_k_all^)
        kv_parts.append(d_v_all^)
        var d_kv = concat_cols(kv_parts)  # [T_k, 2C]

        var d_x = self.q.backward(cache.q_cache, d_q_all)  # [T_q, C]
        var d_memory = self.kv.backward(cache.kv_cache, d_kv)  # [T_k, C]
        return CrossMHAGrads(d_x^, d_memory^)

    def zero_grad(mut self):
        """Reset every parameter gradient (q, kv, proj weight and bias) to zero.
        """
        self.q.weight.zero_grad()
        self.q.bias.zero_grad()
        self.kv.weight.zero_grad()
        self.kv.bias.zero_grad()
        self.proj.weight.zero_grad()
        self.proj.bias.zero_grad()

    def apply_sgd(mut self, lr: Float64) raises:
        """Apply one plain-SGD step (param -= lr * grad) to every parameter.

        Args:
            lr: Learning rate.

        Raises:
            Error: On a shape mismatch (never for a well-formed layer).
        """
        sgd_step(self.q.weight.value, self.q.weight.grad, lr)
        sgd_step(self.q.bias.value, self.q.bias.grad, lr)
        sgd_step(self.kv.weight.value, self.kv.weight.grad, lr)
        sgd_step(self.kv.bias.value, self.kv.bias.grad, lr)
        sgd_step(self.proj.weight.value, self.proj.weight.grad, lr)
        sgd_step(self.proj.bias.value, self.proj.bias.grad, lr)
