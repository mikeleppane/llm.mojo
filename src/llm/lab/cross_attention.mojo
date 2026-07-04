# Cross-multi-head attention — the encoder-decoder lab's one genuinely new layer.
#
# Cross-attention has TWO inputs: queries come from the decoder stream `x`, keys
# and values from the encoder `memory`. That is the whole reason it cannot fuse
# its projections the way GPT-2's self-attention does. Self-attention derives Q,
# K, V from one sequence, so one Linear(C -> 3C) (`c_attn`) fuses all three.
# Here Q is a function of `x` and K, V are functions of `memory` — different
# inputs, different lengths — so only K and V can share a matmul:
#
#     q:    Linear(C -> C)    on x       (queries)
#     kv:   Linear(C -> 2C)   on memory  (keys and values, FUSED)
#     proj: Linear(C -> C)    on the concatenated head outputs
#
# The fusion boundary IS the teaching point: it is drawn exactly where the two
# inputs meet. Parameter cost is (C²+C) + (2C²+2C) + (C²+C) = 4C²+4C — the same
# total as self-MHA, only split across three Linears instead of two.
#
# No new attention math lives here. The per-head core is the SAME
# scaled_dot_product_attention Part X built with separate q vs k/v arguments
# precisely so this layer could consume it (it already accepts T_q != T_k), and
# its backward is reused unchanged. New here is only the plumbing plus the two
# input gradients: backward returns d_x (through the q projection) AND d_memory
# (through the fused kv projection).

from llm.nn.linear import Linear, LinearCache
from llm.tensor.ops import concat_cols, slice_cols
from llm.tensor.tensor2d import Tensor2D
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
    # What CrossMultiHeadAttention.backward needs, mirroring the forward
    # plumbing: the q Linear's cache (its input x), the kv Linear's cache (its
    # input memory), one AttentionCache per head, and the output proj Linear's
    # cache (its input, the concatenated head outputs). Valid only for the
    # forward call that produced it.
    var q_cache: LinearCache  # holds x                     [T_q, C]
    var kv_cache: LinearCache  # holds memory               [T_k, C]
    var head_caches: List[AttentionCache]  # per head: q,k,v,weights
    var proj_cache: LinearCache  # holds concat(head outputs) [T_q, C]


@fieldwise_init
struct CrossMHAForward(Copyable, Movable):
    # forward_cached's output plus the cache its backward consumes.
    var output: Tensor2D  # [T_q, C]
    var cache: CrossMHACache


@fieldwise_init
struct CrossMHAGrads(Copyable, Movable):
    # The two input gradients cross-attention produces, one per differentiable
    # input. d_x flows back through the q projection into the decoder stream;
    # d_memory flows back through the fused kv projection into the encoder
    # output. (The mask is constant additive data — no gradient.)
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
        # A cross-MHA with all three Linears drawn from GPT-2's normal(0, 0.02)
        # scheme and zero biases. Draw order is q, then kv, then proj (matching
        # the field order), so a given generator state reproduces the same
        # layer. Mutates rng (advances its state); allocates the three layers;
        # deterministic given the generator's state. Raises unless n_heads > 0
        # and d_model is a positive multiple of n_heads — otherwise the head
        # width D = C / H is undefined or degenerate.
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
        # @fieldwise_init lets a caller build this struct directly, bypassing
        # init_random's guard, so re-check n_heads here rather than trap on a
        # modulo-by-zero below.
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
        # Cross-attention: x [T_q, C], memory [T_k, C], mask [T_q, T_k] ->
        # [T_q, C]. Q derives from x, K and V from memory. Reads self only;
        # allocates the projections, per-head slices, and result; raises on a
        # feature-count mismatch (via the Linears), an indivisible width, or a
        # mask shape mismatch (via the core). The caller's mask is applied
        # unchanged to every head.
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
            head_outputs.append(result.output.copy())  # [T_q, D]
        var concatenated = concat_cols(head_outputs)  # [T_q, C]

        return self.proj.forward(concatenated)  # [T_q, C]

    def forward_cached(
        self, x: Tensor2D, memory: Tensor2D, mask: Tensor2D
    ) raises -> CrossMHAForward:
        # Same computation as forward, additionally capturing the cache backward
        # needs (the q and kv Linear caches, one AttentionCache per head, and the
        # proj cache). Reads self; allocates the projections, per-head slices,
        # caches, and result; raises on the same mismatches forward does. The
        # cache is valid only for this call.
        var c = self.q.weight.value.cols
        self._check_config(c)
        var d_head = c // self.n_heads

        var q_fwd = self.q.forward_cached(x)  # output [T_q, C], cache = x
        var kv_fwd = self.kv.forward_cached(
            memory
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
            head_outputs.append(head.output.copy())  # [T_q, D]
            head_caches.append(head.cache.copy())
        var concatenated = concat_cols(head_outputs)  # [T_q, C]
        var proj_fwd = self.proj.forward_cached(concatenated)  # [T_q, C]

        var cache = CrossMHACache(
            q_fwd.cache.copy(),
            kv_fwd.cache.copy(),
            head_caches^,
            proj_fwd.cache.copy(),
        )
        return CrossMHAForward(proj_fwd.output.copy(), cache^)

    def backward(
        mut self, cache: CrossMHACache, d_out: Tensor2D
    ) raises -> CrossMHAGrads:
        # Reverse the forward plumbing exactly, right to left, splitting the one
        # upstream gradient into the two input gradients d_x and d_memory:
        #   1. d_concat = proj.backward(proj_cache, d_out)      [T_q, C]
        #   2. split d_concat into H contiguous [T_q, D] head slices; per head
        #      run the core backward to get d_q_h [T_q, D], d_k_h/d_v_h [T_k, D].
        #   3. concat the per-head d_q's back to d_q_all        [T_q, C]
        #      concat the per-head d_k's, d_v's to d_k_all, d_v_all [T_k, C].
        #   4. d_kv = [d_k_all | d_v_all]                       [T_k, 2C] — the
        #      SAME K|V column order the forward's kv projection produced. (The
        #      forward slices k from columns [0, C) and v from [C, 2C); the
        #      gradient must be reassembled in that order or d_memory and the kv
        #      parameter grads are silently wrong.)
        #   5. d_x      = q.backward(q_cache, d_q_all)          [T_q, C]
        #      d_memory = kv.backward(kv_cache, d_kv)           [T_k, C]
        # slice_cols and concat_cols are exact inverses, so the column
        # bookkeeping round-trips. Mutates self.proj, self.q, self.kv parameter
        # grads (+=); allocates and returns the two input gradients; raises on a
        # shape/config mismatch.
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
        # Reset every parameter gradient (q, kv, proj — weight and bias each) to
        # zero. Mutates in place; allocates nothing; cannot raise.
        self.q.weight.zero_grad()
        self.q.bias.zero_grad()
        self.kv.weight.zero_grad()
        self.kv.bias.zero_grad()
        self.proj.weight.zero_grad()
        self.proj.bias.zero_grad()

    def apply_sgd(mut self, lr: Float64) raises:
        # One plain-SGD step (param -= lr * grad) on every parameter tensor.
        # Mutates the parameter values; raises on a shape mismatch (via
        # sgd_step, which cannot occur for well-formed layers).
        sgd_step(self.q.weight.value, self.q.weight.grad, lr)
        sgd_step(self.q.bias.value, self.q.bias.grad, lr)
        sgd_step(self.kv.weight.value, self.kv.weight.grad, lr)
        sgd_step(self.kv.bias.value, self.kv.bias.grad, lr)
        sgd_step(self.proj.weight.value, self.proj.weight.grad, lr)
        sgd_step(self.proj.bias.value, self.proj.bias.grad, lr)
