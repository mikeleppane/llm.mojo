# Pre-LN Transformer blocks for the encoder-decoder lab.
#
# Both blocks are PRE-LN: each sublayer is x + sublayer(ln(x)), the layout GPT-2
# uses (the original Transformer paper is post-LN, ln(x + sublayer(x))). Pre-LN
# is chosen because its wiring is exactly what the real GPT block reuses — this
# lab is the dress rehearsal — and because pre-LN trains stably under the lab's
# plain-SGD loop, which has no learning-rate warmup to tame post-LN.
#
#     EncoderBlock:  a   = x + self_attn(ln1(x), mask)
#                    out = a + mlp(ln2(a))
#     DecoderBlock:  a   = x + self_attn(ln1(x), causal)
#                    b   = a + cross_attn(ln2(a), memory, cross_mask)
#                    out = b + mlp(ln3(b))
#
# The one new gradient rule this part teaches is the residual backward. For
# out = x + f(x) the gradient reaches x by TWO paths that SUM:
#
#     d_x = d_out + f_backward(d_out)
#
# — d_out straight down the skip connection, plus f_backward(d_out) down the
# branch. Dropping the skip term (d_x = f_backward(d_out) only) is the classic
# residual bug; it is off by exactly the identity term and the block-level
# finite-difference tests catch it. No new tensor op is needed — `add` does both
# directions.

from llm.lab.cross_attention import (
    CrossMHACache,
    CrossMultiHeadAttention,
)
from llm.lab.params import (
    zero_layernorm,
    zero_mha,
    zero_mlp,
    sgd_layernorm,
    sgd_mha,
    sgd_mlp,
)
from llm.nn.layernorm import LayerNorm, LayerNormCache
from llm.nn.linear import LinearCache
from llm.nn.mlp import MLP, MLPCache
from llm.tensor.ops import add
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.attention import (
    AttentionCache,
    MHACache,
    MultiHeadAttention,
)
from llm.utils.random import Rng


# ===================== Encoder block =====================


@fieldwise_init
struct EncoderBlockCache(Copyable, Movable):
    # One sub-cache per sublayer stage, in forward order: the ln1 cache, the
    # self-attention cache, the ln2 cache, the mlp cache. Valid only for the
    # forward call that produced it. The residual adds carry no parameters, so
    # they need no cache — the skip path is reconstructed in backward as the
    # additive d_out term.
    var ln1_cache: LayerNormCache
    var attn_cache: MHACache
    var ln2_cache: LayerNormCache
    var mlp_cache: MLPCache


@fieldwise_init
struct EncoderBlockForward(Copyable, Movable):
    # forward_cached's output plus the cache its backward consumes.
    var output: Tensor2D  # [T, C]
    var cache: EncoderBlockCache

    def take_cache(deinit self) -> EncoderBlockCache:
        # Consume this forward and hand back just the (large) cache, dropping the
        # output. The encoder loop reads the block output on for the next block,
        # then moves this whole cache into its list instead of deep-copying it.
        return self.cache^


@fieldwise_init
struct EncoderBlock(Copyable, Movable):
    var ln1: LayerNorm  # pre-attention norm
    var attn: MultiHeadAttention  # self-attention
    var ln2: LayerNorm  # pre-MLP norm
    var mlp: MLP  # position-wise feed-forward

    @staticmethod
    def init_random(
        mut rng: Rng, d_model: Int, n_heads: Int, d_hidden: Int
    ) raises -> EncoderBlock:
        # An encoder block with default LayerNorms (weight ones, bias zeros) and
        # attention + MLP drawn from GPT-2's normal(0, 0.02). Draw order is
        # attention then MLP (the LayerNorms are parameter-free at init), so a
        # given generator state reproduces the same block. Mutates rng; allocates
        # the sublayers; raises on invalid dims (via the sublayer factories).
        var ln1 = LayerNorm.init_default(d_model)
        var attn = MultiHeadAttention.init_random(rng, d_model, n_heads)
        var ln2 = LayerNorm.init_default(d_model)
        var mlp = MLP.init_random(rng, d_model, d_hidden)
        return EncoderBlock(ln1^, attn^, ln2^, mlp^)

    def forward(self, x: Tensor2D, mask: Tensor2D) raises -> Tensor2D:
        # Pre-LN encoder block: [T, C] + mask [T, T] -> [T, C]. Reads self only;
        # allocates the intermediates and result; raises on a shape/config
        # mismatch (via the sublayers).
        var attn_out = self.attn.forward(self.ln1.forward(x), mask)  # [T, C]
        var a = add(x, attn_out)  # residual 1: x + attn(ln1(x))
        var mlp_out = self.mlp.forward(self.ln2.forward(a))  # [T, C]
        return add(a, mlp_out)  # residual 2: a + mlp(ln2(a))

    def forward_cached(
        self, x: Tensor2D, mask: Tensor2D
    ) raises -> EncoderBlockForward:
        # Same computation as forward, capturing each sublayer's cache. Reads
        # self; allocates the intermediates, caches, and result; raises on a
        # shape/config mismatch. The cache is valid only for this call.
        # Each sublayer forward is split into (output, cache): the output moves
        # into the next sublayer, the cache moves into the block cache. x and a are
        # residual streams, still needed at their `add`s, so ln1/ln2 copy them.
        var ln1_fwd = self.ln1.forward_cached(
            x.copy()
        )  # x is the residual, kept
        var ln1_cache = LayerNormCache(
            zeros_2d(0, 0), List[Float64](), List[Float64]()
        )  # placeholder, replaced by the move
        var ln1_out = ln1_fwd^.split(ln1_cache)  # cache -> ln1_cache
        var attn_fwd = self.attn.forward_cached(ln1_out^, mask)
        var attn_cache = MHACache(
            LinearCache(zeros_2d(0, 0)),
            List[AttentionCache](),
            LinearCache(zeros_2d(0, 0)),
        )  # placeholder, replaced by the move
        var attn_out = attn_fwd^.split(attn_cache)  # cache -> attn_cache
        var a = add(x, attn_out)  # [T, C]
        var ln2_fwd = self.ln2.forward_cached(
            a.copy()
        )  # a is the residual, kept
        var ln2_cache = LayerNormCache(
            zeros_2d(0, 0), List[Float64](), List[Float64]()
        )  # placeholder, replaced by the move
        var ln2_out = ln2_fwd^.split(ln2_cache)  # cache -> ln2_cache
        var mlp_fwd = self.mlp.forward_cached(ln2_out^)
        var mlp_cache = MLPCache(
            LinearCache(zeros_2d(0, 0)),
            zeros_2d(0, 0),
            LinearCache(zeros_2d(0, 0)),
        )  # placeholder, replaced by the move
        var mlp_out = mlp_fwd^.split(mlp_cache)  # cache -> mlp_cache
        var out = add(a, mlp_out)  # [T, C]
        var cache = EncoderBlockCache(
            ln1_cache^, attn_cache^, ln2_cache^, mlp_cache^
        )
        return EncoderBlockForward(out^, cache^)

    def backward(
        mut self, cache: EncoderBlockCache, d_out: Tensor2D
    ) raises -> Tensor2D:
        # Reverse the two pre-LN residuals, outer first. For out = a + mlp(ln2(a))
        # the gradient reaches a by both paths and sums:
        #   d_a = d_out + ln2.backward(mlp.backward(d_out)).
        # Then for a = x + attn(ln1(x)) the same rule reaches x:
        #   d_x = d_a + ln1.backward(attn.backward(d_a)).
        # The bare d_out / d_a terms are the skip connections — dropping them is
        # the residual bug the finite-diff catches. Mutates every sublayer's
        # parameter grads (+=); allocates and returns d_x [T, C]; raises on a
        # shape mismatch (via the sublayers).
        var d_ln2_out = self.mlp.backward(cache.mlp_cache, d_out)  # [T, C]
        var d_a_branch = self.ln2.backward(cache.ln2_cache, d_ln2_out)
        var d_a = add(d_out, d_a_branch)  # skip + branch
        var d_ln1_out = self.attn.backward(cache.attn_cache, d_a)  # [T, C]
        var d_x_branch = self.ln1.backward(cache.ln1_cache, d_ln1_out)
        return add(d_a, d_x_branch)  # skip + branch

    def zero_grad(mut self):
        zero_layernorm(self.ln1)
        zero_mha(self.attn)
        zero_layernorm(self.ln2)
        zero_mlp(self.mlp)

    def apply_sgd(mut self, lr: Float64) raises:
        sgd_layernorm(self.ln1, lr)
        sgd_mha(self.attn, lr)
        sgd_layernorm(self.ln2, lr)
        sgd_mlp(self.mlp, lr)


# ===================== Decoder block =====================


@fieldwise_init
struct DecoderBlockCache(Copyable, Movable):
    # One sub-cache per sublayer stage, in forward order: masked self-attention
    # (ln1, self_attn), cross-attention (ln2, cross_attn), feed-forward (ln3,
    # mlp). Valid only for the forward call that produced it.
    var ln1_cache: LayerNormCache
    var self_attn_cache: MHACache
    var ln2_cache: LayerNormCache
    var cross_attn_cache: CrossMHACache
    var ln3_cache: LayerNormCache
    var mlp_cache: MLPCache


@fieldwise_init
struct DecoderBlockForward(Copyable, Movable):
    # forward_cached's output plus the cache its backward consumes.
    var output: Tensor2D  # [T_tgt, C]
    var cache: DecoderBlockCache

    def take_cache(deinit self) -> DecoderBlockCache:
        # Consume this forward and hand back just the (large) cache, dropping the
        # output. The decoder loop reads the block output on for the next block,
        # then moves this whole cache into its list instead of deep-copying it.
        return self.cache^


@fieldwise_init
struct DecoderBlockGrads(Copyable, Movable):
    # The decoder block's two input gradients: d_x back into the decoder stream,
    # d_memory back into the encoder output (produced ONLY by the cross-attention
    # sublayer — the self-attention and MLP never touch memory).
    var d_x: Tensor2D  # [T_tgt, C]
    var d_memory: Tensor2D  # [T_src, C]


@fieldwise_init
struct DecoderBlock(Copyable, Movable):
    var ln1: LayerNorm  # pre-self-attention norm
    var self_attn: MultiHeadAttention  # masked self-attention
    var ln2: LayerNorm  # pre-cross-attention norm
    var cross_attn: CrossMultiHeadAttention  # cross-attention over memory
    var ln3: LayerNorm  # pre-MLP norm
    var mlp: MLP  # position-wise feed-forward

    @staticmethod
    def init_random(
        mut rng: Rng, d_model: Int, n_heads: Int, d_hidden: Int
    ) raises -> DecoderBlock:
        # A decoder block with default LayerNorms and self-attention,
        # cross-attention, MLP drawn from GPT-2's normal(0, 0.02). Draw order is
        # self-attention, cross-attention, MLP. Mutates rng; allocates the
        # sublayers; raises on invalid dims (via the sublayer factories).
        var ln1 = LayerNorm.init_default(d_model)
        var self_attn = MultiHeadAttention.init_random(rng, d_model, n_heads)
        var ln2 = LayerNorm.init_default(d_model)
        var cross_attn = CrossMultiHeadAttention.init_random(
            rng, d_model, n_heads
        )
        var ln3 = LayerNorm.init_default(d_model)
        var mlp = MLP.init_random(rng, d_model, d_hidden)
        return DecoderBlock(ln1^, self_attn^, ln2^, cross_attn^, ln3^, mlp^)

    def forward(
        self,
        x: Tensor2D,
        memory: Tensor2D,
        self_mask: Tensor2D,
        cross_mask: Tensor2D,
    ) raises -> Tensor2D:
        # Pre-LN decoder block: x [T_tgt, C], memory [T_src, C], self_mask
        # [T_tgt, T_tgt] (causal), cross_mask [T_tgt, T_src] -> [T_tgt, C]. Reads
        # self only; allocates the intermediates and result; raises on a
        # shape/config mismatch (via the sublayers).
        var self_out = self.self_attn.forward(
            self.ln1.forward(x), self_mask
        )  # [T_tgt, C]
        var a = add(x, self_out)  # residual 1
        var cross_out = self.cross_attn.forward(
            self.ln2.forward(a), memory, cross_mask
        )  # [T_tgt, C]
        var b = add(a, cross_out)  # residual 2
        var mlp_out = self.mlp.forward(self.ln3.forward(b))  # [T_tgt, C]
        return add(b, mlp_out)  # residual 3

    def forward_cached(
        self,
        x: Tensor2D,
        memory: Tensor2D,
        self_mask: Tensor2D,
        cross_mask: Tensor2D,
    ) raises -> DecoderBlockForward:
        # Same computation as forward, capturing each sublayer's cache. Reads
        # self; allocates the intermediates, caches, and result; raises on a
        # shape/config mismatch. The cache is valid only for this call.
        # Each sublayer forward is split into (output, cache): the output moves
        # into the next sublayer, the cache moves into the block cache. x/a/b are
        # residual streams, still needed at their `add`s, so the LayerNorms copy
        # them; memory is shared across decoder blocks, so cross-attention copies
        # it into its own cache.
        var ln1_fwd = self.ln1.forward_cached(
            x.copy()
        )  # x is the residual, kept
        var ln1_cache = LayerNormCache(
            zeros_2d(0, 0), List[Float64](), List[Float64]()
        )  # placeholder, replaced by the move
        var ln1_out = ln1_fwd^.split(ln1_cache)  # cache -> ln1_cache
        var self_fwd = self.self_attn.forward_cached(ln1_out^, self_mask)
        var self_cache = MHACache(
            LinearCache(zeros_2d(0, 0)),
            List[AttentionCache](),
            LinearCache(zeros_2d(0, 0)),
        )  # placeholder, replaced by the move
        var self_out = self_fwd^.split(self_cache)  # cache -> self_cache
        var a = add(x, self_out)  # [T_tgt, C]

        var ln2_fwd = self.ln2.forward_cached(
            a.copy()
        )  # a is the residual, kept
        var ln2_cache = LayerNormCache(
            zeros_2d(0, 0), List[Float64](), List[Float64]()
        )  # placeholder, replaced by the move
        var ln2_out = ln2_fwd^.split(ln2_cache)  # cache -> ln2_cache
        var cross_fwd = self.cross_attn.forward_cached(
            ln2_out^, memory.copy(), cross_mask  # memory is shared, kept
        )
        var cross_cache = CrossMHACache(
            LinearCache(zeros_2d(0, 0)),
            LinearCache(zeros_2d(0, 0)),
            List[AttentionCache](),
            LinearCache(zeros_2d(0, 0)),
        )  # placeholder, replaced by the move
        var cross_out = cross_fwd^.split(cross_cache)  # cache -> cross_cache
        var b = add(a, cross_out)  # [T_tgt, C]

        var ln3_fwd = self.ln3.forward_cached(
            b.copy()
        )  # b is the residual, kept
        var ln3_cache = LayerNormCache(
            zeros_2d(0, 0), List[Float64](), List[Float64]()
        )  # placeholder, replaced by the move
        var ln3_out = ln3_fwd^.split(ln3_cache)  # cache -> ln3_cache
        var mlp_fwd = self.mlp.forward_cached(ln3_out^)
        var mlp_cache = MLPCache(
            LinearCache(zeros_2d(0, 0)),
            zeros_2d(0, 0),
            LinearCache(zeros_2d(0, 0)),
        )  # placeholder, replaced by the move
        var mlp_out = mlp_fwd^.split(mlp_cache)  # cache -> mlp_cache
        var out = add(b, mlp_out)  # [T_tgt, C]

        var cache = DecoderBlockCache(
            ln1_cache^,
            self_cache^,
            ln2_cache^,
            cross_cache^,
            ln3_cache^,
            mlp_cache^,
        )
        return DecoderBlockForward(out^, cache^)

    def backward(
        mut self, cache: DecoderBlockCache, d_out: Tensor2D
    ) raises -> DecoderBlockGrads:
        # Reverse the three pre-LN residuals, outer first, threading d_x back to
        # the decoder stream and collecting the single d_memory the
        # cross-attention produces. Each residual out = h + f(ln(h)) contributes
        # d_h = d_out + ln.backward(f.backward(d_out)) — skip plus branch:
        #   d_b = d_out + ln3.backward(mlp.backward(d_out))
        #   cross_grads = cross_attn.backward(d_b) -> {d_x = d(ln2 out), d_memory}
        #   d_a = d_b + ln2.backward(cross_grads.d_x)
        #   d_x = d_a + ln1.backward(self_attn.backward(d_a))
        # d_memory comes ONLY from cross_attn — the self-attention and MLP branch
        # never see memory. Mutates every sublayer's parameter grads (+=);
        # allocates and returns (d_x [T_tgt, C], d_memory [T_src, C]); raises on a
        # shape mismatch.
        var d_ln3_out = self.mlp.backward(cache.mlp_cache, d_out)
        var d_b_branch = self.ln3.backward(cache.ln3_cache, d_ln3_out)
        var d_b = add(d_out, d_b_branch)  # skip + branch

        var cross_grads = self.cross_attn.backward(cache.cross_attn_cache, d_b)
        var d_a_branch = self.ln2.backward(cache.ln2_cache, cross_grads.d_x)
        var d_a = add(d_b, d_a_branch)  # skip + branch
        var d_memory = cross_grads.d_memory.copy()  # [T_src, C]

        var d_ln1_out = self.self_attn.backward(cache.self_attn_cache, d_a)
        var d_x_branch = self.ln1.backward(cache.ln1_cache, d_ln1_out)
        var d_x = add(d_a, d_x_branch)  # skip + branch
        return DecoderBlockGrads(d_x^, d_memory^)

    def zero_grad(mut self):
        zero_layernorm(self.ln1)
        zero_mha(self.self_attn)
        zero_layernorm(self.ln2)
        self.cross_attn.zero_grad()
        zero_layernorm(self.ln3)
        zero_mlp(self.mlp)

    def apply_sgd(mut self, lr: Float64) raises:
        sgd_layernorm(self.ln1, lr)
        sgd_mha(self.self_attn, lr)
        sgd_layernorm(self.ln2, lr)
        self.cross_attn.apply_sgd(lr)
        sgd_layernorm(self.ln3, lr)
        sgd_mlp(self.mlp, lr)
