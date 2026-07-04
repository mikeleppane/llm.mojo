# TransformerBlock — GPT-2's pre-LN decoder block (self-attention only).
#
# Two sublayers, each a pre-LN residual (the layout GPT-2 uses; the original
# Transformer paper is post-LN, ln(x + sublayer(x))):
#
#     a   = x + attn(ln1(x), mask)         # masked self-attention sublayer
#     out = a + mlp(ln2(a))                # position-wise feed-forward sublayer
#
# This is the decoder-only block the real GPT stacks — no cross-attention (there
# is no encoder to attend to). The encoder-decoder lab rehearsed this exact
# residual wiring; here it is written fresh on the main line with the two things
# the lab deferred: config-driven construction and dropout in GPT-2's places.
#
# The residual backward rule (the one new gradient idea the block assembly
# teaches): for out = h + f(ln(h)) the upstream gradient reaches h by BOTH paths,
# which SUM:
#
#     d_h = d_out + ln.backward(f.backward(d_out))
#
# — d_out straight down the skip connection, plus the branch term. Dropping the
# skip term (d_h = branch only) is the classic residual bug; it is off by exactly
# the identity d_out, which the block-level finite-difference of d_x catches. No
# new tensor op is needed — `add` does both the forward residual and the backward
# sum.
#
# Dropout (training path only): GPT-2 puts residual dropout on each sublayer's
# output BEFORE the residual add — x + dropout(sublayer(ln(x))). The skip path x
# is NEVER dropped. Attention-weight dropout lives one level down, inside the
# self-attention train core. The plain `forward` is the inference path: no
# dropout, no rng — applying dropout at inference is made unrepresentable, not
# merely tested against.

from llm.nn.dropout import dropout_backward, dropout_cached
from llm.nn.layernorm import LayerNorm, LayerNormCache
from llm.nn.mlp import MLP, MLPCache
from llm.nn.parameter import Parameter
from llm.tensor.ops import add
from llm.tensor.tensor2d import Tensor2D
from llm.transformer.attention import (
    MHATrainCache,
    MultiHeadAttention,
)
from llm.utils.random import Rng


def sgd_parameter(mut p: Parameter, lr: Float64):
    # In-place plain-SGD update p.value -= lr * p.grad. Inlined here rather than
    # importing training.optimizer.sgd_step because transformer/ sits BELOW
    # training/ in the dependency layering (nn -> transformer -> {training,
    # generation}) and a lower layer must never import a higher one — the update
    # itself is a one-liner. Mutates p.value in place; allocates nothing; cannot
    # raise (value and grad always share a shape, allocated together in Parameter).
    for i in range(p.value.rows):
        for j in range(p.value.cols):
            p.value[i, j] = p.value[i, j] - lr * p.grad[i, j]


@fieldwise_init
struct BlockCache(Copyable, Movable):
    # One sub-cache per stage, in forward order, plus the two residual-dropout
    # masks. The self-attention stage caches ln1 then the train-core MHA cache
    # (which carries attention-weight dropout inside it); the feed-forward stage
    # caches ln2 then the MLP. Each residual dropout contributes its mask and
    # scale so backward reproduces the exact forward map (all-ones / unit in eval).
    # The residual adds carry no parameters, so the skip path needs no cache — it
    # is reconstructed in backward as the additive d_out term. Valid only for the
    # forward call that produced it.
    var ln1_cache: LayerNormCache
    var attn_cache: MHATrainCache
    var attn_drop_mask: Tensor2D  # [T, C], attention sublayer residual dropout
    var attn_drop_inv_keep: Float64
    var ln2_cache: LayerNormCache
    var mlp_cache: MLPCache
    var mlp_drop_mask: Tensor2D  # [T, C], MLP sublayer residual dropout
    var mlp_drop_inv_keep: Float64


@fieldwise_init
struct BlockForward(Copyable, Movable):
    # forward_cached's output plus the cache its backward consumes.
    var output: Tensor2D  # [T, C]
    var cache: BlockCache


@fieldwise_init
struct TransformerBlock(Copyable, Movable):
    var ln1: LayerNorm  # pre-attention norm
    var attn: MultiHeadAttention  # masked self-attention
    var ln2: LayerNorm  # pre-MLP norm
    var mlp: MLP  # position-wise feed-forward

    @staticmethod
    def init_random(
        mut rng: Rng, d_model: Int, n_heads: Int, d_hidden: Int
    ) raises -> TransformerBlock:
        # A GPT-2 block with default LayerNorms (weight ones, bias zeros) and
        # attention + MLP drawn from GPT-2's normal(0, 0.02). Draw order is
        # attention (qkv then proj) then MLP (up then down); the LayerNorms are
        # parameter-free at init, so they consume no draws. A given generator state
        # reproduces the same block. Mutates rng; allocates the sublayers; raises
        # on invalid dims (via the sublayer factories).
        var ln1 = LayerNorm.init_default(d_model)
        var attn = MultiHeadAttention.init_random(rng, d_model, n_heads)
        var ln2 = LayerNorm.init_default(d_model)
        var mlp = MLP.init_random(rng, d_model, d_hidden)
        return TransformerBlock(ln1^, attn^, ln2^, mlp^)

    def forward(self, x: Tensor2D, mask: Tensor2D) raises -> Tensor2D:
        # Inference path: pre-LN block with NO dropout and no rng. [T, C] + mask
        # [T, T] -> [T, C]. Reads self only; allocates the intermediates and result;
        # raises on a shape/config mismatch (via the sublayers).
        var attn_out = self.attn.forward(self.ln1.forward(x), mask)  # [T, C]
        var a = add(x, attn_out)  # residual 1: x + attn(ln1(x))
        var mlp_out = self.mlp.forward(self.ln2.forward(a))  # [T, C]
        return add(a, mlp_out)  # residual 2: a + mlp(ln2(a))

    def forward_cached(
        self,
        x: Tensor2D,
        mask: Tensor2D,
        p: Float64,
        training: Bool,
        mut rng: Rng,
    ) raises -> BlockForward:
        # Training path: the same pre-LN block, capturing each stage's cache and
        # applying GPT-2's dropout. Attention-weight dropout runs inside the
        # self-attention train core; residual dropout is applied to each sublayer's
        # output BEFORE the residual add (the skip x / a is never dropped). rng is
        # threaded attention-core-first, then the attention residual dropout, then
        # the MLP residual dropout — the fixed order a seed replays. With
        # training = False (or p = 0) every dropout site is the identity with an
        # all-ones mask and NO rng consumed, so this computes exactly what forward
        # computes. Reads self; allocates the intermediates, caches, and result;
        # mutates rng only in the training/p>0 branch; raises on a shape/config
        # mismatch or an out-of-range p. The cache is valid only for this call.
        var ln1_fwd = self.ln1.forward_cached(x)
        var attn_fwd = self.attn.forward_cached_train(
            ln1_fwd.output, mask, p, training, rng
        )
        var attn_drop = dropout_cached(attn_fwd.output, p, training, rng)
        var a = add(x, attn_drop.output)  # x + dropout(attn(ln1(x)))

        var ln2_fwd = self.ln2.forward_cached(a)
        var mlp_fwd = self.mlp.forward_cached(ln2_fwd.output)
        var mlp_drop = dropout_cached(mlp_fwd.output, p, training, rng)
        var out = add(a, mlp_drop.output)  # a + dropout(mlp(ln2(a)))

        var cache = BlockCache(
            ln1_fwd.cache.copy(),
            attn_fwd.cache.copy(),
            attn_drop.mask.copy(),
            attn_drop.inv_keep,
            ln2_fwd.cache.copy(),
            mlp_fwd.cache.copy(),
            mlp_drop.mask.copy(),
            mlp_drop.inv_keep,
        )
        return BlockForward(out^, cache^)

    def backward(
        mut self, cache: BlockCache, d_out: Tensor2D
    ) raises -> Tensor2D:
        # Reverse the two pre-LN residuals, outer (MLP) first. For
        # out = a + dropout(mlp(ln2(a))) the gradient reaches a by both paths:
        #   d_mlp_out = dropout_backward(mlp_drop, d_out)     (undo residual dropout)
        #   d_a = d_out + ln2.backward(mlp.backward(d_mlp_out))   (skip + branch)
        # Then for a = x + dropout(attn(ln1(x))) the same rule reaches x:
        #   d_attn_out = dropout_backward(attn_drop, d_a)
        #   d_x = d_a + ln1.backward(attn.backward_train(d_attn_out))  (skip + branch)
        # The bare d_out / d_a terms are the skip connections — the residual bug is
        # to drop them. Residual dropout is undone on the BRANCH only; the skip
        # never saw dropout. Mutates every sublayer's parameter grads (+=);
        # allocates and returns d_x [T, C]; raises on a shape mismatch (via the
        # sublayers).
        var d_mlp_out = dropout_backward(
            cache.mlp_drop_mask, cache.mlp_drop_inv_keep, d_out
        )
        var d_ln2_out = self.mlp.backward(cache.mlp_cache, d_mlp_out)  # [T, C]
        var d_a_branch = self.ln2.backward(cache.ln2_cache, d_ln2_out)
        var d_a = add(d_out, d_a_branch)  # skip + branch

        var d_attn_out = dropout_backward(
            cache.attn_drop_mask, cache.attn_drop_inv_keep, d_a
        )
        var d_ln1_out = self.attn.backward_train(
            cache.attn_cache, d_attn_out
        )  # [T, C]
        var d_x_branch = self.ln1.backward(cache.ln1_cache, d_ln1_out)
        return add(d_a, d_x_branch)  # skip + branch

    def zero_grad(mut self):
        # Reset every Parameter's grad to zero — the block's full inventory: ln1
        # (weight, bias), attn (qkv weight/bias, proj weight/bias), ln2, mlp (up
        # weight/bias, down weight/bias). Mutates in place; cannot raise.
        self.ln1.weight.zero_grad()
        self.ln1.bias.zero_grad()
        self.attn.qkv.weight.zero_grad()
        self.attn.qkv.bias.zero_grad()
        self.attn.proj.weight.zero_grad()
        self.attn.proj.bias.zero_grad()
        self.ln2.weight.zero_grad()
        self.ln2.bias.zero_grad()
        self.mlp.up.weight.zero_grad()
        self.mlp.up.bias.zero_grad()
        self.mlp.down.weight.zero_grad()
        self.mlp.down.bias.zero_grad()

    def apply_sgd(mut self, lr: Float64):
        # One plain-SGD step on every Parameter — the same inventory zero_grad
        # walks. Mutates parameter values in place; allocates nothing; cannot raise.
        sgd_parameter(self.ln1.weight, lr)
        sgd_parameter(self.ln1.bias, lr)
        sgd_parameter(self.attn.qkv.weight, lr)
        sgd_parameter(self.attn.qkv.bias, lr)
        sgd_parameter(self.attn.proj.weight, lr)
        sgd_parameter(self.attn.proj.bias, lr)
        sgd_parameter(self.ln2.weight, lr)
        sgd_parameter(self.ln2.bias, lr)
        sgd_parameter(self.mlp.up.weight, lr)
        sgd_parameter(self.mlp.up.bias, lr)
        sgd_parameter(self.mlp.down.weight, lr)
        sgd_parameter(self.mlp.down.bias, lr)
