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
from llm.nn.linear import LinearCache
from llm.nn.mlp import MLP, MLPCache
from llm.nn.optim import adamw_update, sgd_update
from llm.nn.parameter import Parameter
from llm.tensor.ops import add
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.attention import (
    AttentionTrainCache,
    MHATrainCache,
    MultiHeadAttention,
)
from llm.utils.random import Rng

# The number of Parameters one block owns, walked in the fixed order below (ln1
# weight/bias, attn qkv weight/bias, attn proj weight/bias, ln2 weight/bias, mlp
# up weight/bias, mlp down weight/bias). Every walk method visits exactly these
# twelve in exactly this order — the contract the optimizer, gradient clipping,
# and the checkpoint format all lean on.
comptime BLOCK_PARAM_COUNT = 12


@fieldwise_init
struct ParamShape(Copyable, Movable):
    # A single parameter tensor's shape, produced by the walk. A named pair
    # rather than a bare tuple so `.rows`/`.cols` read clearly at every use site
    # (sizing optimizer state, validating a checkpoint header).
    var rows: Int
    var cols: Int


def _grad_sum_sq(p: Parameter) -> Float64:
    # Sum of squares of one Parameter's gradient — a partial contribution to the
    # global gradient norm. Reads p.grad; allocates nothing; cannot raise.
    var s = 0.0
    for i in range(p.grad.rows):
        for j in range(p.grad.cols):
            var g = p.grad[i, j]
            s += g * g
    return s


def _grad_scale(mut p: Parameter, factor: Float64):
    # Multiply one Parameter's gradient in place by `factor` (gradient clipping).
    # Mutates p.grad; allocates nothing; cannot raise.
    for i in range(p.grad.rows):
        for j in range(p.grad.cols):
            p.grad[i, j] = p.grad[i, j] * factor


def _load_value(mut p: Parameter, src: Tensor2D) raises:
    # Copy `src` into one Parameter's value in place (checkpoint restore).
    # Mutates p.value; allocates nothing; raises on a shape mismatch (a header
    # that passed validation but a tensor that did not — a corrupt file).
    if src.rows != p.value.rows or src.cols != p.value.cols:
        raise Error(
            "import_parameters: tensor shape ("
            + String(src.rows)
            + ", "
            + String(src.cols)
            + ") does not match parameter ("
            + String(p.value.rows)
            + ", "
            + String(p.value.cols)
            + ")"
        )
    for i in range(src.rows):
        for j in range(src.cols):
            p.value[i, j] = src[i, j]


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

    def take_cache(deinit self) -> BlockCache:
        # Consume this forward and hand back just the (large) cache, dropping the
        # output. The model loop reads the block output on for the next block, then
        # moves this whole cache into its list instead of deep-copying it.
        return self.cache^


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
        # ln1 keeps its own copy of x because x is the residual stream, still
        # needed at the `x + ...` add below. Each stage's forward is split into
        # (output, cache): the output moves into the next stage, the cache moves
        # into this block's cache — no [T, *] activation is copied.
        var ln1_fwd = self.ln1.forward_cached(
            x.copy()
        )  # x is the residual, kept
        var ln1_cache = LayerNormCache(
            zeros_2d(0, 0), List[Float64](), List[Float64]()
        )  # placeholder, replaced by the move
        var ln1_out = ln1_fwd^.split(ln1_cache)  # cache -> ln1_cache
        var attn_fwd = self.attn.forward_cached_train(
            ln1_out^, mask, p, training, rng
        )
        var attn_cache = MHATrainCache(
            LinearCache(zeros_2d(0, 0)),
            List[AttentionTrainCache](),
            LinearCache(zeros_2d(0, 0)),
        )  # placeholder, replaced by the move
        var attn_out = attn_fwd^.split(attn_cache)  # cache -> attn_cache
        var attn_drop = dropout_cached(attn_out, p, training, rng)
        var a = add(x, attn_drop.output)  # x + dropout(attn(ln1(x)))

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
        var mlp_drop = dropout_cached(mlp_out, p, training, rng)
        var out = add(a, mlp_drop.output)  # a + dropout(mlp(ln2(a)))

        var cache = BlockCache(
            ln1_cache^,
            attn_cache^,
            # DropoutResult is a shared temporary; its mask field can't be moved
            # out, so the cache copies it.
            attn_drop.mask.copy(),
            attn_drop.inv_keep,
            ln2_cache^,
            mlp_cache^,
            mlp_drop.mask.copy(),  # same: DropoutResult's field can't move out
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
        # walks, in the same order. Delegates to nn.optim.sgd_update (transformer/
        # may import nn/, which owns the Parameter-level update math). Mutates
        # parameter values in place; allocates nothing; cannot raise.
        sgd_update(self.ln1.weight, lr)
        sgd_update(self.ln1.bias, lr)
        sgd_update(self.attn.qkv.weight, lr)
        sgd_update(self.attn.qkv.bias, lr)
        sgd_update(self.attn.proj.weight, lr)
        sgd_update(self.attn.proj.bias, lr)
        sgd_update(self.ln2.weight, lr)
        sgd_update(self.ln2.bias, lr)
        sgd_update(self.mlp.up.weight, lr)
        sgd_update(self.mlp.up.bias, lr)
        sgd_update(self.mlp.down.weight, lr)
        sgd_update(self.mlp.down.bias, lr)

    def parameter_shapes(self, mut out: List[ParamShape]):
        # Append this block's 12 parameter shapes in walk order. Reads self;
        # grows `out`; cannot raise.
        out.append(
            ParamShape(self.ln1.weight.value.rows, self.ln1.weight.value.cols)
        )
        out.append(
            ParamShape(self.ln1.bias.value.rows, self.ln1.bias.value.cols)
        )
        out.append(
            ParamShape(
                self.attn.qkv.weight.value.rows, self.attn.qkv.weight.value.cols
            )
        )
        out.append(
            ParamShape(
                self.attn.qkv.bias.value.rows, self.attn.qkv.bias.value.cols
            )
        )
        out.append(
            ParamShape(
                self.attn.proj.weight.value.rows,
                self.attn.proj.weight.value.cols,
            )
        )
        out.append(
            ParamShape(
                self.attn.proj.bias.value.rows, self.attn.proj.bias.value.cols
            )
        )
        out.append(
            ParamShape(self.ln2.weight.value.rows, self.ln2.weight.value.cols)
        )
        out.append(
            ParamShape(self.ln2.bias.value.rows, self.ln2.bias.value.cols)
        )
        out.append(
            ParamShape(
                self.mlp.up.weight.value.rows, self.mlp.up.weight.value.cols
            )
        )
        out.append(
            ParamShape(self.mlp.up.bias.value.rows, self.mlp.up.bias.value.cols)
        )
        out.append(
            ParamShape(
                self.mlp.down.weight.value.rows, self.mlp.down.weight.value.cols
            )
        )
        out.append(
            ParamShape(
                self.mlp.down.bias.value.rows, self.mlp.down.bias.value.cols
            )
        )

    def parameter_decay_flags(self, mut out: List[Bool]):
        # Append this block's 12 weight-decay flags in walk order, the GPT-family
        # partition: the four weight MATRICES (qkv, proj, mlp up, mlp down) decay;
        # the four biases and the four LayerNorm vectors (ln1/ln2 weight and bias)
        # do not. Reads nothing; grows `out`; cannot raise.
        out.append(False)  # ln1.weight  (LayerNorm vector)
        out.append(False)  # ln1.bias    (LayerNorm vector)
        out.append(True)  # attn.qkv.weight  (matrix)
        out.append(False)  # attn.qkv.bias
        out.append(True)  # attn.proj.weight  (matrix)
        out.append(False)  # attn.proj.bias
        out.append(False)  # ln2.weight  (LayerNorm vector)
        out.append(False)  # ln2.bias    (LayerNorm vector)
        out.append(True)  # mlp.up.weight  (matrix)
        out.append(False)  # mlp.up.bias
        out.append(True)  # mlp.down.weight  (matrix)
        out.append(False)  # mlp.down.bias

    def grad_norm_sq(self) -> Float64:
        # Sum of squares over this block's 12 gradients — a partial contribution
        # to the model's global gradient norm (the whole-model vector norm is the
        # sqrt of the sum of these across every Parameter). Reads self; allocates
        # nothing; cannot raise.
        var s = 0.0
        s += _grad_sum_sq(self.ln1.weight)
        s += _grad_sum_sq(self.ln1.bias)
        s += _grad_sum_sq(self.attn.qkv.weight)
        s += _grad_sum_sq(self.attn.qkv.bias)
        s += _grad_sum_sq(self.attn.proj.weight)
        s += _grad_sum_sq(self.attn.proj.bias)
        s += _grad_sum_sq(self.ln2.weight)
        s += _grad_sum_sq(self.ln2.bias)
        s += _grad_sum_sq(self.mlp.up.weight)
        s += _grad_sum_sq(self.mlp.up.bias)
        s += _grad_sum_sq(self.mlp.down.weight)
        s += _grad_sum_sq(self.mlp.down.bias)
        return s

    def scale_grads(mut self, factor: Float64):
        # Multiply every gradient in this block by `factor` in place (gradient
        # clipping). Mutates self's grads; allocates nothing; cannot raise.
        _grad_scale(self.ln1.weight, factor)
        _grad_scale(self.ln1.bias, factor)
        _grad_scale(self.attn.qkv.weight, factor)
        _grad_scale(self.attn.qkv.bias, factor)
        _grad_scale(self.attn.proj.weight, factor)
        _grad_scale(self.attn.proj.bias, factor)
        _grad_scale(self.ln2.weight, factor)
        _grad_scale(self.ln2.bias, factor)
        _grad_scale(self.mlp.up.weight, factor)
        _grad_scale(self.mlp.up.bias, factor)
        _grad_scale(self.mlp.down.weight, factor)
        _grad_scale(self.mlp.down.bias, factor)

    def export_parameters(self, mut out: List[Tensor2D]):
        # Append copies of this block's 12 parameter VALUES in walk order (for
        # checkpoint IO — copies are fine at IO time). Reads self; grows `out`;
        # cannot raise.
        out.append(self.ln1.weight.value.copy())
        out.append(self.ln1.bias.value.copy())
        out.append(self.attn.qkv.weight.value.copy())
        out.append(self.attn.qkv.bias.value.copy())
        out.append(self.attn.proj.weight.value.copy())
        out.append(self.attn.proj.bias.value.copy())
        out.append(self.ln2.weight.value.copy())
        out.append(self.ln2.bias.value.copy())
        out.append(self.mlp.up.weight.value.copy())
        out.append(self.mlp.up.bias.value.copy())
        out.append(self.mlp.down.weight.value.copy())
        out.append(self.mlp.down.bias.value.copy())

    def export_gradients(self, mut out: List[Tensor2D]):
        # Append copies of this block's 12 parameter GRADIENTS in walk order
        # (symmetric to export_parameters; used for per-layer gradient inspection
        # and to drive the walk-consistency check against apply_adamw). Reads
        # self; grows `out`; cannot raise.
        out.append(self.ln1.weight.grad.copy())
        out.append(self.ln1.bias.grad.copy())
        out.append(self.attn.qkv.weight.grad.copy())
        out.append(self.attn.qkv.bias.grad.copy())
        out.append(self.attn.proj.weight.grad.copy())
        out.append(self.attn.proj.bias.grad.copy())
        out.append(self.ln2.weight.grad.copy())
        out.append(self.ln2.bias.grad.copy())
        out.append(self.mlp.up.weight.grad.copy())
        out.append(self.mlp.up.bias.grad.copy())
        out.append(self.mlp.down.weight.grad.copy())
        out.append(self.mlp.down.bias.grad.copy())

    def import_parameters(
        mut self, params: List[Tensor2D], start: Int
    ) raises -> Int:
        # Copy params[start : start+12] into this block's 12 parameter values in
        # walk order, returning the next offset (start + 12). Mutates self's
        # values; allocates nothing; raises on a shape mismatch (via _load_value).
        _load_value(self.ln1.weight, params[start + 0])
        _load_value(self.ln1.bias, params[start + 1])
        _load_value(self.attn.qkv.weight, params[start + 2])
        _load_value(self.attn.qkv.bias, params[start + 3])
        _load_value(self.attn.proj.weight, params[start + 4])
        _load_value(self.attn.proj.bias, params[start + 5])
        _load_value(self.ln2.weight, params[start + 6])
        _load_value(self.ln2.bias, params[start + 7])
        _load_value(self.mlp.up.weight, params[start + 8])
        _load_value(self.mlp.up.bias, params[start + 9])
        _load_value(self.mlp.down.weight, params[start + 10])
        _load_value(self.mlp.down.bias, params[start + 11])
        return start + BLOCK_PARAM_COUNT

    def apply_adamw(
        mut self,
        mut m: List[Tensor2D],
        mut v: List[Tensor2D],
        start: Int,
        t: Int,
        lr: Float64,
        beta1: Float64,
        beta2: Float64,
        eps: Float64,
        weight_decay: Float64,
    ) raises -> Int:
        # One AdamW step on this block's 12 parameters, indexing the caller's m/v
        # state lists at [start : start+12] in walk order, returning the next
        # offset. The four weight matrices receive `weight_decay`; the biases and
        # LayerNorm vectors receive 0.0 (the GPT-family selective-decay partition,
        # the same one parameter_decay_flags reports). Mutates self's values and
        # the m/v entries; allocates nothing; raises on a bad t or a mis-shaped
        # state tensor (via adamw_update).
        adamw_update(
            self.ln1.weight,
            m[start + 0],
            v[start + 0],
            t,
            lr,
            beta1,
            beta2,
            eps,
            0.0,
        )
        adamw_update(
            self.ln1.bias,
            m[start + 1],
            v[start + 1],
            t,
            lr,
            beta1,
            beta2,
            eps,
            0.0,
        )
        adamw_update(
            self.attn.qkv.weight,
            m[start + 2],
            v[start + 2],
            t,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        )
        adamw_update(
            self.attn.qkv.bias,
            m[start + 3],
            v[start + 3],
            t,
            lr,
            beta1,
            beta2,
            eps,
            0.0,
        )
        adamw_update(
            self.attn.proj.weight,
            m[start + 4],
            v[start + 4],
            t,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        )
        adamw_update(
            self.attn.proj.bias,
            m[start + 5],
            v[start + 5],
            t,
            lr,
            beta1,
            beta2,
            eps,
            0.0,
        )
        adamw_update(
            self.ln2.weight,
            m[start + 6],
            v[start + 6],
            t,
            lr,
            beta1,
            beta2,
            eps,
            0.0,
        )
        adamw_update(
            self.ln2.bias,
            m[start + 7],
            v[start + 7],
            t,
            lr,
            beta1,
            beta2,
            eps,
            0.0,
        )
        adamw_update(
            self.mlp.up.weight,
            m[start + 8],
            v[start + 8],
            t,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        )
        adamw_update(
            self.mlp.up.bias,
            m[start + 9],
            v[start + 9],
            t,
            lr,
            beta1,
            beta2,
            eps,
            0.0,
        )
        adamw_update(
            self.mlp.down.weight,
            m[start + 10],
            v[start + 10],
            t,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        )
        adamw_update(
            self.mlp.down.bias,
            m[start + 11],
            v[start + 11],
            t,
            lr,
            beta1,
            beta2,
            eps,
            0.0,
        )
        return start + BLOCK_PARAM_COUNT
