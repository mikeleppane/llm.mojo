"""TransformerBlock: GPT-2's pre-LN decoder block (self-attention only).

Two sublayers, each a pre-LN residual (GPT-2's layout; the original Transformer
paper is post-LN):

    a   = x + attn(ln1(x), mask)         # masked self-attention sublayer
    out = a + mlp(ln2(a))                # position-wise feed-forward sublayer

Decoder-only: no cross-attention. The residual backward sums both paths,
d_h = d_out + ln.backward(f.backward(d_out)); dropping the skip d_out term is the
classic residual bug. Residual dropout (training path only) is applied to each
sublayer's output before the residual add — the skip path x is never dropped;
attention-weight dropout lives one level down in the self-attention train core.
The plain `forward` is the inference path with no dropout and no rng.
"""

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

# The number of Parameters one block owns, walked in a fixed order (ln1 w/b, attn
# qkv w/b, attn proj w/b, ln2 w/b, mlp up w/b, mlp down w/b). Every walk method
# visits exactly these twelve in this order — the contract the optimizer,
# gradient clipping, and the checkpoint format all lean on.
comptime BLOCK_PARAM_COUNT = 12


@fieldwise_init
struct ParamShape(Copyable, Movable):
    """A single parameter tensor's [rows, cols] shape, produced by the walk.

    A named pair rather than a bare tuple so `.rows`/`.cols` read clearly at
    every use site (sizing optimizer state, validating a checkpoint header).
    """

    var rows: Int
    var cols: Int


def _grad_sum_sq(p: Parameter) -> Float64:
    """Sum of squares of one Parameter's gradient (partial global-norm term).

    Reads p.grad; allocates nothing; cannot raise.

    Args:
        p: Parameter whose gradient is summed.

    Returns:
        The scalar sum of squared gradient entries.
    """
    var s = 0.0
    for i in range(p.grad.rows):
        for j in range(p.grad.cols):
            var g = p.grad[i, j]
            s += g * g
    return s


def _grad_scale(mut p: Parameter, factor: Float64):
    """Multiply one Parameter's gradient in place by `factor` (gradient clipping).

    Mutates p.grad; allocates nothing; cannot raise.

    Args:
        p: Parameter whose gradient is scaled.
        factor: Scale applied to every gradient entry.
    """
    for i in range(p.grad.rows):
        for j in range(p.grad.cols):
            p.grad[i, j] = p.grad[i, j] * factor


def _load_value(mut p: Parameter, src: Tensor2D) raises:
    """Copy `src` into one Parameter's value in place (checkpoint restore).

    Mutates p.value; allocates nothing.

    Args:
        p: Parameter whose value is overwritten.
        src: Source tensor, must match p.value's shape.

    Raises:
        Error: If src's shape does not match the parameter (a corrupt file).
    """
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
    """One sub-cache per stage in forward order, plus the two residual-dropout masks.

    The self-attention stage caches ln1 then the train-core MHA cache (which
    carries attention-weight dropout); the feed-forward stage caches ln2 then the
    MLP. Each residual dropout contributes its mask and scale so backward
    reproduces the exact forward map. The residual adds carry no parameters, so
    the skip path needs no cache. Valid only for the forward call that produced it.
    """

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
    """Bundle forward_cached's output with the cache its backward consumes."""

    var output: Tensor2D  # [T, C]
    var cache: BlockCache

    def take_cache(deinit self) -> BlockCache:
        """Consume this forward and return just the (large) cache.

        The model loop reads the output first, then moves this cache into its
        list instead of deep-copying it.

        Returns:
            The block cache, moved out.
        """
        return self.cache^


@fieldwise_init
struct TransformerBlock(Copyable, Movable):
    """GPT-2 pre-LN decoder block: masked self-attention then a feed-forward MLP.
    """

    var ln1: LayerNorm  # pre-attention norm
    var attn: MultiHeadAttention  # masked self-attention
    var ln2: LayerNorm  # pre-MLP norm
    var mlp: MLP  # position-wise feed-forward

    @staticmethod
    def init_random(
        mut rng: Rng, d_model: Int, n_heads: Int, d_hidden: Int
    ) raises -> TransformerBlock:
        """Build a GPT-2 block: default LayerNorms and normal(0, 0.02) attn + MLP.

        Draw order is attention (qkv then proj) then MLP (up then down); the
        LayerNorms are parameter-free at init and draw nothing, so a given
        generator state reproduces the same block.

        Args:
            rng: Random generator; advanced by the draws.
            d_model: Model width C.
            n_heads: Number of attention heads.
            d_hidden: MLP hidden width.

        Returns:
            A fresh block. Allocates the sublayers.

        Raises:
            Error: On invalid dims (via the sublayer factories).
        """
        var ln1 = LayerNorm.init_default(d_model)
        var attn = MultiHeadAttention.init_random(rng, d_model, n_heads)
        var ln2 = LayerNorm.init_default(d_model)
        var mlp = MLP.init_random(rng, d_model, d_hidden)
        return TransformerBlock(ln1^, attn^, ln2^, mlp^)

    def forward(self, x: Tensor2D, mask: Tensor2D) raises -> Tensor2D:
        """Inference path: pre-LN block with no dropout and no rng.

        Args:
            x: Residual stream, shape [T, C].
            mask: Additive attention mask, shape [T, T].

        Returns:
            The block output [T, C]. Reads self only; allocates.

        Raises:
            Error: On a shape/config mismatch (via the sublayers).
        """
        var attn_out = self.attn.forward(self.ln1.forward(x), mask)  # [T, C]
        var a = add(x, attn_out)  # residual 1: x + attn(ln1(x))
        var mlp_out = self.mlp.forward(self.ln2.forward(a))  # [T, C]
        return add(a, mlp_out)  # residual 2: a + mlp(ln2(a))

    def step(
        self,
        x: Tensor2D,
        mut k_cache: Tensor2D,
        mut v_cache: Tensor2D,
        pos: Int,
    ) raises -> Tensor2D:
        """KV-cached single-token block: the same pre-LN wiring as `forward`, one row wide.

        The self-attention sublayer is the cached `attn.step`, which appends this
        position's K/V to the buffers and attends over the whole valid region:

            a   = x + attn.step(ln1(x), k_cache, v_cache, pos)
            out = a + mlp(ln2(a))

        Args:
            x: Residual stream for the new position, shape [1, C].
            k_cache: This layer's key buffer, mutated at row `pos`.
            v_cache: This layer's value buffer, mutated at row `pos`.
            pos: The newest position (the cache's length on entry).

        Returns:
            The block output [1, C]. Reads self; mutates the cache buffers;
            allocates.

        Raises:
            Error: On a shape/config mismatch or an out-of-range pos.
        """
        var attn_out = self.attn.step(
            self.ln1.forward(x), k_cache, v_cache, pos
        )  # [1, C]
        var a = add(x, attn_out)  # residual 1: x + attn(ln1(x))
        var mlp_out = self.mlp.forward(self.ln2.forward(a))  # [1, C]
        return add(a, mlp_out)  # residual 2: a + mlp(ln2(a))

    def forward_cached(
        self,
        x: Tensor2D,
        mask: Tensor2D,
        p: Float64,
        training: Bool,
        mut rng: Rng,
    ) raises -> BlockForward:
        """Training path: the pre-LN block capturing each stage's cache, with dropout.

        Attention-weight dropout runs inside the self-attention train core;
        residual dropout is applied to each sublayer's output before the residual
        add (the skip is never dropped). rng is threaded attention-core-first,
        then attention residual, then MLP residual — the fixed order a seed
        replays. With training = False (or p = 0) every dropout site is the
        identity with no rng consumed, so this equals `forward`.

        Args:
            x: Residual stream, shape [T, C]; moved into the ln1 cache.
            mask: Additive attention mask, shape [T, T].
            p: Dropout probability.
            training: Whether to apply dropout and draw from rng.
            rng: Random generator; mutated only in the training/p>0 branch.

        Returns:
            The output [T, C] and the cache its backward consumes, valid only for
            this call. Allocates the intermediates and caches.

        Raises:
            Error: On a shape/config mismatch or an out-of-range p.
        """
        # Each stage's forward is split into (output, cache): the output moves to
        # the next stage, the cache into this block's cache — no [T, *] copy.
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
        """Reverse the two pre-LN residuals, outer (MLP) first.

        For out = a + dropout(mlp(ln2(a))) the gradient reaches a by both paths:
        undo the residual dropout, then d_a = d_out + ln2.backward(mlp.backward(
        d_mlp_out)) (skip + branch). Then for a = x + dropout(attn(ln1(x))) the
        same rule reaches x. The bare d_out / d_a skip terms must not be dropped
        (the residual bug); residual dropout is undone on the branch only.

        Args:
            cache: The forward cache from forward_cached.
            d_out: Upstream gradient, shape [T, C].

        Returns:
            Gradient d_x [T, C]. Mutates every sublayer's parameter grads (+=);
            allocates.

        Raises:
            Error: On a shape mismatch (via the sublayers).
        """
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
        """Reset every Parameter's grad to zero (the block's 12 parameters).

        Mutates in place; cannot raise.
        """
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
        """One plain-SGD step on every Parameter, in walk order.

        Delegates to nn.optim.sgd_update. Mutates parameter values in place;
        allocates nothing; cannot raise.

        Args:
            lr: Learning rate.
        """
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
        """Append this block's 12 parameter shapes in walk order.

        Reads self; grows `out`; cannot raise.

        Args:
            out: List the shapes are appended to.
        """
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
        """Append this block's 12 weight-decay flags in walk order.

        The GPT-family partition: the four weight matrices (qkv, proj, mlp up,
        mlp down) decay; the four biases and four LayerNorm vectors do not.
        Grows `out`; cannot raise.

        Args:
            out: List the flags are appended to.
        """
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
        """Sum of squares over this block's 12 gradients (partial global-norm term).

        Reads self; allocates nothing; cannot raise.

        Returns:
            The scalar sum of squared gradient entries.
        """
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
        """Multiply every gradient in this block by `factor` in place (clipping).

        Mutates self's grads; allocates nothing; cannot raise.

        Args:
            factor: Scale applied to every gradient entry.
        """
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
        """Append copies of this block's 12 parameter values in walk order.

        For checkpoint IO. Reads self; grows `out`; cannot raise.

        Args:
            out: List the value copies are appended to.
        """
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
        """Append copies of this block's 12 parameter gradients in walk order.

        Symmetric to export_parameters, for per-layer gradient inspection. Reads
        self; grows `out`; cannot raise.

        Args:
            out: List the gradient copies are appended to.
        """
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
        """Copy params[start : start+12] into this block's values in walk order.

        Mutates self's values; allocates nothing.

        Args:
            params: Full parameter list in walk order.
            start: Offset of this block's first parameter.

        Returns:
            The next offset (start + 12).

        Raises:
            Error: On a shape mismatch (via _load_value).
        """
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
        """One AdamW step on this block's 12 parameters, in walk order.

        Indexes the caller's m/v state at [start : start+12]. The four weight
        matrices receive `weight_decay`; the biases and LayerNorm vectors receive
        0.0 (the GPT-family selective-decay partition). Mutates self's values and
        the m/v entries; allocates nothing.

        Args:
            m: First-moment state list.
            v: Second-moment state list.
            start: Offset of this block's first parameter.
            t: AdamW step counter (1-based).
            lr: Learning rate.
            beta1: First-moment decay.
            beta2: Second-moment decay.
            eps: Numerical epsilon.
            weight_decay: Decay applied to the weight matrices only.

        Returns:
            The next offset (start + 12).

        Raises:
            Error: On a bad t or a mis-shaped state tensor (via adamw_update).
        """
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
