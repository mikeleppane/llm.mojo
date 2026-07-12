"""GPT: the decoder-only Transformer this project converges on.

    x      = wte(ids) + wpe([0..T-1])    # token + learned positional embeddings
    x      = block_i(x, causal_mask(T))  # L pre-LN blocks, self-attention only
    h      = ln_f(x)                     # final LayerNorm (GPT-2's ln_f)
    logits = h @ wte.table^T             # weight-tied head, no separate matrix

Everything is [T, C]; a batch is the caller's loop over sequences, so there is
no batch tensor. Two things are new; the rest is assembly of proven layers.

1. Weight tying: the LM head has no Parameter of its own — logits = h @
   wte.table^T reuses the token-embedding matrix transposed (GPT-2's head, no
   bias). In backward the token table receives gradient through two paths that
   sum into its one grad: the head matmul at the top and the embedding gather at
   the bottom.

2. Dropout in GPT-2's three places, all driven by cfg.dropout: embedding dropout
   at the model level, attention-weight dropout inside each block's core, and
   residual dropout on each sublayer branch. The cached path is the training
   path; the plain forward is inference and never sees an rng. With
   training = False (or cfg.dropout = 0) forward_cached equals forward exactly.

apply_sgd updates via nn.optim.sgd_update, not training.optimizer: transformer/
sits below training/ in the dependency layering (nn -> transformer -> training),
so it imports the Parameter-level update math downward from nn/.
"""

from std.math import sqrt

from llm.config import GPTConfig
from llm.nn.dropout import dropout_backward, dropout_cached
from llm.nn.embedding import Embedding, EmbeddingCache
from llm.nn.layernorm import LayerNorm, LayerNormCache
from llm.nn.optim import adamw_update, sgd_update
from llm.tensor.ops import (
    add,
    cross_entropy_rows,
    matmul_transpose_a,
    matmul_transpose_b,
)
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.block import (
    _grad_scale,
    _grad_sum_sq,
    _load_value,
    BLOCK_PARAM_COUNT,
    BlockCache,
    ParamShape,
    TransformerBlock,
)
from llm.transformer.kv_cache import KVCache
from llm.transformer.masks import causal_mask
from llm.utils.random import Rng


def position_ids(t: Int) -> List[Int]:
    """Build the positional ids [0, 1, ..., t-1] the learned wpe gathers.

    Position p sits at row p of wpe, so this is the identity range, not a shuffle.

    Args:
        t: Sequence length.

    Returns:
        The list [0..t-1]. Allocates it; cannot fail.
    """
    var out = List[Int]()
    for i in range(t):
        out.append(i)
    return out^


@fieldwise_init
struct GPTCache(Copyable, Movable):
    """Everything GPT.backward needs, mirroring the forward flow.

    The two embedding caches (token, positional), the embedding-dropout mask and
    scale, one cache per block, the final-LN cache, and the head input h (the
    tied head is a matmul, so its backward needs h). Valid only for the forward
    call that produced it.
    """

    var wte_cache: EmbeddingCache
    var wpe_cache: EmbeddingCache
    var emb_drop_mask: Tensor2D  # [T, C], embedding dropout mask
    var emb_drop_inv_keep: Float64
    var block_caches: List[BlockCache]
    var ln_f_cache: LayerNormCache
    var h: Tensor2D  # [T, C], ln_f output = the tied head's input


@fieldwise_init
struct GPTForward(Copyable, Movable):
    """Bundle forward_cached's logits with the cache its backward consumes."""

    var logits: Tensor2D  # [T, V]
    var cache: GPTCache


@fieldwise_init
struct GPT(Copyable, Movable):
    """A decoder-only Transformer: embeddings, L pre-LN blocks, ln_f, tied head.
    """

    var cfg: GPTConfig
    var wte: Embedding  # [V, C]  token embedding — the tied head reuses this table
    var wpe: Embedding  # [context_length, C]  learned positional embedding
    var blocks: List[TransformerBlock]
    var ln_f: LayerNorm  # final LayerNorm, GPT-2's ln_f

    @staticmethod
    def init_random(cfg: GPTConfig, mut rng: Rng) raises -> GPT:
        """Build a seeded GPT from a validated config.

        cfg.validate() runs first so a bad shape fails at the edge. Draw order
        (fixed, so a seed reproduces the model): wte, wpe, then each block in
        layer order (qkv, proj, mlp up, down; LayerNorms draw nothing). MLP hidden
        width is 4C. Following GPT-2's recipe, the residual-feeding projections
        (each block's attention proj and MLP down) are scaled in place by
        1/sqrt(2*n_layers) so the residual stream's variance does not grow with
        depth; scaling after drawing keeps the rng draw stream identical.

        Args:
            cfg: Model config; validated first.
            rng: Random generator; advanced by the draws.

        Returns:
            A fresh model. Allocates every layer.

        Raises:
            Error: On an invalid config or dims.
        """
        cfg.validate()
        var wte = Embedding.init_random(rng, cfg.vocab_size, cfg.d_model)
        var wpe = Embedding.init_random(rng, cfg.context_length, cfg.d_model)
        var d_hidden = 4 * cfg.d_model  # GPT-2's 4x feed-forward ratio
        var blocks = List[TransformerBlock]()
        for _ in range(cfg.n_layers):
            blocks.append(
                TransformerBlock.init_random(
                    rng, cfg.d_model, cfg.n_heads, d_hidden
                )
            )

        # Residual scaling: multiply proj and down weights in place by 1/sqrt(2L).
        var residual_scale = 1.0 / sqrt(2.0 * Float64(cfg.n_layers))
        for i in range(len(blocks)):
            var pw_rows = blocks[i].attn.proj.weight.value.rows
            var pw_cols = blocks[i].attn.proj.weight.value.cols
            for r in range(pw_rows):
                for c in range(pw_cols):
                    blocks[i].attn.proj.weight.value[r, c] = (
                        blocks[i].attn.proj.weight.value[r, c] * residual_scale
                    )
            var dw_rows = blocks[i].mlp.down.weight.value.rows
            var dw_cols = blocks[i].mlp.down.weight.value.cols
            for r in range(dw_rows):
                for c in range(dw_cols):
                    blocks[i].mlp.down.weight.value[r, c] = (
                        blocks[i].mlp.down.weight.value[r, c] * residual_scale
                    )

        var ln_f = LayerNorm.init_default(cfg.d_model)
        return GPT(cfg.copy(), wte^, wpe^, blocks^, ln_f^)

    def _check_length(self, t: Int) raises:
        """Guard the sequence length at the edge with a named error.

        Runs before any embedding gather, so a T of 0 or one past the learned
        positions fails clearly instead of as an opaque range error inside wpe.

        Args:
            t: The sequence length to check.

        Raises:
            Error: Unless 0 < t <= context_length.
        """
        if t <= 0:
            raise Error(
                "GPT.forward: sequence length T must be positive, got "
                + String(t)
            )
        if t > self.cfg.context_length:
            raise Error(
                "GPT.forward: sequence length T="
                + String(t)
                + " exceeds context_length="
                + String(self.cfg.context_length)
            )

    def forward(self, ids: List[Int]) raises -> Tensor2D:
        """Inference path: token ids -> logits, no dropout, no rng.

        Builds the causal mask once and shares it across all blocks; the head is
        the tied matmul h @ wte.table^T.

        Args:
            ids: Token ids, length T.

        Returns:
            The logits [T, V]. Reads self only; allocates.

        Raises:
            Error: On a bad length (not 0 < T <= context_length) or a bad id.
        """
        var t = len(ids)
        self._check_length(t)
        var x = add(
            self.wte.forward(ids),
            self.wpe.forward(position_ids(t)),
        )  # [T, C]
        var mask = causal_mask(t)  # [T, T], built once, shared by every block
        for i in range(len(self.blocks)):
            x = self.blocks[i].forward(x, mask)
        var h = self.ln_f.forward(x)  # [T, C]
        # Tied head: logits = h @ wte.table^T, computed directly — no [C, V]
        # transpose copy of the table (~309 MB at 124M). This is the SAME kernel
        # the cached `step` head uses, so the batch and step tied heads produce
        # identical bits for a shared row: the parity the step-vs-forward test pins.
        return matmul_transpose_b(h, self.wte.table.value)  # [T, V]

    def step(self, token_id: Int, mut cache: KVCache) raises -> Tensor2D:
        """KV-cached single-token forward: feed one new token, reuse the cached K/V.

        `forward` for a length-1 sequence at absolute position `cache.length`,
        so the logits are bit-identical to the last row of forward(ids[0..pos]).
        Checks the cache, embeds the token and its position, threads [1, C]
        through every block's `step` (each appends its K/V at row `pos`), applies
        ln_f and the tied head, then bumps cache.length once. Takes no rng, so the
        eval-path zero-draw invariant is structural. length is advanced only after
        a fully successful pass, so a caller can always trust it.

        Args:
            token_id: The new token to feed.
            cache: KV cache for this model; mutated (buffers and length).

        Returns:
            The logits row [1, V]. Reads self; allocates.

        Raises:
            Error: On an incompatible or full cache, or a bad token id; the cache
                is left untouched in that case.
        """
        cache.check_compatible(self.cfg)
        if cache.length >= cache.capacity:
            raise Error(
                "GPT.step: KV cache full — all "
                + String(cache.capacity)
                + " context_length positions consumed"
            )
        var pos = cache.length

        # Token + positional embedding of the single new position. A bad token id
        # raises here, BEFORE any cache row is written or length is advanced.
        var token_ids: List[Int] = [token_id]
        var pos_ids: List[Int] = [pos]
        var x = add(
            self.wte.forward(token_ids),
            self.wpe.forward(pos_ids),
        )  # [1, C]

        for i in range(len(self.blocks)):
            x = self.blocks[i].step(x, cache.k[i], cache.v[i], pos)  # [1, C]

        var h = self.ln_f.forward(x)  # [1, C]
        # Tied head: logits = h @ wte.table^T, computed without the per-token
        # [C, V] transpose the batch spelling allocates.
        var logits = matmul_transpose_b(h, self.wte.table.value)  # [1, V]

        # Every layer has now written row `pos`; commit the new length once.
        cache.length += 1
        return logits^

    def loss(self, ids: List[Int], targets: List[Int]) raises -> Float64:
        """Mean cross-entropy of the inference-path logits against the targets.

        Args:
            ids: Token ids, length T.
            targets: Next-token id per position, length T.

        Returns:
            The mean cross-entropy. Reads self; allocates.

        Raises:
            Error: On a bad length, id, or target range.
        """
        return cross_entropy_rows(self.forward(ids), targets)

    def forward_cached(
        self, ids: List[Int], training: Bool, mut rng: Rng
    ) raises -> GPTForward:
        """Training path: the forward computation, capturing caches and applying dropout.

        All three dropout sites are driven by cfg.dropout. rng is threaded
        embedding-dropout-first, then through the blocks in order (each block
        draws its attention-core masks, then its two residual masks) — the fixed
        order a seed replays. With training = False (or cfg.dropout = 0) nothing
        draws and the logits equal forward's.

        Args:
            ids: Token ids, length T.
            training: Whether to apply dropout and draw from rng.
            rng: Random generator; mutated only when training and cfg.dropout > 0.

        Returns:
            The logits [T, V] and the cache, valid only for this call. Allocates.

        Raises:
            Error: On a bad length or id.
        """
        var t = len(ids)
        self._check_length(t)
        var p = self.cfg.dropout

        # ids is borrowed; the token embedding consumes it into its cache.
        var wte_fwd = self.wte.forward_cached(ids.copy())
        var wpe_fwd = self.wpe.forward_cached(position_ids(t))
        var emb = add(wte_fwd.output, wpe_fwd.output)  # [T, C]
        # The two embedding outputs are already added into emb, so move each
        # embedding cache out (split's returned output is dropped here).
        var wte_cache = EmbeddingCache(List[Int]())  # placeholder, moved into
        _ = wte_fwd^.split(wte_cache)
        var wpe_cache = EmbeddingCache(List[Int]())  # placeholder, moved into
        _ = wpe_fwd^.split(wpe_cache)

        var emb_drop = dropout_cached(
            emb, p, training, rng
        )  # embedding dropout
        # DropoutResult is a shared temporary; its output field can't be moved
        # out (the mask is needed later), so the residual stream copies it.
        var x = emb_drop.output.copy()  # [T, C]

        var mask = causal_mask(t)  # [T, T], shared by every block
        var block_caches = List[BlockCache]()
        for i in range(len(self.blocks)):
            var blk = self.blocks[i].forward_cached(x, mask, p, training, rng)
            # The block output [T, C] feeds the next block; the block's whole
            # (large) cache moves into the list rather than being deep-copied.
            x = blk.output.copy()
            block_caches.append(blk^.take_cache())

        var ln_f_fwd = self.ln_f.forward_cached(x^)  # output h [T, C]
        var ln_f_cache = LayerNormCache(
            zeros_2d(0, 0), List[Float64](), List[Float64]()
        )  # placeholder, replaced by the move
        var h = ln_f_fwd^.split(ln_f_cache)  # cache -> ln_f_cache, returns h
        # Tied head h @ table^T, direct (no [C, V] transpose copy); same kernel
        # and k-order as forward's head, so forward_cached's logits match.
        var logits = matmul_transpose_b(h, self.wte.table.value)  # [T, V]

        var cache = GPTCache(
            wte_cache^,
            wpe_cache^,
            emb_drop.mask.copy(),  # same: DropoutResult's field can't move out
            emb_drop.inv_keep,
            block_caches^,
            ln_f_cache^,
            h^,
        )
        return GPTForward(logits^, cache^)

    def backward(mut self, cache: GPTCache, d_logits: Tensor2D) raises:
        """Thread d_logits back through the whole model, accumulating every grad (+=).

        No d_input is returned: token ids are not differentiable. Order (reverse
        of forward): the tied head (d_table = d_logits^T @ h, d_h = d_logits @
        table), then ln_f, each block in reverse, embedding dropout, and the two
        gathers off the same d_emb. The tied weight wte receives gradient through
        both the head and the bottom-gather paths; they are summed into one [V, C]
        delta and added to wte.table.grad exactly once, so two backward passes
        double the grad bit-for-bit (float addition is not associative, so a
        separate head += and gather += would not). wpe, reached by a single path,
        uses Embedding.backward directly.

        Args:
            cache: The forward cache from forward_cached.
            d_logits: Upstream gradient, shape [T, V].

        Raises:
            Error: On a shape mismatch.
        """
        var v = self.wte.table.value.rows
        var c = self.wte.table.value.cols

        # Head path into a fresh [V, C] delta; d_h flows on down the stack.
        var d_table = matmul_transpose_a(
            d_logits, cache.h
        )  # d_logits^T @ h [V, C]
        var d_h = d_logits @ self.wte.table.value  # [T, C]

        # ln_f and the block stack.
        var d_x = self.ln_f.backward(cache.ln_f_cache, d_h)  # [T, C]
        for i in range(len(self.blocks) - 1, -1, -1):
            d_x = self.blocks[i].backward(cache.block_caches[i], d_x)

        # Embedding dropout, then the two gathers off the same d_emb.
        var d_emb = dropout_backward(
            cache.emb_drop_mask, cache.emb_drop_inv_keep, d_x
        )  # [T, C]

        # Bottom path (token gather): scatter-add d_emb rows into the SAME d_table
        # by token id, so repeats sum and the head + gather paths combine into one
        # delta before it touches the grad.
        for i in range(len(cache.wte_cache.ids)):
            var idx = cache.wte_cache.ids[i]
            for col in range(c):
                d_table[idx, col] = d_table[idx, col] + d_emb[i, col]
        # Add the combined tied-weight delta to wte.table.grad ONCE (exact doubling).
        for row in range(v):
            for col in range(c):
                self.wte.table.grad[row, col] = (
                    self.wte.table.grad[row, col] + d_table[row, col]
                )

        # Positional gather: single path, the proven Embedding.backward.
        self.wpe.backward(cache.wpe_cache, d_emb)

    def zero_grad(mut self):
        """Reset every parameter gradient to zero (wte walked once, being tied).

        Mutates in place; cannot raise.
        """
        self.wte.table.zero_grad()
        self.wpe.table.zero_grad()
        for i in range(len(self.blocks)):
            self.blocks[i].zero_grad()
        self.ln_f.weight.zero_grad()
        self.ln_f.bias.zero_grad()

    def apply_sgd(mut self, lr: Float64):
        """One plain-SGD step (p -= lr*grad) on every parameter, wte updated once.

        Delegates to nn.optim.sgd_update. Mutates parameter values in place;
        allocates nothing; cannot raise.

        Args:
            lr: Learning rate.
        """
        sgd_update(self.wte.table, lr)
        sgd_update(self.wpe.table, lr)
        for i in range(len(self.blocks)):
            self.blocks[i].apply_sgd(lr)
        sgd_update(self.ln_f.weight, lr)
        sgd_update(self.ln_f.bias, lr)

    def parameter_count_actual(self) -> Int:
        """Sum value.size() over every Parameter: the actual float count allocated.

        wte is summed once (the tied head owns no Parameter), so this reconciles
        with cfg.parameter_count(), which also counts the head as 0. Reads self;
        allocates nothing; cannot raise.

        Returns:
            The total number of parameter floats.
        """
        var total = 0
        total += self.wte.table.value.size()  # V*C, counted once (tied head)
        total += self.wpe.table.value.size()  # context_length * C
        for i in range(len(self.blocks)):
            total += self.blocks[i].ln1.weight.value.size()
            total += self.blocks[i].ln1.bias.value.size()
            total += self.blocks[i].attn.qkv.weight.value.size()
            total += self.blocks[i].attn.qkv.bias.value.size()
            total += self.blocks[i].attn.proj.weight.value.size()
            total += self.blocks[i].attn.proj.bias.value.size()
            total += self.blocks[i].ln2.weight.value.size()
            total += self.blocks[i].ln2.bias.value.size()
            total += self.blocks[i].mlp.up.weight.value.size()
            total += self.blocks[i].mlp.up.bias.value.size()
            total += self.blocks[i].mlp.down.weight.value.size()
            total += self.blocks[i].mlp.down.bias.value.size()
        total += self.ln_f.weight.value.size()
        total += self.ln_f.bias.value.size()
        return total

    # --- The parameter walk as a registry --------------------------------------
    #
    # The model has no framework parameter dict; one documented traversal order IS
    # the registry, and every method below consumes it. The order is: wte (once —
    # the tied head owns no Parameter), wpe, each block's 12 parameters in layer
    # order, then ln_f w/b. Optimizer state (m, v) is trainer-owned parallel lists
    # sized from parameter_shapes and indexed in this same order. Drift between the
    # methods is what the optimizer and checkpoint tests catch.

    def parameter_tensor_count(self) -> Int:
        """Count the Parameter tensors the walk visits (not the float count).

        wte, wpe, 12 per block, ln_f weight and bias — sizes the m/v state lists
        and the checkpoint. Reads self; allocates nothing; cannot raise.

        Returns:
            The number of Parameter tensors.
        """
        return 2 + BLOCK_PARAM_COUNT * len(self.blocks) + 2

    def parameter_shapes(self) -> List[ParamShape]:
        """The (rows, cols) of every Parameter in walk order (wte once).

        Used to allocate zeros m/v state and to validate a checkpoint header.
        Reads self; allocates the returned list; cannot raise.

        Returns:
            The parameter shapes in walk order.
        """
        var out = List[ParamShape]()
        out.append(
            ParamShape(self.wte.table.value.rows, self.wte.table.value.cols)
        )
        out.append(
            ParamShape(self.wpe.table.value.rows, self.wpe.table.value.cols)
        )
        for i in range(len(self.blocks)):
            self.blocks[i].parameter_shapes(out)
        out.append(
            ParamShape(self.ln_f.weight.value.rows, self.ln_f.weight.value.cols)
        )
        out.append(
            ParamShape(self.ln_f.bias.value.rows, self.ln_f.bias.value.cols)
        )
        return out^

    def parameter_decay_flags(self) -> List[Bool]:
        """The weight-decay flag of every Parameter in walk order (GPT-family partition).

        The embedding matrices wte and wpe and every Linear weight decay; every
        bias and LayerNorm weight/bias do not. This is the partition apply_adamw
        applies. Reads self; allocates the returned list; cannot raise.

        Returns:
            The decay flags in walk order.
        """
        var out = List[Bool]()
        out.append(True)  # wte (token embedding matrix)
        out.append(True)  # wpe (positional embedding matrix)
        for i in range(len(self.blocks)):
            self.blocks[i].parameter_decay_flags(out)
        out.append(False)  # ln_f.weight (LayerNorm vector)
        out.append(False)  # ln_f.bias   (LayerNorm vector)
        return out^

    def grad_norm(self) -> Float64:
        """The global L2 norm of the whole-model gradient.

        sqrt of the sum of squares over every gradient entry (wte once, wpe, all
        blocks, ln_f) — the single vector norm gradient clipping thresholds
        against, not a per-tensor norm. Reads self; allocates nothing; cannot
        raise.

        Returns:
            The global gradient L2 norm.
        """
        var s = 0.0
        s += _grad_sum_sq(self.wte.table)
        s += _grad_sum_sq(self.wpe.table)
        for i in range(len(self.blocks)):
            s += self.blocks[i].grad_norm_sq()
        s += _grad_sum_sq(self.ln_f.weight)
        s += _grad_sum_sq(self.ln_f.bias)
        return sqrt(s)

    def scale_grads(mut self, factor: Float64):
        """Multiply every gradient in the model by `factor` in place (wte once).

        The second half of gradient clipping (grad *= clip / norm). Mutates every
        grad; allocates nothing; cannot raise.

        Args:
            factor: Scale applied to every gradient entry.
        """
        _grad_scale(self.wte.table, factor)
        _grad_scale(self.wpe.table, factor)
        for i in range(len(self.blocks)):
            self.blocks[i].scale_grads(factor)
        _grad_scale(self.ln_f.weight, factor)
        _grad_scale(self.ln_f.bias, factor)

    def export_parameters(self) -> List[Tensor2D]:
        """A copy of every Parameter's value in walk order (wte once), for checkpoint save.

        Reads self; allocates the returned list; cannot raise.

        Returns:
            The value copies in walk order.
        """
        var out = List[Tensor2D]()
        out.append(self.wte.table.value.copy())
        out.append(self.wpe.table.value.copy())
        for i in range(len(self.blocks)):
            self.blocks[i].export_parameters(out)
        out.append(self.ln_f.weight.value.copy())
        out.append(self.ln_f.bias.value.copy())
        return out^

    def export_gradients(self) -> List[Tensor2D]:
        """A copy of every Parameter's gradient in walk order (symmetric to export_parameters).

        wte appears once — its grad is the summed two-path tied-weight gradient.
        Used for per-layer gradient inspection. Reads self; allocates the returned
        list; cannot raise.

        Returns:
            The gradient copies in walk order.
        """
        var out = List[Tensor2D]()
        out.append(self.wte.table.grad.copy())
        out.append(self.wpe.table.grad.copy())
        for i in range(len(self.blocks)):
            self.blocks[i].export_gradients(out)
        out.append(self.ln_f.weight.grad.copy())
        out.append(self.ln_f.bias.grad.copy())
        return out^

    def import_parameters(mut self, params: List[Tensor2D]) raises:
        """Copy `params` (in walk order) into every Parameter's value in place.

        For checkpoint restore. Mutates every value; allocates nothing.

        Args:
            params: Parameter values in walk order.

        Raises:
            Error: If the count or any shape does not match the live model.
        """
        var expected = self.parameter_tensor_count()
        if len(params) != expected:
            raise Error(
                "import_parameters: expected "
                + String(expected)
                + " tensors, got "
                + String(len(params))
            )
        _load_value(self.wte.table, params[0])
        _load_value(self.wpe.table, params[1])
        var off = 2
        for i in range(len(self.blocks)):
            off = self.blocks[i].import_parameters(params, off)
        _load_value(self.ln_f.weight, params[off])
        _load_value(self.ln_f.bias, params[off + 1])

    def apply_adamw(
        mut self,
        mut m: List[Tensor2D],
        mut v: List[Tensor2D],
        t: Int,
        lr: Float64,
        beta1: Float64,
        beta2: Float64,
        eps: Float64,
        weight_decay: Float64,
    ) raises:
        """One AdamW step on every parameter, indexing the trainer-owned m/v in walk order.

        wte and wpe (matrices) decay; ln_f (LayerNorm) does not; each block
        applies the selective-decay partition to its own twelve. m and v must
        each have parameter_tensor_count() entries shaped like their parameter.
        Mutates every value and every m/v entry; allocates nothing.

        Args:
            m: First-moment state list.
            v: Second-moment state list.
            t: AdamW step counter (1-based).
            lr: Learning rate.
            beta1: First-moment decay.
            beta2: Second-moment decay.
            eps: Numerical epsilon.
            weight_decay: Decay applied to the decaying parameters.

        Raises:
            Error: On a length mismatch, a bad t, or a mis-shaped state tensor.
        """
        var expected = self.parameter_tensor_count()
        if len(m) != expected or len(v) != expected:
            raise Error(
                "apply_adamw: m/v must have "
                + String(expected)
                + " entries (one per parameter), got len(m)="
                + String(len(m))
                + ", len(v)="
                + String(len(v))
            )
        # wte and wpe: embedding matrices, decayed.
        adamw_update(
            self.wte.table, m[0], v[0], t, lr, beta1, beta2, eps, weight_decay
        )
        adamw_update(
            self.wpe.table, m[1], v[1], t, lr, beta1, beta2, eps, weight_decay
        )
        var off = 2
        for i in range(len(self.blocks)):
            off = self.blocks[i].apply_adamw(
                m, v, off, t, lr, beta1, beta2, eps, weight_decay
            )
        # ln_f: LayerNorm vectors, never decayed.
        adamw_update(
            self.ln_f.weight, m[off], v[off], t, lr, beta1, beta2, eps, 0.0
        )
        adamw_update(
            self.ln_f.bias,
            m[off + 1],
            v[off + 1],
            t,
            lr,
            beta1,
            beta2,
            eps,
            0.0,
        )
