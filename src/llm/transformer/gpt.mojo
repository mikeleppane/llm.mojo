# GPT — the decoder-only Transformer this whole project has been converging on.
#
#   x      = wte(ids) + wpe([0..T-1])        # token + learned positional embeddings
#   x      = block_i(x, causal_mask(T))      # L pre-LN blocks, self-attention only
#   h      = ln_f(x)                         # final LayerNorm (GPT-2's ln_f)
#   logits = h @ wte.table^T                 # WEIGHT-TIED head — no separate matrix
#
# Everything is [T, C]; a batch is the caller's loop over sequences (loop
# discipline, matching every other layer here), so there is no batch tensor.
#
# Two things are genuinely new here; the rest is assembly of proven layers.
#
# 1. Weight tying. The language-model head has NO Parameter of its own: the
#    logits are h @ wte.table^T, reusing the token-embedding matrix transposed
#    (GPT-2's head, which also has no bias). In backward the token table receives
#    gradient through TWO paths that SUM into its one Parameter.grad: the head
#    matmul at the top (d_table += d_logits^T @ h) and the embedding gather at
#    the bottom (scatter-add of the embedding gradient). This is exactly what the
#    += accumulation contract every layer's backward follows was built for; a
#    model-level finite-difference of the table grad pins that BOTH paths are
#    present (a missing path, or `=` for `+=`, is off by a whole term).
#
# 2. Dropout in GPT-2's three places, all driven by the single cfg.dropout:
#    embedding dropout on wte(ids)+wpe(pos) here at the model level; attention-
#    weight dropout inside each block's self-attention core; residual dropout on
#    each sublayer branch inside each block. The cached path IS the training path
#    (dropout lives only there); the plain forward is the inference path and
#    never sees an rng — applying dropout at inference is unrepresentable. With
#    training = False (or cfg.dropout = 0) every site is the identity with an
#    all-ones mask and NO rng consumed, so forward_cached equals forward exactly.
#
# GPT.apply_sgd performs the in-place p -= lr*grad update via nn.optim.sgd_update
# rather than training.optimizer.sgd_step: transformer/ sits BELOW training/ in
# the dependency layering (nn -> transformer -> training), so it must not import
# upward; nn/ owns the Parameter-level update math and transformer/ may import it.

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
    # The positional ids [0, 1, ..., t-1] the learned positional Embedding gathers.
    # Position p sits at row p of wpe, so this is the identity range, NOT a shuffle.
    # Reads nothing; allocates the result; cannot fail.
    var out = List[Int]()
    for i in range(t):
        out.append(i)
    return out^


@fieldwise_init
struct GPTCache(Copyable, Movable):
    # Everything GPT.backward needs, mirroring the forward flow: the two embedding
    # caches (token, positional), the embedding-dropout mask and scale, one cache
    # per block, the final-LN cache, and the head input h. The head is a matmul,
    # so its backward needs its input h just as a LinearCache holds x. Valid only
    # for the forward call that produced it.
    var wte_cache: EmbeddingCache
    var wpe_cache: EmbeddingCache
    var emb_drop_mask: Tensor2D  # [T, C], embedding dropout mask
    var emb_drop_inv_keep: Float64
    var block_caches: List[BlockCache]
    var ln_f_cache: LayerNormCache
    var h: Tensor2D  # [T, C], ln_f output = the tied head's input


@fieldwise_init
struct GPTForward(Copyable, Movable):
    # forward_cached's output (the logits) plus the cache its backward consumes.
    var logits: Tensor2D  # [T, V]
    var cache: GPTCache


@fieldwise_init
struct GPT(Copyable, Movable):
    var cfg: GPTConfig
    var wte: Embedding  # [V, C]  token embedding — the tied head reuses this table
    var wpe: Embedding  # [context_length, C]  learned positional embedding
    var blocks: List[TransformerBlock]
    var ln_f: LayerNorm  # final LayerNorm, GPT-2's ln_f

    @staticmethod
    def init_random(cfg: GPTConfig, mut rng: Rng) raises -> GPT:
        # Build a seeded GPT from a validated config. cfg.validate() runs FIRST so
        # a bad shape fails at the edge. Draw order (fixed and documented, so a
        # seed reproduces the model): wte, wpe, then each block in layer order
        # (inside a block: qkv, proj, then mlp up, down — the LayerNorms are
        # parameter-free at init and draw nothing). The MLP hidden width is 4C,
        # GPT-2's ratio.
        #
        # Residual init scaling (GPT-2's published recipe): the weights of the
        # residual-FEEDING projections — each block's attention proj and MLP down —
        # are scaled in place by 1/sqrt(N), N = 2*n_layers = the number of residual
        # additions, so the residual stream's variance does not grow with depth
        # (std 0.02 -> 0.02/sqrt(2L), ~0.00408 at L=12). qkv and mlp up stay at
        # 0.02. Scaling AFTER drawing keeps the rng draw stream identical to the
        # unscaled layout and every layer factory signature untouched. Mutates rng;
        # allocates every layer; raises on an invalid config or dims.
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
        # Guard the sequence length at the edge with a NAMED error, before any
        # embedding gather runs — a T of 0 or one past the learned positions would
        # otherwise surface as an opaque range error deep inside wpe. Reads self;
        # allocates nothing.
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
        # Inference path: token ids [T] -> logits [T, V], no dropout, no rng.
        # Raises (named) unless 0 < T <= context_length. Builds the causal mask
        # ONCE and shares it across all blocks. The head is the tied matmul
        # h @ wte.table^T. Reads self only; allocates the intermediates and logits;
        # raises on a bad length or a bad id (via the embeddings).
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
        # KV-cached single-token forward: feed ONE new token, reusing the cached
        # K/V of every earlier position, and return its logits row [1, V]. This is
        # `forward` for a sequence of length 1 sitting at absolute position
        # `cache.length` — same embeddings, same blocks, same tied head, so the
        # logits are BIT-IDENTICAL to the last row of forward(ids[0..pos]) run on
        # the full prefix. It reuses the frozen primitives in the same order; it
        # re-derives no math.
        #
        # Flow:
        #   1. check the cache matches this model; raise NAMED if it is already
        #      full (all context_length positions consumed).
        #   2. pos = cache.length; embed = wte[token_id] + wpe[pos], each gathered
        #      as a one-element id list -> [1, C]. (The batch path adds the same
        #      two rows for position pos.)
        #   3. thread [1, C] through every block's `step`, passing that layer's
        #      k/v cache buffers; each block appends its K/V at row `pos`.
        #   4. h = ln_f(x); logits = h @ wte.table^T via matmul_transpose_b — the
        #      tied head without materializing the [C, V] transpose per token.
        #   5. bump cache.length ONCE, after every layer wrote row `pos`.
        # Takes NO rng and consumes zero draws: there is no rng parameter to
        # misuse, so the eval-path zero-draw invariant is structural, not merely
        # tested. Reads self; mutates the cache (its buffers and length); allocates
        # the intermediates and logits; raises (named) on an incompatible or full
        # cache, or a bad token id (via the embedding), in which case the cache is
        # left untouched — length is only advanced after a fully successful pass,
        # so a caller can always trust it.
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
        # Mean cross-entropy of the inference-path logits against the targets:
        # cross_entropy_rows(forward(ids), targets). targets has length T (the
        # next-token id per position). Reads self; allocates; raises on a bad
        # length, id, or target range (surfaced by forward / cross_entropy_rows).
        return cross_entropy_rows(self.forward(ids), targets)

    def forward_cached(
        self, ids: List[Int], training: Bool, mut rng: Rng
    ) raises -> GPTForward:
        # Training path: the same computation as forward, capturing every layer's
        # cache and applying GPT-2's dropout (all three sites driven by
        # cfg.dropout). rng is threaded embedding-dropout-first, then through the
        # blocks in order (each block draws its attention-core masks, then its two
        # residual masks) — the fixed order a seed replays. With training = False
        # (or cfg.dropout = 0) nothing draws and the logits equal forward's.
        # Reads self; allocates the intermediates, caches, and logits; mutates rng
        # only when training and cfg.dropout > 0; raises on a bad length or id. The
        # cache is valid only for this call.
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
        # Thread d_logits [T, V] back through the whole model, accumulating every
        # parameter grad (+=). No d_input is returned: the model's inputs are
        # integer token ids, which are not differentiable (same rule as
        # Embedding.backward). Order (reverse of forward):
        #
        #   Tied head (a bias-free matmul logits = h @ table^T):
        #     d_table (head path)  = d_logits^T @ h      [V, T] @ [T, C] -> [V, C]
        #     d_h = d_logits @ table                     [T, V] @ [V, C] -> [T, C]
        #   ln_f -> each block (reverse) -> embedding dropout ->
        #     the token gather (bottom path) and the positional gather, both fed
        #     the same embedding gradient d_emb (token + position were ADDED to
        #     form the stream).
        #
        # The tied weight wte receives gradient through BOTH paths. They are summed
        # into ONE [V, C] delta (d_table = head delta, then the bottom path's
        # scatter-add is folded into that same tensor) which is added to
        # wte.table.grad exactly ONCE. Adding one fully-formed delta per call — not
        # a head `+=` and a separate gather `+=` — is what makes two backward passes
        # double the grad bit-for-bit: two separately-rounded additions per call
        # would accumulate as ((h+g)+h)+g, which is not bit-identical to 2*(h+g)
        # (float addition is not associative). This is Part XI's accumulation rule
        # applied to the tied weight, the reason the doubling test exists. The
        # scatter mirrors Embedding.backward (repeated ids sum); wpe, reached by a
        # single path, uses Embedding.backward directly.
        #
        # Mutates every parameter grad; allocates the intermediate gradients;
        # raises on a shape mismatch.
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
        # Reset every parameter gradient to zero — the model's full inventory.
        # wte appears ONCE (tying means one Parameter, walked once). Mutates in
        # place; cannot raise.
        self.wte.table.zero_grad()
        self.wpe.table.zero_grad()
        for i in range(len(self.blocks)):
            self.blocks[i].zero_grad()
        self.ln_f.weight.zero_grad()
        self.ln_f.bias.zero_grad()

    def apply_sgd(mut self, lr: Float64):
        # One plain-SGD step (p -= lr*grad) on every parameter — the same
        # inventory zero_grad walks, in the same order, wte updated ONCE.
        # Delegates to nn.optim.sgd_update. Mutates parameter values in place;
        # allocates nothing; cannot raise.
        sgd_update(self.wte.table, lr)
        sgd_update(self.wpe.table, lr)
        for i in range(len(self.blocks)):
            self.blocks[i].apply_sgd(lr)
        sgd_update(self.ln_f.weight, lr)
        sgd_update(self.ln_f.bias, lr)

    def parameter_count_actual(self) -> Int:
        # Walk every Parameter and sum value.size() — the count of ACTUAL floats
        # the model allocates. wte is summed ONCE (the tied head owns no
        # Parameter), so this must reconcile with cfg.parameter_count(), which also
        # counts the head as 0. Reads self; allocates nothing; cannot raise.
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
    # The model has no framework parameter dict; instead ONE documented traversal
    # order IS the registry, and these methods all consume it. The order is: wte
    # (once — the tied head owns no Parameter), wpe, then each block's 12
    # parameters in layer order (ln1 w/b, attn qkv w/b, attn proj w/b, ln2 w/b,
    # mlp up w/b, mlp down w/b), then ln_f w/b. Optimizer state (m, v) is
    # trainer-owned: plain parallel List[Tensor2D] sized from parameter_shapes and
    # indexed in this same order. Every method below — parameter_shapes,
    # parameter_decay_flags, grad_norm, scale_grads, export/import_parameters,
    # apply_adamw — visits exactly these parameters in exactly this order; drift
    # between them is the named failure mode the optimizer and checkpoint tests
    # exist to catch.

    def parameter_tensor_count(self) -> Int:
        # How many Parameter TENSORS the walk visits (NOT the float count): wte,
        # wpe, 12 per block, ln_f weight and bias. This sizes the m/v state lists
        # and the checkpoint. Reads self; allocates nothing; cannot raise.
        return 2 + BLOCK_PARAM_COUNT * len(self.blocks) + 2

    def parameter_shapes(self) -> List[ParamShape]:
        # The (rows, cols) of every Parameter, in walk order — used to allocate
        # zeros m/v state and to validate a checkpoint header against the live
        # model. wte appears ONCE. Reads self; allocates the returned list; cannot
        # raise.
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
        # The weight-decay flag of every Parameter, in walk order (the GPT-family
        # partition): the embedding matrices wte and wpe and every Linear weight
        # decay; every bias and every LayerNorm weight/bias (ln1, ln2, ln_f) do
        # not. Reads self; allocates the returned list; cannot raise. This is the
        # partition apply_adamw applies; a test pins the two agree.
        var out = List[Bool]()
        out.append(True)  # wte (token embedding matrix)
        out.append(True)  # wpe (positional embedding matrix)
        for i in range(len(self.blocks)):
            self.blocks[i].parameter_decay_flags(out)
        out.append(False)  # ln_f.weight (LayerNorm vector)
        out.append(False)  # ln_f.bias   (LayerNorm vector)
        return out^

    def grad_norm(self) -> Float64:
        # The GLOBAL L2 norm of the whole-model gradient: sqrt of the sum of
        # squares over EVERY gradient entry (wte once, wpe, all blocks, ln_f) —
        # the single vector norm gradient clipping thresholds against, NOT a
        # per-tensor norm. Reads self; allocates nothing; cannot raise.
        var s = 0.0
        s += _grad_sum_sq(self.wte.table)
        s += _grad_sum_sq(self.wpe.table)
        for i in range(len(self.blocks)):
            s += self.blocks[i].grad_norm_sq()
        s += _grad_sum_sq(self.ln_f.weight)
        s += _grad_sum_sq(self.ln_f.bias)
        return sqrt(s)

    def scale_grads(mut self, factor: Float64):
        # Multiply EVERY gradient in the model by `factor` in place — the second
        # half of gradient clipping (grad *= clip / norm). wte scaled ONCE.
        # Mutates every grad; allocates nothing; cannot raise.
        _grad_scale(self.wte.table, factor)
        _grad_scale(self.wpe.table, factor)
        for i in range(len(self.blocks)):
            self.blocks[i].scale_grads(factor)
        _grad_scale(self.ln_f.weight, factor)
        _grad_scale(self.ln_f.bias, factor)

    def export_parameters(self) -> List[Tensor2D]:
        # A copy of every Parameter's VALUE, in walk order (for checkpoint save;
        # copies are fine at IO time). wte appears ONCE. Reads self; allocates the
        # returned list; cannot raise.
        var out = List[Tensor2D]()
        out.append(self.wte.table.value.copy())
        out.append(self.wpe.table.value.copy())
        for i in range(len(self.blocks)):
            self.blocks[i].export_parameters(out)
        out.append(self.ln_f.weight.value.copy())
        out.append(self.ln_f.bias.value.copy())
        return out^

    def export_gradients(self) -> List[Tensor2D]:
        # A copy of every Parameter's GRADIENT, in walk order (symmetric to
        # export_parameters). wte appears ONCE — its grad is the summed two-path
        # tied-weight gradient. Reads self; allocates the returned list; cannot
        # raise. Used for per-layer gradient inspection and to drive the
        # walk-consistency check against apply_adamw.
        var out = List[Tensor2D]()
        out.append(self.wte.table.grad.copy())
        out.append(self.wpe.table.grad.copy())
        for i in range(len(self.blocks)):
            self.blocks[i].export_gradients(out)
        out.append(self.ln_f.weight.grad.copy())
        out.append(self.ln_f.bias.grad.copy())
        return out^

    def import_parameters(mut self, params: List[Tensor2D]) raises:
        # Copy `params` (in walk order) into every Parameter's value in place (for
        # checkpoint restore). Raises if the count or any shape does not match the
        # live model — a checkpoint for a different architecture must fail loudly,
        # never load garbage. Mutates every value; allocates nothing.
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
        # One AdamW step on EVERY parameter, indexing the trainer-owned m/v state
        # lists in walk order. wte and wpe (matrices) decay; ln_f (LayerNorm) does
        # not; each block applies the selective-decay partition to its own twelve.
        # m and v must have exactly parameter_tensor_count() entries, each shaped
        # like its parameter (allocate them from parameter_shapes). Mutates every
        # value and every m/v entry; allocates nothing; raises on a length
        # mismatch, a bad t, or a mis-shaped state tensor.
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
