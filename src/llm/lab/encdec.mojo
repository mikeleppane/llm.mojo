# EncDec — the small encoder-decoder Transformer the lab assembles and trains.
#
# This is the first model in the repo built by ASSEMBLING already-proven layers,
# and the first to train through cross-attention back into an encoder. The
# architecture (all pre-LN, GPT-2's layout):
#
#   Encoder:  x = src_tok(src) + src_pos(0..T-1)
#             x = enc_block(x, no_mask)   (xN)
#             memory = enc_ln_f(x)                      # final LN, GPT-2's ln_f
#   Decoder:  y = tgt_tok(tgt_in) + tgt_pos(0..T-1)
#             y = dec_block(y, memory, causal, no_mask) (xN)
#             y = dec_ln_f(y)                           # final LN
#             logits = head(y)                          # [T, V], untied
#
# Teacher forcing: tgt_in = [BOS] + tgt[:-1], and the loss is
# cross_entropy_rows(logits, tgt) — the target, NOT the input. Source and target
# get SEPARATE token and positional embeddings, and the head is UNTIED (weight
# tying is a later concept); the enumeration in zero_grad/apply_sgd below is the
# model's full parameter inventory.
#
# The backward's one subtle wire is d_memory. Every decoder block's
# cross-attention produces a gradient with respect to memory; the encoder's
# backward starts from their SUM (and nothing else — the head and the decoder
# self-attention never touch memory). With one decoder block the sum is trivial;
# the += accumulator is what keeps it correct when there is more than one.

from llm.lab.blocks import (
    DecoderBlock,
    DecoderBlockCache,
    EncoderBlock,
    EncoderBlockCache,
)
from llm.lab.params import (
    zero_embedding,
    zero_layernorm,
    zero_linear,
    sgd_embedding,
    sgd_layernorm,
    sgd_linear,
)
from llm.nn.embedding import Embedding, EmbeddingCache
from llm.nn.layernorm import LayerNorm, LayerNormCache
from llm.nn.linear import Linear, LinearCache
from llm.tensor.ops import add, argmax, cross_entropy_rows
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.masks import causal_mask, no_mask
from llm.utils.random import Rng


def position_ids(t: Int) -> List[Int]:
    # The positional ids [0, 1, ..., t-1] a learned positional Embedding gathers.
    # Reads nothing; allocates the result; cannot fail.
    var out = List[Int]()
    for i in range(t):
        out.append(i)
    return out^


@fieldwise_init
struct EncDecCache(Copyable, Movable):
    # Everything EncDec.backward needs, mirroring the forward flow: the four
    # embedding caches (source token/pos, target token/pos), one cache per
    # encoder block, the encoder final-LN cache, one cache per decoder block, the
    # decoder final-LN cache, and the head cache. Valid only for the forward call
    # that produced it.
    var src_tok_cache: EmbeddingCache
    var src_pos_cache: EmbeddingCache
    var tgt_tok_cache: EmbeddingCache
    var tgt_pos_cache: EmbeddingCache
    var enc_caches: List[EncoderBlockCache]
    var enc_ln_f_cache: LayerNormCache
    var dec_caches: List[DecoderBlockCache]
    var dec_ln_f_cache: LayerNormCache
    var head_cache: LinearCache


@fieldwise_init
struct EncDecForward(Copyable, Movable):
    # forward_cached's output (the logits) plus the cache its backward consumes.
    var logits: Tensor2D  # [T, V]
    var cache: EncDecCache


@fieldwise_init
struct EncDec(Copyable, Movable):
    var src_tok: Embedding  # [V, C]  source token table
    var src_pos: Embedding  # [T_max, C]  source positional table
    var tgt_tok: Embedding  # [V, C]  target token table
    var tgt_pos: Embedding  # [T_max, C]  target positional table
    var encoder: List[EncoderBlock]
    var enc_ln_f: LayerNorm  # final encoder LayerNorm (produces memory)
    var decoder: List[DecoderBlock]
    var dec_ln_f: LayerNorm  # final decoder LayerNorm (feeds the head)
    var head: Linear  # C -> V, untied

    @staticmethod
    def init_random(
        mut rng: Rng,
        vocab_size: Int,
        d_model: Int,
        n_heads: Int,
        n_enc: Int,
        n_dec: Int,
        d_hidden: Int,
        t_max: Int,
    ) raises -> EncDec:
        # Build a seeded encoder-decoder. Draw order: source token, source pos,
        # target token, target pos embeddings; then each encoder block (attn then
        # mlp inside); then each decoder block (self-attn, cross-attn, mlp); then
        # the head. The two final LayerNorms are parameter-free at init (weight
        # ones, bias zeros). Mutates rng; allocates every layer; raises on
        # non-positive dims (via the sublayer factories) or n_enc/n_dec < 1.
        if n_enc < 1 or n_dec < 1:
            raise Error(
                "EncDec.init_random: need at least one encoder and one decoder"
                " block, got n_enc="
                + String(n_enc)
                + " n_dec="
                + String(n_dec)
            )
        var src_tok = Embedding.init_random(rng, vocab_size, d_model)
        var src_pos = Embedding.init_random(rng, t_max, d_model)
        var tgt_tok = Embedding.init_random(rng, vocab_size, d_model)
        var tgt_pos = Embedding.init_random(rng, t_max, d_model)
        var encoder = List[EncoderBlock]()
        for _ in range(n_enc):
            encoder.append(
                EncoderBlock.init_random(rng, d_model, n_heads, d_hidden)
            )
        var enc_ln_f = LayerNorm.init_default(d_model)
        var decoder = List[DecoderBlock]()
        for _ in range(n_dec):
            decoder.append(
                DecoderBlock.init_random(rng, d_model, n_heads, d_hidden)
            )
        var dec_ln_f = LayerNorm.init_default(d_model)
        var head = Linear.init_random(rng, d_model, vocab_size)
        return EncDec(
            src_tok^,
            src_pos^,
            tgt_tok^,
            tgt_pos^,
            encoder^,
            enc_ln_f^,
            decoder^,
            dec_ln_f^,
            head^,
        )

    def encode(self, src: List[Int]) raises -> Tensor2D:
        # Run the encoder stack and final LN, returning memory [T_src, C]. Shared
        # by forward and greedy_decode (which encodes once, then decodes many
        # steps). Exposed so the training capstone can ablate by decoding from a
        # corrupted (zeroed) memory without re-encoding. Reads self; allocates;
        # raises on a bad id or shape.
        var t_src = len(src)
        var x = add(
            self.src_tok.forward(src),
            self.src_pos.forward(position_ids(t_src)),
        )  # [T_src, C]
        var enc_mask = no_mask(t_src, t_src)
        for i in range(len(self.encoder)):
            x = self.encoder[i].forward(x, enc_mask)
        return self.enc_ln_f.forward(x)  # memory [T_src, C]

    def forward(self, src: List[Int], tgt_in: List[Int]) raises -> Tensor2D:
        # Full teacher-forced forward: src [T_src], tgt_in [T_tgt] -> logits
        # [T_tgt, V]. Encodes src to memory, then decodes tgt_in under a causal
        # self-mask and a no-mask cross-attention over memory. Reads self;
        # allocates; raises on a bad id, an out-of-range position (T > T_max), or
        # a shape mismatch.
        var memory = self.encode(src)  # [T_src, C]
        var t_src = len(src)
        var t_tgt = len(tgt_in)
        var y = add(
            self.tgt_tok.forward(tgt_in),
            self.tgt_pos.forward(position_ids(t_tgt)),
        )  # [T_tgt, C]
        var self_mask = causal_mask(t_tgt)
        var cross_mask = no_mask(t_tgt, t_src)
        for i in range(len(self.decoder)):
            y = self.decoder[i].forward(y, memory, self_mask, cross_mask)
        y = self.dec_ln_f.forward(y)  # [T_tgt, C]
        return self.head.forward(y)  # [T_tgt, V]

    def forward_cached(
        self, src: List[Int], tgt_in: List[Int]
    ) raises -> EncDecForward:
        # Same computation as forward, capturing every layer's cache for
        # backward. Reads self; allocates the intermediates, caches, and logits;
        # raises on the same conditions forward does. The cache is valid only for
        # this call.
        var t_src = len(src)
        var t_tgt = len(tgt_in)

        # --- encoder ---
        var src_tok_fwd = self.src_tok.forward_cached(src.copy())
        var src_pos_fwd = self.src_pos.forward_cached(position_ids(t_src))
        var x = add(src_tok_fwd.output, src_pos_fwd.output)  # [T_src, C]
        var enc_mask = no_mask(t_src, t_src)
        var enc_caches = List[EncoderBlockCache]()
        for i in range(len(self.encoder)):
            var blk = self.encoder[i].forward_cached(x, enc_mask)
            x = blk.output.copy()
            enc_caches.append(blk.cache.copy())
        var enc_ln_f_fwd = self.enc_ln_f.forward_cached(x^)
        var memory = enc_ln_f_fwd.output.copy()  # [T_src, C]

        # --- decoder ---
        var tgt_tok_fwd = self.tgt_tok.forward_cached(tgt_in.copy())
        var tgt_pos_fwd = self.tgt_pos.forward_cached(position_ids(t_tgt))
        var y = add(tgt_tok_fwd.output, tgt_pos_fwd.output)  # [T_tgt, C]
        var self_mask = causal_mask(t_tgt)
        var cross_mask = no_mask(t_tgt, t_src)
        var dec_caches = List[DecoderBlockCache]()
        for i in range(len(self.decoder)):
            var blk = self.decoder[i].forward_cached(
                y, memory, self_mask, cross_mask
            )
            y = blk.output.copy()
            dec_caches.append(blk.cache.copy())
        var dec_ln_f_fwd = self.dec_ln_f.forward_cached(y^)
        var head_fwd = self.head.forward_cached(dec_ln_f_fwd.output.copy())

        var cache = EncDecCache(
            src_tok_fwd.cache.copy(),
            src_pos_fwd.cache.copy(),
            tgt_tok_fwd.cache.copy(),
            tgt_pos_fwd.cache.copy(),
            enc_caches^,
            enc_ln_f_fwd.cache.copy(),
            dec_caches^,
            dec_ln_f_fwd.cache.copy(),
            head_fwd.cache.copy(),
        )
        return EncDecForward(head_fwd.output.copy(), cache^)

    def loss(
        self, src: List[Int], tgt_in: List[Int], tgt: List[Int]
    ) raises -> Float64:
        # Teacher-forced mean cross-entropy: forward(src, tgt_in) then
        # cross_entropy_rows(logits, tgt). The loss targets tgt (the true next
        # tokens), NOT tgt_in (the shifted inputs) — the off-by-one that would
        # otherwise cap training at the uniform baseline. Reads self; allocates;
        # raises on a shape/id/target-range error.
        var logits = self.forward(src, tgt_in)
        return cross_entropy_rows(logits, tgt)

    def backward(mut self, cache: EncDecCache, d_logits: Tensor2D) raises:
        # Thread d_logits [T_tgt, V] back through the whole model, accumulating
        # every parameter grad (+=). Order (reverse of forward):
        #   head -> dec_ln_f -> each decoder block (reverse), SUMMING each block's
        #   d_memory into a running encoder-output gradient -> tgt embeddings ->
        #   enc_ln_f (seeded by the summed d_memory) -> each encoder block
        #   (reverse) -> src embeddings.
        # The token and positional embeddings were ADDED to form each stream, so
        # the same stream gradient feeds both tables. d_memory is seeded to zeros
        # and each decoder block adds its contribution — with n_dec > 1 every
        # block contributes, and nothing but cross-attention feeds it. Mutates
        # every parameter grad; allocates the intermediate gradients; raises on a
        # shape mismatch.
        var d_y = self.head.backward(cache.head_cache, d_logits)  # [T_tgt, C]
        d_y = self.dec_ln_f.backward(cache.dec_ln_f_cache, d_y)

        # Encoder-output gradient, summed over decoder blocks. Shape [T_src, C]
        # comes from the encoder final-LN's cached input.
        var d_memory = zeros_2d(
            cache.enc_ln_f_cache.x.rows, cache.enc_ln_f_cache.x.cols
        )
        for i in range(len(self.decoder) - 1, -1, -1):
            var grads = self.decoder[i].backward(cache.dec_caches[i], d_y)
            d_y = grads.d_x.copy()
            d_memory = add(d_memory, grads.d_memory)  # SUM, do not overwrite

        # Target embeddings: token and positional both fed by d_y (they summed).
        self.tgt_tok.backward(cache.tgt_tok_cache, d_y)
        self.tgt_pos.backward(cache.tgt_pos_cache, d_y)

        # Encoder side, seeded by the summed memory gradient.
        var d_x = self.enc_ln_f.backward(cache.enc_ln_f_cache, d_memory)
        for i in range(len(self.encoder) - 1, -1, -1):
            d_x = self.encoder[i].backward(cache.enc_caches[i], d_x)
        self.src_tok.backward(cache.src_tok_cache, d_x)
        self.src_pos.backward(cache.src_pos_cache, d_x)

    def zero_grad(mut self):
        # Reset every parameter gradient to zero — the model's full inventory.
        # Called before each grad-accumulation batch. Mutates in place; cannot
        # raise.
        zero_embedding(self.src_tok)
        zero_embedding(self.src_pos)
        zero_embedding(self.tgt_tok)
        zero_embedding(self.tgt_pos)
        for i in range(len(self.encoder)):
            self.encoder[i].zero_grad()
        zero_layernorm(self.enc_ln_f)
        for i in range(len(self.decoder)):
            self.decoder[i].zero_grad()
        zero_layernorm(self.dec_ln_f)
        zero_linear(self.head)

    def apply_sgd(mut self, lr: Float64) raises:
        # One plain-SGD step (param -= lr * grad) on every parameter — the same
        # inventory as zero_grad. Mutates parameter values; raises on a shape
        # mismatch (via sgd_step — never for a well-formed model).
        sgd_embedding(self.src_tok, lr)
        sgd_embedding(self.src_pos, lr)
        sgd_embedding(self.tgt_tok, lr)
        sgd_embedding(self.tgt_pos, lr)
        for i in range(len(self.encoder)):
            self.encoder[i].apply_sgd(lr)
        sgd_layernorm(self.enc_ln_f, lr)
        for i in range(len(self.decoder)):
            self.decoder[i].apply_sgd(lr)
        sgd_layernorm(self.dec_ln_f, lr)
        sgd_linear(self.head, lr)

    def greedy_decode(
        self, src: List[Int], t_out: Int, bos: Int
    ) raises -> List[Int]:
        # Autoregressive greedy decoding, no KV cache: encode src once, then
        # decode t_out steps from that memory. Reads self; allocates; raises on a
        # shape/id error or an out-of-range position (t_out > T_max). Returns the
        # t_out emitted ids.
        return self.greedy_decode_from_memory(self.encode(src), t_out, bos)

    def greedy_decode_from_memory(
        self, memory: Tensor2D, t_out: Int, bos: Int
    ) raises -> List[Int]:
        # The decode loop given a fixed memory: loop t_out steps feeding [BOS] +
        # tokens-emitted-so-far, take the argmax of the LAST logits row, append
        # it, repeat. Recomputes the decoder per step (fine at the lab's tiny T).
        # Split out from greedy_decode so a test can pass a CORRUPTED memory (e.g.
        # zeros) and watch exact-match collapse — the proof the decoder actually
        # reads the encoder. Reads self; allocates; raises on a shape/id error or
        # an out-of-range position (t_out > T_max). Returns the t_out emitted ids.
        var t_src = memory.rows
        var emitted = List[Int]()
        for _ in range(t_out):
            var din = List[Int]()
            din.append(bos)
            for j in range(len(emitted)):
                din.append(emitted[j])
            var t_cur = len(din)
            var y = add(
                self.tgt_tok.forward(din),
                self.tgt_pos.forward(position_ids(t_cur)),
            )
            var self_mask = causal_mask(t_cur)
            var cross_mask = no_mask(t_cur, t_src)
            for i in range(len(self.decoder)):
                y = self.decoder[i].forward(y, memory, self_mask, cross_mask)
            y = self.dec_ln_f.forward(y)
            var logits = self.head.forward(y)  # [t_cur, V]
            var last = List[Float64]()
            for j in range(logits.cols):
                last.append(logits[t_cur - 1, j])
            emitted.append(argmax(last))
        return emitted^
