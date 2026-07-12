"""EncDec: the small encoder-decoder Transformer the lab assembles and trains.

Built by assembling already-proven layers and trained through cross-attention
back into an encoder. All pre-LN, GPT-2's layout:

    Encoder:  x = src_tok(src) + src_pos(0..T-1)
              x = enc_block(x, no_mask)   (xN)
              memory = enc_ln_f(x)                      # final LN
    Decoder:  y = tgt_tok(tgt_in) + tgt_pos(0..T-1)
              y = dec_block(y, memory, causal, no_mask) (xN)
              y = dec_ln_f(y)                           # final LN
              logits = head(y)                          # [T, V], untied

Teacher forcing: tgt_in = [BOS] + tgt[:-1], and the loss targets tgt, not the
input. Source and target get separate token and positional embeddings, and the
head is untied. In backward, every decoder block's cross-attention produces a
gradient w.r.t. memory; the encoder backward starts from their sum, which the
+= accumulator keeps correct for more than one decoder block.
"""

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
    """Build the positional ids [0, 1, ..., t-1] a positional Embedding gathers.

    Args:
        t: Sequence length.

    Returns:
        The id list. Allocates.
    """
    var out = List[Int]()
    for i in range(t):
        out.append(i)
    return out^


@fieldwise_init
struct EncDecCache(Copyable, Movable):
    """Everything EncDec.backward needs, mirroring the forward flow.

    The four embedding caches (source token/pos, target token/pos), one cache per
    encoder block, the encoder final-LN cache, one cache per decoder block, the
    decoder final-LN cache, and the head cache. Valid only for the forward call
    that produced it.
    """

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
    """The logits from forward_cached plus the cache its backward consumes."""

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
        """Build a seeded encoder-decoder Transformer.

        Draw order: source token, source pos, target token, target pos
        embeddings; then each encoder block; then each decoder block; then the
        head. The two final LayerNorms are parameter-free at init.

        Args:
            rng: Random generator; its state is advanced.
            vocab_size: Model vocabulary size V.
            d_model: Model width C.
            n_heads: Number of attention heads.
            n_enc: Number of encoder blocks; must be >= 1.
            n_dec: Number of decoder blocks; must be >= 1.
            d_hidden: MLP hidden width.
            t_max: Maximum sequence length for the positional tables.

        Returns:
            A new model. Allocates every layer.

        Raises:
            Error: On non-positive dims (via the sublayer factories) or
                n_enc/n_dec < 1.
        """
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
        """Run the encoder stack and final LN, returning memory.

        Shared by forward and greedy_decode. Exposed so a caller can decode from a
        corrupted (zeroed) memory without re-encoding.

        Args:
            src: Source token ids, length T_src.

        Returns:
            Encoder memory, shape [T_src, C]. Reads self; allocates.

        Raises:
            Error: On a bad id or shape.
        """
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
        """Run the full teacher-forced forward pass.

        Encodes src to memory, then decodes tgt_in under a causal self-mask and a
        no-mask cross-attention over memory.

        Args:
            src: Source token ids, length T_src.
            tgt_in: Teacher-forced decoder inputs, length T_tgt.

        Returns:
            Logits, shape [T_tgt, V]. Reads self; allocates.

        Raises:
            Error: On a bad id, an out-of-range position (T > T_max), or a shape
                mismatch.
        """
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
        """Run the forward pass, capturing every layer's cache for backward.

        Args:
            src: Source token ids, length T_src.
            tgt_in: Teacher-forced decoder inputs, length T_tgt.

        Returns:
            Logits [T_tgt, V] plus a cache valid only for this call. Allocates.

        Raises:
            Error: On the same conditions forward raises.
        """
        var t_src = len(src)
        var t_tgt = len(tgt_in)

        # --- encoder ---
        # src is a small id list, borrowed; the token embedding consumes it. The
        # two embedding outputs are added into x, then each cache moves out (the
        # split's returned output is dropped). Block caches move via take_cache.
        var src_tok_fwd = self.src_tok.forward_cached(src.copy())
        var src_pos_fwd = self.src_pos.forward_cached(position_ids(t_src))
        var x = add(src_tok_fwd.output, src_pos_fwd.output)  # [T_src, C]
        var src_tok_cache = EmbeddingCache(
            List[Int]()
        )  # placeholder, moved into
        _ = src_tok_fwd^.split(src_tok_cache)
        var src_pos_cache = EmbeddingCache(
            List[Int]()
        )  # placeholder, moved into
        _ = src_pos_fwd^.split(src_pos_cache)
        var enc_mask = no_mask(t_src, t_src)
        var enc_caches = List[EncoderBlockCache]()
        for i in range(len(self.encoder)):
            var blk = self.encoder[i].forward_cached(x, enc_mask)
            x = blk.output.copy()  # [T_src, C] feeds the next block
            enc_caches.append(blk^.take_cache())  # move the large block cache
        var enc_ln_f_fwd = self.enc_ln_f.forward_cached(x^)
        var enc_ln_f_cache = LayerNormCache(
            zeros_2d(0, 0), List[Float64](), List[Float64]()
        )  # placeholder, replaced by the move
        var memory = enc_ln_f_fwd^.split(enc_ln_f_cache)  # [T_src, C]

        # --- decoder ---
        var tgt_tok_fwd = self.tgt_tok.forward_cached(tgt_in.copy())
        var tgt_pos_fwd = self.tgt_pos.forward_cached(position_ids(t_tgt))
        var y = add(tgt_tok_fwd.output, tgt_pos_fwd.output)  # [T_tgt, C]
        var tgt_tok_cache = EmbeddingCache(
            List[Int]()
        )  # placeholder, moved into
        _ = tgt_tok_fwd^.split(tgt_tok_cache)
        var tgt_pos_cache = EmbeddingCache(
            List[Int]()
        )  # placeholder, moved into
        _ = tgt_pos_fwd^.split(tgt_pos_cache)
        var self_mask = causal_mask(t_tgt)
        var cross_mask = no_mask(t_tgt, t_src)
        var dec_caches = List[DecoderBlockCache]()
        for i in range(len(self.decoder)):
            # memory is borrowed by every decoder block (each caches its own copy).
            var blk = self.decoder[i].forward_cached(
                y, memory, self_mask, cross_mask
            )
            y = blk.output.copy()  # [T_tgt, C] feeds the next block
            dec_caches.append(blk^.take_cache())  # move the large block cache
        var dec_ln_f_fwd = self.dec_ln_f.forward_cached(y^)
        var dec_ln_f_cache = LayerNormCache(
            zeros_2d(0, 0), List[Float64](), List[Float64]()
        )  # placeholder, replaced by the move
        var dec_ln_f_out = dec_ln_f_fwd^.split(dec_ln_f_cache)  # [T_tgt, C]
        var head_fwd = self.head.forward_cached(dec_ln_f_out^)
        var head_cache = LinearCache(zeros_2d(0, 0))  # placeholder, moved into
        var logits = head_fwd^.split(head_cache)  # [T_tgt, V]

        var cache = EncDecCache(
            src_tok_cache^,
            src_pos_cache^,
            tgt_tok_cache^,
            tgt_pos_cache^,
            enc_caches^,
            enc_ln_f_cache^,
            dec_caches^,
            dec_ln_f_cache^,
            head_cache^,
        )
        return EncDecForward(logits^, cache^)

    def loss(
        self, src: List[Int], tgt_in: List[Int], tgt: List[Int]
    ) raises -> Float64:
        """Compute teacher-forced mean cross-entropy loss.

        Forwards then cross_entropy_rows(logits, tgt). The loss targets tgt, the
        true next tokens, not tgt_in (the shifted inputs).

        Args:
            src: Source token ids, length T_src.
            tgt_in: Teacher-forced decoder inputs, length T_tgt.
            tgt: True target tokens, length T_tgt.

        Returns:
            Mean cross-entropy loss. Reads self; allocates.

        Raises:
            Error: On a shape/id/target-range error.
        """
        var logits = self.forward(src, tgt_in)
        return cross_entropy_rows(logits, tgt)

    def backward(mut self, cache: EncDecCache, d_logits: Tensor2D) raises:
        """Thread d_logits back through the whole model, accumulating every grad.

        Order (reverse of forward): head, dec_ln_f, each decoder block (summing
        each block's d_memory into a running encoder-output gradient), tgt
        embeddings, enc_ln_f (seeded by the summed d_memory), each encoder block,
        src embeddings. The token and positional embeddings were added to form
        each stream, so the same stream gradient feeds both tables. Mutates every
        parameter grad (+=).

        Args:
            cache: The cache from the matching forward_cached call.
            d_logits: Upstream gradient of the logits, shape [T_tgt, V].

        Raises:
            Error: On a shape mismatch.
        """
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
        """Reset every parameter gradient to zero, before each grad-accum batch.
        """
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
        """Apply one plain-SGD step (param -= lr * grad) to every parameter.

        Args:
            lr: Learning rate.

        Raises:
            Error: On a shape mismatch (never for a well-formed model).
        """
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
        """Greedily decode, no KV cache: encode src once, then decode t_out steps.

        Args:
            src: Source token ids, length T_src.
            t_out: Number of decode steps / emitted ids.
            bos: Beginning-of-sequence token id.

        Returns:
            The t_out emitted ids. Reads self; allocates.

        Raises:
            Error: On a shape/id error or an out-of-range position (t_out >
                T_max).
        """
        return self.greedy_decode_from_memory(self.encode(src), t_out, bos)

    def greedy_decode_from_memory(
        self, memory: Tensor2D, t_out: Int, bos: Int
    ) raises -> List[Int]:
        """Run the decode loop against a fixed memory.

        Loops t_out steps feeding [BOS] + tokens-emitted-so-far, takes the argmax
        of the last logits row, appends it, repeats (recomputing the decoder per
        step). Split out from greedy_decode so a caller can pass a corrupted memory
        and watch exact-match collapse.

        Args:
            memory: Encoder output, shape [T_src, C].
            t_out: Number of decode steps / emitted ids.
            bos: Beginning-of-sequence token id.

        Returns:
            The t_out emitted ids. Reads self; allocates.

        Raises:
            Error: On a shape/id error or an out-of-range position (t_out >
                T_max).
        """
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
