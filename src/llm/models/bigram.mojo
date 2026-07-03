# A bigram language model: one lookup table from current token to next-token
# logits.
#
# The whole model is a single [V, V] table. Row i holds the unnormalized logits
# for the token that follows token i, so a forward pass is just a row lookup
# followed by softmax — no matmul, no hidden state. It is the simplest model
# that can actually be *trained* by gradient descent, which makes it the right
# place to prove the loss, gradient, and optimizer wiring before a real network.
#
# The same struct wears two hats:
#   - a *count* model, filled by from_counts: table[i, j] = log P̂(j | i) from
#     smoothed bigram frequencies. Feeding log-probabilities through softmax
#     reproduces exactly those probabilities, so this is the analytic optimum a
#     trained model should approach.
#   - a *trainable* model, filled with zeros (uniform) or small random logits and
#     then optimized. loss_and_grad computes the cross-entropy gradient in the
#     fused p - onehot form and scatter-adds it into a [V, V] gradient table.
#
# Everything is Float64 for a clean, gradient-checkable reference. Shapes: the
# table and its gradient are [V, V]; a batch supplies B*T (current, next) pairs.

from std.math import log

from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.tensor.ops import (
    softmax_row,
    softmax_row_temperature,
    cross_entropy_one,
)
from llm.data.batch import TokenBatch
from llm.utils.random import Rng


struct BigramLM(Copyable, Movable):
    var vocab_size: Int  # V
    var table: Tensor2D  # [V, V]; row i = logits for the token after token i

    def __init__(out self, vocab_size: Int) raises:
        # A zeros table — every row is uniform, so the untrained model predicts
        # every token with probability 1/V (loss log V on any batch). Raises on a
        # non-positive vocab size.
        if vocab_size <= 0:
            raise Error("BigramLM: vocab_size must be positive")
        self.vocab_size = vocab_size
        self.table = zeros_2d(vocab_size, vocab_size)

    @staticmethod
    def random_init(vocab_size: Int, std: Float64, mut rng: Rng) raises -> Self:
        # A table of small normal logits (mean 0, given std) drawn from `rng`.
        # A nonzero start breaks the symmetry of the zeros table; determinism
        # comes from the generator's state. Mutates rng.
        var model = BigramLM(vocab_size)
        for i in range(vocab_size):
            for j in range(vocab_size):
                model.table[i, j] = rng.normal(0.0, std)
        return model^

    @staticmethod
    def from_counts(
        ids: List[Int], vocab_size: Int, smoothing: Float64
    ) raises -> Self:
        # Build the count model: table[i, j] = log((count(i, j) + smoothing) /
        # (count(i, *) + V * smoothing)). Add-`smoothing` (Laplace) keeps every
        # probability positive so the log is finite even for unseen bigrams; a
        # row for a token that never appears as a source becomes uniform
        # (log(1/V)). Raises unless smoothing > 0 (log of a zero count otherwise)
        # and every id is in [0, V).
        if vocab_size <= 0:
            raise Error("BigramLM.from_counts: vocab_size must be positive")
        if smoothing <= 0.0:
            raise Error("BigramLM.from_counts: smoothing must be positive")

        # Validate every id up front, not only those in a bigram pair — a
        # single-token corpus like [5] with V=2 forms no pairs, so an in-loop
        # check would let the bad id slip through and return a uniform model.
        for k in range(len(ids)):
            if ids[k] < 0 or ids[k] >= vocab_size:
                raise Error("BigramLM.from_counts: token id out of range")

        var counts = zeros_2d(vocab_size, vocab_size)
        for k in range(len(ids) - 1):
            counts[ids[k], ids[k + 1]] = counts[ids[k], ids[k + 1]] + 1.0

        var model = BigramLM(vocab_size)
        for i in range(vocab_size):
            var row_total = 0.0
            for j in range(vocab_size):
                row_total += counts[i, j]
            var denom = row_total + Float64(vocab_size) * smoothing
            for j in range(vocab_size):
                model.table[i, j] = log((counts[i, j] + smoothing) / denom)
        return model^

    def next_logits(self, token_id: Int) raises -> List[Float64]:
        # A copy of row `token_id` — the length-V logits for the next token.
        # Raises if token_id is outside [0, V). Allocates the returned list.
        if token_id < 0 or token_id >= self.vocab_size:
            raise Error("BigramLM.next_logits: token_id out of range")
        var out = List[Float64]()
        for j in range(self.vocab_size):
            out.append(self.table[token_id, j])
        return out^

    def next_probs(
        self, token_id: Int, temperature: Float64
    ) raises -> List[Float64]:
        # The next-token distribution after temperature scaling. Raises via
        # next_logits (bad token) or the softmax (temperature <= 0).
        return softmax_row_temperature(self.next_logits(token_id), temperature)

    def loss_on_batch(self, batch: TokenBatch) raises -> Float64:
        # Mean cross-entropy over all B*T (current, next) pairs. Raises on an
        # out-of-range id (via next_logits) or target (via cross_entropy_one).
        var count = batch.batch_size * batch.seq_len
        var total = 0.0
        for b in range(batch.batch_size):
            for t in range(batch.seq_len):
                var i = batch.input_at(b, t)
                var target = batch.target_at(b, t)
                total += cross_entropy_one(self.next_logits(i), target)
        return total / Float64(count)

    def loss_and_grad(
        self, batch: TokenBatch, mut grad: Tensor2D
    ) raises -> Float64:
        # Zero `grad`, fill it with the cross-entropy gradient, and return the
        # mean loss. Per position (b, t) with current token i and next token
        # `target`, the gradient of the row's cross-entropy wrt its logits is
        # p - onehot(target) where p = softmax(table[i]); this is scatter-added
        # into grad[i, :]. No [B, T, V] logits tensor is ever built — the forward
        # is a row lookup, so materializing one would be pure waste. Everything is
        # averaged by 1/(B*T), matching loss_on_batch. `grad` must be [V, V].
        #
        # This reference favors clarity over speed: it is O(B*T*V) and computes
        # exp twice per position (once in cross_entropy_one's logsumexp, once in
        # softmax_row), re-doing that work for every repeat of the same current
        # token. A batched implementation that reuses each row's softmax arrives
        # with the GPT; here the row lookup keeps even the naive form cheap.
        if grad.rows != self.vocab_size or grad.cols != self.vocab_size:
            raise Error("BigramLM.loss_and_grad: grad shape must be [V, V]")
        grad.fill(0.0)
        var count = batch.batch_size * batch.seq_len
        var total = 0.0
        for b in range(batch.batch_size):
            for t in range(batch.seq_len):
                var i = batch.input_at(b, t)
                var target = batch.target_at(b, t)
                var logits = self.next_logits(i)  # validates i
                total += cross_entropy_one(logits, target)  # validates target
                var p = softmax_row(logits)
                for j in range(self.vocab_size):
                    grad[i, j] = grad[i, j] + p[j]
                grad[i, target] = grad[i, target] - 1.0
        var inv = 1.0 / Float64(count)
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                grad[i, j] = grad[i, j] * inv
        return total * inv
