"""A bigram language model: one lookup table from token to next-token logits.

The whole model is a single [V, V] table; row i holds the unnormalized logits
for the token that follows token i, so a forward pass is a row lookup followed
by softmax — the simplest model trainable by gradient descent, ideal for
proving the loss, gradient, and optimizer wiring before a real network. The
struct wears two hats: a count model (from_counts fills row i with log P(j | i)
from smoothed bigram frequencies, the analytic optimum) and a trainable model
(zeros or small random logits, then optimized via loss_and_grad, which
scatter-adds the fused p - onehot cross-entropy gradient into a [V, V] table).
Everything is Float64 for a clean, gradient-checkable reference.
"""

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
    """A bigram model: a [V, V] table of next-token logits per current token."""

    var vocab_size: Int  # V
    var table: Tensor2D  # [V, V]; row i = logits for the token after token i

    def __init__(out self, vocab_size: Int) raises:
        """Initialize a zeros table: every row uniform.

        The untrained model predicts every token with probability 1/V (loss
        log V on any batch).

        Args:
            vocab_size: V, must be positive.

        Raises:
            Error: If vocab_size is not positive.
        """
        if vocab_size <= 0:
            raise Error("BigramLM: vocab_size must be positive")
        self.vocab_size = vocab_size
        self.table = zeros_2d(vocab_size, vocab_size)

    @staticmethod
    def random_init(vocab_size: Int, std: Float64, mut rng: Rng) raises -> Self:
        """Build a table of small normal logits (mean 0, given std) from `rng`.

        A nonzero start breaks the symmetry of the zeros table; determinism
        comes from the generator's state. Mutates rng.

        Args:
            vocab_size: V.
            std: Standard deviation of the initial logits.
            rng: Random generator (mutated).

        Returns:
            The initialized model. Allocates.

        Raises:
            Error: If vocab_size is not positive.
        """
        var model = BigramLM(vocab_size)
        for i in range(vocab_size):
            for j in range(vocab_size):
                model.table[i, j] = rng.normal(0.0, std)
        return model^

    @staticmethod
    def from_counts(
        ids: List[Int], vocab_size: Int, smoothing: Float64
    ) raises -> Self:
        """Build the count model from smoothed bigram frequencies.

        table[i, j] = log((count(i, j) + smoothing) / (count(i, *) +
        V * smoothing)). Add-smoothing (Laplace) keeps every probability
        positive so the log is finite even for unseen bigrams; a row for a token
        that never appears as a source becomes uniform (log(1/V)).

        Args:
            ids: The token id sequence to count bigrams over.
            vocab_size: V, must be positive.
            smoothing: Laplace smoothing constant, must be > 0.

        Returns:
            The count model. Allocates.

        Raises:
            Error: If vocab_size <= 0, smoothing <= 0, or any id is outside
                [0, V).
        """
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
        """Return a copy of row `token_id`: the length-V next-token logits.

        Args:
            token_id: Current token, in [0, V).

        Returns:
            The logits list. Allocates.

        Raises:
            Error: If token_id is outside [0, V).
        """
        if token_id < 0 or token_id >= self.vocab_size:
            raise Error("BigramLM.next_logits: token_id out of range")
        var out = List[Float64]()
        for j in range(self.vocab_size):
            out.append(self.table[token_id, j])
        return out^

    def next_probs(
        self, token_id: Int, temperature: Float64
    ) raises -> List[Float64]:
        """Return the next-token distribution after temperature scaling.

        Args:
            token_id: Current token.
            temperature: Softmax temperature, must be > 0.

        Returns:
            The probability distribution over the next token. Allocates.

        Raises:
            Error: If token_id is out of range or temperature <= 0.
        """
        return softmax_row_temperature(self.next_logits(token_id), temperature)

    def loss_on_batch(self, batch: TokenBatch) raises -> Float64:
        """Return the mean cross-entropy over all B*T (current, next) pairs.

        Args:
            batch: The token batch.

        Returns:
            The mean cross-entropy loss.

        Raises:
            Error: On an out-of-range id or target.
        """
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
        """Fill `grad` with the cross-entropy gradient and return the mean loss.

        Per position (b, t) with current token i and next token target, the
        gradient of the row's cross-entropy wrt its logits is p - onehot(target)
        where p = softmax(table[i]); this is scatter-added into grad[i, :] and
        averaged by 1/(B*T), matching loss_on_batch. No [B, T, V] logits tensor
        is built — the forward is a row lookup. Zeros grad first; mutates grad.

        Args:
            batch: The token batch.
            grad: Gradient accumulator, must be [V, V] (zeroed and overwritten).

        Returns:
            The mean cross-entropy loss.

        Raises:
            Error: If grad is not [V, V], or on an out-of-range id or target.
        """
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
