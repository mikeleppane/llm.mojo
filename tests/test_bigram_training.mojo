"""Tests for the bigram training loop.

The overfit floor is not 0 in general: if a token is followed by different tokens
in the batch, no table can drive loss to 0; the floor is the batch's conditional
entropy H(next | current). So the criterion splits: a deterministic-bigram batch
(one successor per token) reaches ~0 loss, while a real-text batch converges to
the count model's loss on that same batch (the analytic optimum).
"""

from std.testing import assert_almost_equal, assert_true, TestSuite

from llm.models.bigram import BigramLM
from llm.data.batch import TokenBatch
from llm.data.dataset import TokenDataset
from llm.data.loader import BatchLoader, overfit_batch
from llm.data.corpus import load_text
from llm.tokenizer.char import CharTokenizer
from llm.training.trainer import train_bigram
from llm.training.loss import perplexity
from llm.utils.random import Rng


def _cyclic_batch(vocab_size: Int, repeats: Int) raises -> TokenBatch:
    """A deterministic-bigram batch 0,1,...,V-1,0,1,... where every token has
    exactly one successor: inputs[k] -> targets[k] is (k mod V) -> ((k+1) mod V).
    One row (B=1)."""
    var inputs: List[Int] = []
    var targets: List[Int] = []
    var length = vocab_size * repeats
    for k in range(length):
        inputs.append(k % vocab_size)
        targets.append((k + 1) % vocab_size)
    return TokenBatch(inputs^, targets^, 1, length)


def _prefix(text: String, n: Int) -> String:
    """The first n codepoints of `text` as an owned String (String has no slice).
    """
    var out = String("")
    var count = 0
    for cp in text.codepoint_slices():
        if count >= n:
            break
        out += String(cp)
        count += 1
    return out^


def test_loss_decreases() raises:
    """Full-batch GD on a fixed synthetic batch is non-increasing (with float slack)
    with a clear net drop."""
    var rng = Rng(1)
    var model = BigramLM.random_init(5, 0.1, rng)
    var batch = _cyclic_batch(5, 4)
    var history = train_bigram(model, batch, 50, 0.5)
    assert_true(len(history) == 50)
    for k in range(len(history) - 1):
        assert_true(history[k + 1] <= history[k] + 1e-12)
    assert_true(history[len(history) - 1] < history[0] - 0.1)


def test_overfit_deterministic_batch() raises:
    """With one successor per token the loss floor is 0, so enough GD steps drive it
    below 0.05."""
    var model = BigramLM(6)
    var batch = _cyclic_batch(6, 3)
    var history = train_bigram(model, batch, 2000, 1.0)
    assert_true(history[len(history) - 1] < 0.05)


def test_converges_to_count_optimum() raises:
    """On a real-text batch, a model trained from zeros reaches the count model's
    loss (the analytic optimum) within tolerance."""
    var text = load_text("data/tinyshakespeare/input.txt")
    var slice = _prefix(text, 200)
    var tok = CharTokenizer.from_text(slice)
    var ids = tok.encode(slice)
    var v = tok.vocab_size()

    # One window over the whole slice, so the batch's bigrams are exactly the
    # slice's bigrams that from_counts tallies.
    var inputs: List[Int] = []
    var targets: List[Int] = []
    for k in range(len(ids) - 1):
        inputs.append(ids[k])
        targets.append(ids[k + 1])
    var batch = TokenBatch(inputs^, targets^, 1, len(ids) - 1)

    var count_model = BigramLM.from_counts(ids, v, 1e-6)
    var count_loss = count_model.loss_on_batch(batch)

    # Everything here is deterministic (zeros init, no RNG), so this gap is
    # exactly reproducible; measured ~0.006 at these settings, frozen at 0.02.
    var trained = BigramLM(v)
    var history = train_bigram(trained, batch, 8000, 2.0)
    var trained_loss = history[len(history) - 1]
    assert_true(trained_loss <= count_loss + 0.02)


def test_training_deterministic() raises:
    """Same init seed and batch yield an identical loss history element for element.
    """
    var batch = _cyclic_batch(4, 3)
    var ra = Rng(42)
    var rb = Rng(42)
    var ma = BigramLM.random_init(4, 0.2, ra)
    var mb = BigramLM.random_init(4, 0.2, rb)
    var ha = train_bigram(ma, batch, 30, 0.5)
    var hb = train_bigram(mb, batch, 30, 0.5)
    for k in range(len(ha)):
        assert_almost_equal(ha[k], hb[k], atol=0.0)


def test_loss_decreases_on_tinyshakespeare() raises:
    """On char-tokenized real text, a few hundred steps drop the loss clearly and
    push perplexity below the vocab size (a uniform model scores exactly V)."""
    var text = load_text("data/tinyshakespeare/input.txt")
    var slice = _prefix(text, 4000)
    var tok = CharTokenizer.from_text(slice)
    var ids = tok.encode(slice)
    var v = tok.vocab_size()

    var dataset = TokenDataset(ids^)
    var batch = overfit_batch(dataset, 32, 16)

    var model = BigramLM(v)
    var history = train_bigram(model, batch, 400, 1.0)
    var start = history[0]
    var end = history[len(history) - 1]
    assert_true(end < start - 0.5)  # a clear drop
    assert_true(perplexity(end) < Float64(v))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
