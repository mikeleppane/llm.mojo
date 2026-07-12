"""Tests for categorical sampling and a bigram plausibility pin.

The sampler is where determinism and support-respect matter: the same seed
replays the same draws, zero-probability tokens never appear, and a degenerate
one-hot always picks its single support point. The plausibility pin is a hard
assertion: on the full corpus, the count model's most likely successor of 'q'
is 'u'.
"""

from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from llm.generation.sampler import sample_categorical
from llm.models.bigram import BigramLM
from llm.tensor.ops import argmax
from llm.data.corpus import load_text
from llm.tokenizer.char import CharTokenizer
from llm.utils.random import Rng


def test_degenerate_distribution() raises:
    """A one-hot distribution always returns its single support index."""
    var rng = Rng(1)
    var probs: List[Float64] = [0.0, 0.0, 1.0, 0.0]
    for _ in range(20):
        assert_equal(sample_categorical(probs, rng), 2)


def test_seed_deterministic() raises:
    """Same seed gives an identical draw sequence; a different seed diverges."""
    var probs: List[Float64] = [0.25, 0.25, 0.25, 0.25]
    var a = Rng(7)
    var b = Rng(7)
    var draws_a: List[Int] = []
    var draws_b: List[Int] = []
    for _ in range(100):
        draws_a.append(sample_categorical(probs, a))
        draws_b.append(sample_categorical(probs, b))
    for k in range(100):
        assert_equal(draws_a[k], draws_b[k])

    var c = Rng(8)
    var diverged = False
    for k in range(100):
        if sample_categorical(probs, c) != draws_a[k]:
            diverged = True
    assert_true(diverged)


def test_draws_respect_support() raises:
    """A zero-probability entry is never selected across many draws."""
    var probs: List[Float64] = [0.5, 0.0, 0.5]
    var rng = Rng(3)
    for _ in range(1000):
        assert_true(sample_categorical(probs, rng) != 1)


def test_invalid_probs_raise() raises:
    """`sample_categorical` raises on an empty distribution or one not summing to 1.
    """
    var rng = Rng(1)
    with assert_raises(contains="empty"):
        _ = sample_categorical(List[Float64](), rng)
    with assert_raises(contains="sum to 1"):
        _ = sample_categorical([0.2, 0.2], rng)  # sums to 0.4


def test_negative_probability_raises() raises:
    """`sample_categorical` raises on a negative probability."""
    var rng = Rng(1)
    with assert_raises(contains="negative"):
        _ = sample_categorical([1.5, -0.5], rng)  # sums to 1 but invalid


def test_sum_slack_never_selects_zero_entry() raises:
    """A distribution summing just under 1 still never draws a zero entry."""
    # The draw is scaled by the actual total, so u can't land in the [sum, 1) gap
    # and fall through to the last index. Rng(953094) returned the zero entry
    # before the fix.
    var probs: List[Float64] = [0.9999995, 0.0]
    var r = Rng(953094)
    assert_equal(sample_categorical(probs, r), 0)


def test_draw_frequencies_track_probabilities() raises:
    """Drawn mass is proportional to probability (10k draws from [0.25, 0.75]).
    """
    # Catches an inverse-CDF that is in-support but systematically skewed (e.g. a
    # `<` vs `<=` off-by-one).
    var probs: List[Float64] = [0.25, 0.75]
    var rng = Rng(2024)
    var ones = 0
    var draws = 10000
    for _ in range(draws):
        if sample_categorical(probs, rng) == 1:
            ones += 1
    var frac = Float64(ones) / Float64(draws)
    assert_true(frac > 0.72 and frac < 0.78)  # 0.75 +/- 0.03


def test_bigram_plausibility_pin() raises:
    """On the full corpus, the count model's argmax successor of 'q' is 'u'."""
    # No training, no sampling randomness — just the counts.
    var text = load_text("data/tinyshakespeare/input.txt")
    var tok = CharTokenizer.from_text(text)
    var ids = tok.encode(text)
    var model = BigramLM.from_counts(ids, tok.vocab_size(), 1e-6)

    var q = tok.encode("q")[0]
    var u = tok.encode("u")[0]
    assert_equal(argmax(model.next_logits(q)), u)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
