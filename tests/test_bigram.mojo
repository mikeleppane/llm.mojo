# Tests for the BigramLM model: loss plumbing, the count model, and the gradient.
#
# The gradient is the load-bearing part, so it gets two independent checks: a
# full finite-difference comparison on every table entry (V = 3) and the
# structural invariant that every gradient row sums to zero (each p - onehot
# contribution sums to 0). The zeros-table loss (log V) and the hand-computed
# count model pin the forward pass before any training runs.

from std.testing import assert_almost_equal, assert_raises, TestSuite
from std.math import log, exp

from llm.models.bigram import BigramLM
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.data.batch import TokenBatch
from llm.utils.random import Rng
from llm.training.loss import perplexity


def _batch(
    inputs: List[Int], targets: List[Int], b: Int, t: Int
) raises -> TokenBatch:
    return TokenBatch(inputs.copy(), targets.copy(), b, t)


def test_zeros_table_loss_is_log_v() raises:
    # An untrained (zeros) model is uniform, so the loss is log(V) on any batch.
    var model = BigramLM(3)
    var batch = _batch([0, 1, 2], [1, 2, 0], 1, 3)
    assert_almost_equal(model.loss_on_batch(batch), log(3.0), atol=1e-12)


def test_from_counts_hand_computed() raises:
    # Corpus [0,1,0,1,0], V=2, smoothing 1. Bigrams: (0->1) x2, (1->0) x2.
    # counts: c(0,0)=0, c(0,1)=2, c(1,0)=2, c(1,1)=0; row totals 2 and 2.
    # table[i,j] = log((c+1) / (2 + 2*1)):
    #   table[0,0]=log(1/4), table[0,1]=log(3/4),
    #   table[1,0]=log(3/4), table[1,1]=log(1/4).
    var model = BigramLM.from_counts([0, 1, 0, 1, 0], 2, 1.0)
    assert_almost_equal(model.table[0, 0], log(0.25), atol=1e-12)
    assert_almost_equal(model.table[0, 1], log(0.75), atol=1e-12)
    assert_almost_equal(model.table[1, 0], log(0.75), atol=1e-12)
    assert_almost_equal(model.table[1, 1], log(0.25), atol=1e-12)

    # Loss on the same corpus: every bigram has P = 3/4, so CE = -log(3/4).
    var batch = _batch([0, 1, 0, 1], [1, 0, 1, 0], 1, 4)
    assert_almost_equal(model.loss_on_batch(batch), -log(0.75), atol=1e-12)


def test_from_counts_requires_positive_smoothing() raises:
    with assert_raises(contains="smoothing"):
        _ = BigramLM.from_counts([0, 1, 0], 2, 0.0)


def test_from_counts_rejects_out_of_range_id() raises:
    with assert_raises(contains="out of range"):
        _ = BigramLM.from_counts([0, 5, 1], 2, 1.0)


def test_grad_matches_finite_difference() raises:
    # Central-difference check of every table entry against loss_and_grad, on a
    # tiny V=3 case with a non-trivial batch and a random (non-uniform) table.
    var rng = Rng(7)
    var model = BigramLM.random_init(3, 0.5, rng)
    var batch = _batch([0, 1, 2, 0], [1, 2, 0, 2], 1, 4)

    var grad = zeros_2d(3, 3)
    _ = model.loss_and_grad(batch, grad)

    var h = 1e-5
    for i in range(3):
        for j in range(3):
            var plus = model.copy()
            plus.table[i, j] = plus.table[i, j] + h
            var minus = model.copy()
            minus.table[i, j] = minus.table[i, j] - h
            var numeric = (
                plus.loss_on_batch(batch) - minus.loss_on_batch(batch)
            ) / (2.0 * h)
            assert_almost_equal(grad[i, j], numeric, atol=1e-4)


def test_grad_rows_sum_to_zero() raises:
    # Each gradient row accumulates (softmax - onehot) contributions, and every
    # such contribution sums to 0, so every row sums to 0.
    var rng = Rng(3)
    var model = BigramLM.random_init(4, 0.7, rng)
    var batch = _batch([0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 1], 2, 3)
    var grad = zeros_2d(4, 4)
    _ = model.loss_and_grad(batch, grad)
    for i in range(4):
        var s = 0.0
        for j in range(4):
            s += grad[i, j]
        assert_almost_equal(s, 0.0, atol=1e-9)


def test_loss_and_grad_matches_loss_on_batch() raises:
    # The loss returned by loss_and_grad is the same mean CE as loss_on_batch.
    var rng = Rng(9)
    var model = BigramLM.random_init(3, 0.4, rng)
    var batch = _batch([0, 1, 2], [2, 0, 1], 1, 3)
    var grad = zeros_2d(3, 3)
    var g_loss = model.loss_and_grad(batch, grad)
    assert_almost_equal(g_loss, model.loss_on_batch(batch), atol=1e-12)


def test_loss_and_grad_rejects_wrong_grad_shape() raises:
    var model = BigramLM(3)
    var batch = _batch([0, 1], [1, 2], 1, 2)
    var grad = zeros_2d(2, 3)  # wrong shape
    with assert_raises(contains="grad shape"):
        _ = model.loss_and_grad(batch, grad)


def test_perplexity_is_exp_loss() raises:
    var model = BigramLM(5)
    var batch = _batch([0, 1, 2, 3], [1, 2, 3, 4], 1, 4)
    var loss = model.loss_on_batch(batch)
    assert_almost_equal(perplexity(loss), exp(loss), atol=1e-12)
    # A uniform model over V=5 has perplexity exactly 5.
    assert_almost_equal(perplexity(loss), 5.0, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
