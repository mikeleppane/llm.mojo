"""Overfit-one-batch smoke: the cheapest end-to-end proof the assembled model learns, driving the loss on one fixed batch below the log V baseline (dropout 0 deterministic + decreasing; dropout 0.1 still ends below init)."""

from std.math import log

from std.testing import assert_true, TestSuite

from llm.config import GPTConfig
from llm.tensor.ops import cross_entropy_rows, cross_entropy_rows_backward
from llm.transformer.gpt import GPT
from llm.utils.random import Rng


def fixed_batch() raises -> List[Int]:
    """One fixed input sequence (T=5) to overfit; ids in [0, V=6)."""
    var out = List[Int]()
    out.append(1)
    out.append(4)
    out.append(2)
    out.append(5)
    out.append(3)
    return out^


def fixed_targets() raises -> List[Int]:
    """The next-token targets for the fixed batch (also in [0, V))."""
    var out = List[Int]()
    out.append(4)
    out.append(2)
    out.append(5)
    out.append(3)
    out.append(0)
    return out^


def train_run(
    dropout: Float64, training: Bool, seed: UInt64, steps: Int, lr: Float64
) raises -> List[Float64]:
    """Overfit the fixed batch for `steps` SGD steps; return the loss before each step (index 0 is the init loss). Deterministic given the seed; the cached forward carries both gradients and dropout.
    """
    var cfg = GPTConfig(6, 8, 8, 2, 2, dropout)
    var rng = Rng(seed)
    var gpt = GPT.init_random(cfg, rng)
    var ids = fixed_batch()
    var targets = fixed_targets()
    var losses = List[Float64]()
    for _ in range(steps):
        var fwd = gpt.forward_cached(ids, training, rng)
        var d_logits = cross_entropy_rows_backward(fwd.logits, targets)
        # Loss before this step, taken off the cached logits (no re-run).
        losses.append(cross_entropy_rows(fwd.logits, targets))
        gpt.zero_grad()
        gpt.backward(fwd.cache, d_logits)
        gpt.apply_sgd(lr)
    return losses^


def test_dropout_zero_overfits_and_decreases() raises:
    """With dropout 0 the loss starts near log V, ends well below it, and falls across checkpoints.
    """
    var steps = 120
    var losses = train_run(0.0, False, 7, steps, 0.5)
    var init = losses[0]
    var final = losses[steps - 1]
    var log_v = log(6.0)
    # Init loss is near the uniform baseline.
    assert_true(
        abs(init - log_v) < 0.6,
        "init loss " + String(init) + " not near log V " + String(log_v),
    )
    # Final loss strictly below the init baseline (the model learned).
    assert_true(
        final < log_v - 0.5,
        "final loss " + String(final) + " not well below log V",
    )
    # Decreasing across checkpoints (indices 0, 30, 60, 90, last).
    var c0 = losses[0]
    var c1 = losses[30]
    var c2 = losses[60]
    var c3 = losses[90]
    var c4 = losses[steps - 1]
    assert_true(c1 < c0, "checkpoint 30 did not fall below 0")
    assert_true(c2 < c1, "checkpoint 60 did not fall below 30")
    assert_true(c3 < c2, "checkpoint 90 did not fall below 60")
    assert_true(c4 < c3, "final did not fall below 90")


def test_dropout_zero_is_deterministic() raises:
    """Dropout 0 draws no rng in the forward, so two runs from the same seed give bit-identical losses.
    """
    var a = train_run(0.0, False, 11, 40, 0.5)
    var b = train_run(0.0, False, 11, 40, 0.5)
    for i in range(len(a)):
        assert_true(
            a[i] == b[i], "dropout=0 run not deterministic at step " + String(i)
        )


def test_dropout_active_still_learns() raises:
    """Dropout 0.1, training on: loss ends below the init baseline (no per-step monotonicity claim under dropout noise).
    """
    var steps = 120
    var losses = train_run(0.1, True, 21, steps, 0.5)
    var log_v = log(6.0)
    var final = losses[steps - 1]
    assert_true(
        final < log_v - 0.3,
        "dropout run final loss " + String(final) + " not below init baseline",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
