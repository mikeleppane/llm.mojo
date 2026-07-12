"""Capstone overfit-one-batch integration test: the assembled encoder-decoder trains end to end (forward + hand-written backward + SGD) and drives a tiny seeded batch to EXACT greedy-decode reproduction. Copy is the warmup; reverse (target position i fetches source position T-1-i) proves cross-attention works, and a zeroed-memory ablation pins cross-attention as load-bearing when exact-match collapses. Everything is seeded and deterministic; lr and step count were tuned, then the exact-match outcome pinned here."""

from std.math import log

from std.testing import assert_true, assert_equal, TestSuite

from llm.lab.encdec import EncDec
from llm.lab.tasks import (
    copy_target,
    reverse_target,
    decoder_input,
    sequences_equal,
    unique_sources,
)
from llm.tensor.ops import (
    cross_entropy_rows,
    cross_entropy_rows_backward,
    scale,
)
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.utils.random import Rng

comptime V_DATA = 8
comptime VOCAB = 9  # V_DATA + 1 (BOS)
comptime C = 8
comptime H = 2
comptime HIDDEN = 32
comptime T = 6
comptime BOS = 8  # = V_DATA


def target_for(src: List[Int], reverse: Bool) -> List[Int]:
    if reverse:
        return reverse_target(src)
    return copy_target(src)


def train(
    mut model: EncDec,
    srcs: List[List[Int]],
    reverse: Bool,
    lr: Float64,
    steps: Int,
    batch: Int,
) raises -> List[Float64]:
    """Gradient-accumulation training loop.

    Each step zeroes grads, runs `batch` sequences through forward_cached +
    backward with d_logits scaled by 1/batch (so the accumulated grad is the
    batch MEAN), then takes one SGD step.

    Returns:
        The mean batch loss every 25 steps, as a coarse loss curve.
    """
    var losses = List[Float64]()
    var n = len(srcs)
    var inv_b = 1.0 / Float64(batch)
    for step in range(steps):
        model.zero_grad()
        var batch_loss = 0.0
        for b in range(batch):
            var idx = (step * batch + b) % n
            var src = srcs[idx].copy()
            var tgt = target_for(src, reverse)
            var tgt_in = decoder_input(tgt, BOS)
            var fwd = model.forward_cached(src, tgt_in)
            batch_loss += cross_entropy_rows(fwd.logits, tgt)
            var d_logits = cross_entropy_rows_backward(fwd.logits, tgt)
            model.backward(fwd.cache, scale(d_logits, inv_b))
        model.apply_sgd(lr)
        if step % 25 == 0:
            losses.append(batch_loss * inv_b)
    return losses^


def exact_matches(
    model: EncDec, srcs: List[List[Int]], reverse: Bool
) raises -> Int:
    var count = 0
    for i in range(len(srcs)):
        var decoded = model.greedy_decode(srcs[i], T, BOS)
        if sequences_equal(decoded, target_for(srcs[i], reverse)):
            count += 1
    return count


def exact_matches_zero_memory(
    model: EncDec, srcs: List[List[Int]], reverse: Bool
) raises -> Int:
    """Count exact matches when decoding from a ZEROED memory (the ablation).

    The decoder gets no information from the source, so it cannot reproduce a
    source-dependent target.
    """
    var count = 0
    for i in range(len(srcs)):
        var zero_mem = zeros_2d(T, C)
        var decoded = model.greedy_decode_from_memory(zero_mem, T, BOS)
        if sequences_equal(decoded, target_for(srcs[i], reverse)):
            count += 1
    return count


def test_copy_overfit() raises:
    """Warmup: copy (target = source) drives loss below the log(V) baseline and greedy-decodes every training pair exactly.
    """
    var rng = Rng(101)
    var model = EncDec.init_random(rng, VOCAB, C, H, 1, 1, HIDDEN, T)
    var data_rng = Rng(202)
    var srcs = unique_sources(data_rng, V_DATA, T, 2)
    var losses = train(model, srcs, False, 0.5, 300, 2)
    assert_true(losses[0] > log(Float64(VOCAB)) - 0.6)  # starts near baseline
    assert_true(losses[len(losses) - 1] < losses[0])  # decreases
    assert_true(losses[len(losses) - 1] < 0.3)  # ends well below baseline
    assert_equal(exact_matches(model, srcs, False), len(srcs))


def test_reverse_overfit_and_ablation() raises:
    """Reverse (source position T-1-i for target i) needs cross-attention: training decodes every pair exactly, then zeroing memory collapses exact-match, proving the decoder reads the encoder rather than memorizing target statistics.
    """
    var rng = Rng(101)
    var model = EncDec.init_random(rng, VOCAB, C, H, 1, 1, HIDDEN, T)
    var data_rng = Rng(303)
    var srcs = unique_sources(data_rng, V_DATA, T, 4)
    var losses = train(model, srcs, True, 0.5, 300, 4)

    assert_true(losses[0] > log(Float64(VOCAB)) - 0.6)  # starts near baseline
    assert_true(losses[len(losses) - 1] < losses[0])  # decreases
    assert_true(losses[len(losses) - 1] < 0.3)  # ends well below baseline

    # intact == len(srcs) is the load-bearing gate: greedy-decoding EVERY source
    # to its exact reversal requires a trained decoder that reads the encoder (an
    # untrained or memory-ignoring model cannot hit 4/4).
    var intact = exact_matches(model, srcs, True)
    assert_equal(
        intact, len(srcs)
    )  # every training pair reversed exactly (4/4)

    # Ablation: zero the memory (one corruption, no retraining). With no source
    # signal the decode is source-BLIND — one constant sequence — so it can match
    # at most one of the four DISTINCT reversals whatever the model learned. The
    # collapse from 4/4 (intact) to <=1 (ablated) is the proof the 4/4 came from
    # reading the encoder, not from memorized target statistics.
    var ablated = exact_matches_zero_memory(model, srcs, True)
    assert_true(intact > ablated)  # exact-match collapses
    assert_true(ablated <= 1)  # source-blind decode matches at most one target


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
