# The capstone: the assembled encoder-decoder actually TRAINS end to end. This is
# the overfit-one-batch integration test — a correct model + hand-written backward
# + SGD loop must drive a tiny batch to EXACT reproduction under greedy decode; if
# it can't, the loss, a gradient, or the optimizer is wrong. Copy is the warmup
# (near-diagonal alignment); reverse is the proof — target position i must fetch
# source position T-1-i, an alignment no local/bigram shortcut can fake, so only
# working cross-attention solves it. The corrupted-memory ablation (zero memory,
# no retraining) then pins cross-attention as load-bearing: exact-match collapses.
#
# The task uses a small model (C=8, T=6, V_data=8) and a tiny 4-pair batch so the
# whole training run stays in test-suite time — the reference forward_cached is
# allocation-heavy (correctness before speed) and its cost climbs steeply with
# d_model, so a small model is the practical choice for a training test. Held-out
# GENERALIZATION at a larger config, and the anti-diagonal cross-attention
# alignment map, are demonstrated in examples/encdec_reverse.mojo (see
# notes/part-12-notes.md for why the test overfits rather than generalizes).
#
# Everything is seeded and deterministic. The thresholds are the one empirical
# spot in the suite: lr and step count were tuned on the branch, then the seeded
# exact-match outcome pinned here.

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
    # Gradient-accumulation training: each step zeroes grads, runs `batch`
    # sequences through forward_cached + backward with d_logits scaled by 1/batch
    # (so the accumulated grad is the batch MEAN), then takes one SGD step.
    # Returns the mean batch loss every 25 steps as a coarse loss curve.
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
    # Decode from a ZEROED memory (the ablation): the decoder gets no information
    # from the source, so it cannot reproduce a source-dependent target.
    var count = 0
    for i in range(len(srcs)):
        var zero_mem = zeros_2d(T, C)
        var decoded = model.greedy_decode_from_memory(zero_mem, T, BOS)
        if sequences_equal(decoded, target_for(srcs[i], reverse)):
            count += 1
    return count


def test_copy_overfit() raises:
    # Warmup: copy is easy (target = source). A short full-batch run drives the
    # loss far below the uniform baseline log(V) and greedy-decodes every training
    # pair exactly — the training loop demonstrably works before reverse leans on
    # it.
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
    # The proof: reverse needs cross-attention (source position T-1-i for target
    # position i). Full-batch training drives the loss down and greedy-decodes
    # every training pair EXACTLY — the assembled backward chain, threaded through
    # cross-attention back into the encoder, trains. Then zeroing memory (one
    # corruption, no retraining) collapses exact-match, proving the decoder
    # actually reads the encoder rather than memorizing target statistics.
    var rng = Rng(101)
    var model = EncDec.init_random(rng, VOCAB, C, H, 1, 1, HIDDEN, T)
    var data_rng = Rng(303)
    var srcs = unique_sources(data_rng, V_DATA, T, 4)
    var losses = train(model, srcs, True, 0.5, 300, 4)

    assert_true(losses[0] > log(Float64(VOCAB)) - 0.6)  # starts near baseline
    assert_true(losses[len(losses) - 1] < losses[0])  # decreases
    assert_true(losses[len(losses) - 1] < 0.3)  # ends well below baseline

    var intact = exact_matches(model, srcs, True)
    assert_equal(intact, len(srcs))  # every training pair reversed exactly

    var ablated = exact_matches_zero_memory(model, srcs, True)
    assert_true(intact > ablated)  # ablation collapses exact-match
    assert_true(ablated <= 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
