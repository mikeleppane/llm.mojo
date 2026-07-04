# Train the encoder-decoder lab on the reverse task and watch cross-attention
# learn the alignment the task demands.
#
# The figure this prints is the chapter's thesis: before training the decoder's
# cross-attention row-argmax map is noise; after training it is the anti-diagonal
# — target position i attends to source position T-1-i, exactly the fetch reverse
# requires. Alongside it, a handful of held-out sources are greedy-decoded next
# to their reversed truths: the model reverses sequences it never trained on.
#
# Run:  pixi run mojo run -I src examples/encdec_reverse.mojo

from std.math import log

from llm.lab.encdec import EncDec
from llm.lab.tasks import (
    reverse_target,
    decoder_input,
    sequences_equal,
    unique_sources,
)
from llm.tensor.ops import (
    argmax,
    cross_entropy_rows,
    cross_entropy_rows_backward,
    scale,
)
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.utils.random import Rng

comptime V_DATA = 8
comptime VOCAB = 9
comptime C = 8
comptime H = 2
comptime HIDDEN = 32
comptime T = 6
comptime BOS = 8


def averaged_cross_weights(model: EncDec, src: List[Int]) raises -> Tensor2D:
    # The decoder block's cross-attention weights, averaged over heads:
    # [T_tgt, T_src], row i = what target position i attends to across the source.
    # Uses teacher-forced inputs (the true reverse target) so the map reflects the
    # aligned decoding path. Pulled straight from the forward cache — no second
    # code path recomputes the weights.
    var tgt = reverse_target(src)
    var tgt_in = decoder_input(tgt, BOS)
    var fwd = model.forward_cached(src, tgt_in)
    var heads = fwd.cache.dec_caches[0].cross_attn_cache.head_caches.copy()
    var avg = zeros_2d(T, T)
    for h in range(len(heads)):
        var w = heads[h].weights.copy()  # [T_tgt, T_src]
        for i in range(T):
            for j in range(T):
                avg[i, j] = avg[i, j] + w[i, j] / Float64(len(heads))
    return avg^


def print_alignment(model: EncDec, src: List[Int]) raises:
    # Print the row-argmax of the averaged cross-attention weights: for each
    # target position, the source position it most attends to. A trained reverse
    # model tends toward 5 4 3 2 1 0 (the anti-diagonal at T=6).
    var avg = averaged_cross_weights(model, src)
    var line = String("  target pos -> source pos:  ")
    for i in range(T):
        var row = List[Float64]()
        for j in range(T):
            row.append(avg[i, j])
        line += String(argmax(row))
        line += " "
    print(line)


def train_reverse(
    mut model: EncDec,
    srcs: List[List[Int]],
    lr: Float64,
    steps: Int,
    batch: Int,
) raises:
    var n = len(srcs)
    var inv_b = 1.0 / Float64(batch)
    for step in range(steps):
        model.zero_grad()
        var batch_loss = 0.0
        for b in range(batch):
            var idx = (step * batch + b) % n
            var src = srcs[idx].copy()
            var tgt = reverse_target(src)
            var tgt_in = decoder_input(tgt, BOS)
            var fwd = model.forward_cached(src, tgt_in)
            batch_loss += cross_entropy_rows(fwd.logits, tgt)
            var d_logits = cross_entropy_rows_backward(fwd.logits, tgt)
            model.backward(fwd.cache, scale(d_logits, inv_b))
        model.apply_sgd(lr)
        if step % 50 == 0:
            print("  step", step, " loss", batch_loss * inv_b)


def print_sequence(label: String, seq: List[Int]):
    var line = label
    for i in range(len(seq)):
        line += String(seq[i])
        line += " "
    print(line)


def main() raises:
    print("Encoder-decoder lab: the reverse task")
    print("=====================================")
    var rng = Rng(101)
    var model = EncDec.init_random(rng, VOCAB, C, H, 1, 1, HIDDEN, T)

    var data_rng = Rng(303)
    var all_srcs = unique_sources(data_rng, V_DATA, T, 40)
    var train_srcs = List[List[Int]]()
    var held_srcs = List[List[Int]]()
    for i in range(len(all_srcs)):
        if i < 32:
            train_srcs.append(all_srcs[i].copy())
        else:
            held_srcs.append(all_srcs[i].copy())

    print("\nCross-attention alignment BEFORE training (expect noise):")
    print_alignment(model, held_srcs[0])
    print(
        "\nInitial loss is near the uniform baseline log(V) =",
        log(Float64(VOCAB)),
    )

    print("\nTraining on", len(train_srcs), "reverse pairs...")
    train_reverse(model, train_srcs, 0.5, 800, 8)

    print(
        "\nCross-attention alignment AFTER training (anti-diagonal 5 4 3 2"
        " 1 0):"
    )
    print_alignment(model, held_srcs[0])

    print("\nGreedy-decoding held-out sources (never trained):")
    var correct = 0
    for i in range(len(held_srcs)):
        var decoded = model.greedy_decode(held_srcs[i], T, BOS)
        var truth = reverse_target(held_srcs[i])
        if sequences_equal(decoded, truth):
            correct += 1
        if i < 4:
            print_sequence("  source:  ", held_srcs[i])
            print_sequence("  decoded: ", decoded)
            print_sequence("  truth:   ", truth)
            print("  ---")
    print("Held-out exact matches:", correct, "/", len(held_srcs))
