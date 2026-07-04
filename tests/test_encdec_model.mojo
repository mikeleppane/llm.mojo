# Tests for the EncDec model — forward shape, the uniform-baseline init loss,
# model-level causality, the longest gradient path in the repo (source token
# embedding grad, reached only by threading head -> decoder -> cross-attention ->
# encoder -> embedding), the n_dec=2 d_memory-summing pin, and zero_grad /
# apply_sgd coverage.
#
# Finite-difference convention (Part XI, inline): projected here IS the real
# scalar loss cross_entropy_rows, differentiated centrally with h = 1e-5, mixed
# tolerance |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|.

from std.math import log

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_true,
    TestSuite,
)

from llm.lab.encdec import EncDec
from llm.tensor.ops import cross_entropy_rows_backward
from llm.tensor.tensor2d import Tensor2D
from llm.utils.random import Rng


# Small config kept cheap so the finite-difference loops stay fast.
comptime V_DATA = 5
comptime VOCAB = 6  # V_DATA + 1 (BOS)
comptime C = 8
comptime H = 2
comptime HIDDEN = 16
comptime T = 4
comptime BOS = 5  # = V_DATA


def build_model(mut rng: Rng, n_dec: Int) raises -> EncDec:
    return EncDec.init_random(rng, VOCAB, C, H, 1, n_dec, HIDDEN, T)


def sample_src() -> List[Int]:
    return [1, 3, 0, 4]


def sample_tgt_in() -> List[Int]:
    # [BOS] + tgt[:-1] for the reverse of sample_src ([4,0,3,1]) -> tgt_in
    # [BOS,4,0,3]. The exact values do not matter to these tests; a fixed,
    # in-range teacher-forcing input does.
    return [BOS, 4, 0, 3]


def sample_tgt() -> List[Int]:
    return [4, 0, 3, 1]


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


def test_logits_shape() raises:
    var rng = Rng(1)
    var model = build_model(rng, 1)
    var logits = model.forward(sample_src(), sample_tgt_in())
    assert_equal(logits.rows, T)
    assert_equal(logits.cols, VOCAB)


def test_init_loss_near_log_vocab() raises:
    # A freshly initialized model has tiny logits (0.02-scale weights, LayerNorm
    # normalizing the stream), so softmax is near-uniform and the mean
    # cross-entropy sits close to log(V) — the uniform baseline. Averaged over a
    # few sequences to smooth the small residual.
    var rng = Rng(7)
    var model = build_model(rng, 1)
    var total = 0.0
    var n = 5
    for s in range(n):
        var src = List[Int]()
        var tgt = List[Int]()
        var tgt_in = List[Int]()
        tgt_in.append(BOS)
        for i in range(T):
            src.append((s + i) % V_DATA)
            tgt.append((s * 2 + i) % V_DATA)
        for i in range(T - 1):
            tgt_in.append(tgt[i])
        total += model.loss(src, tgt_in, tgt)
    var mean = total / Float64(n)
    assert_almost_equal(mean, log(Float64(VOCAB)), atol=0.15)


def test_model_causality_in_tgt_in() raises:
    # Changing the decoder input at position j leaves logits rows < j unchanged:
    # causal self-attention plus the fact that token j enters only stream row j.
    var rng = Rng(3)
    var model = build_model(rng, 1)
    var src = sample_src()
    var tgt_in = sample_tgt_in()
    var base = model.forward(src, tgt_in)
    var perturbed_in = tgt_in.copy()
    perturbed_in[2] = (perturbed_in[2] + 1) % V_DATA  # change position 2
    var perturbed = model.forward(src, perturbed_in)
    for i in range(2):  # rows 0, 1 must be identical
        for j in range(VOCAB):
            assert_almost_equal(perturbed[i, j], base[i, j], atol=1e-14)


def finite_diff_src_table(
    mut model: EncDec,
    r: Int,
    c: Int,
    src: List[Int],
    tgt_in: List[Int],
    tgt: List[Int],
) raises -> Float64:
    var h = 1e-5
    var saved = model.src_tok.table.value[r, c]
    model.src_tok.table.value[r, c] = saved + h
    var plus = model.loss(src, tgt_in, tgt)
    model.src_tok.table.value[r, c] = saved - h
    var minus = model.loss(src, tgt_in, tgt)
    model.src_tok.table.value[r, c] = saved  # restore
    return (plus - minus) / (2.0 * h)


def check_src_table_grad(n_dec: Int) raises:
    # End-to-end finite-difference of the SOURCE token embedding grad — the
    # longest path in the repo (loss -> head -> decoder stack -> cross-attention
    # -> encoder stack -> src embedding). With n_dec = 2 this ALSO pins that the
    # decoder blocks' d_memory contributions SUM: overwrite instead of sum and
    # the encoder (hence this grad) sees only one block's contribution and the
    # check fails.
    var rng = Rng(11)
    var model = build_model(rng, n_dec)
    var src = sample_src()
    var tgt_in = sample_tgt_in()
    var tgt = sample_tgt()
    var fwd = model.forward_cached(src, tgt_in)
    var d_logits = cross_entropy_rows_backward(fwd.logits, tgt)
    model.zero_grad()
    model.backward(fwd.cache, d_logits)
    var analytic = model.src_tok.table.grad.copy()
    for r in range(analytic.rows):
        for c in range(analytic.cols):
            var numeric = finite_diff_src_table(model, r, c, src, tgt_in, tgt)
            assert_grad_close(analytic[r, c], numeric)


def test_src_table_grad_finite_difference_n_dec_1() raises:
    check_src_table_grad(1)


def test_src_table_grad_finite_difference_n_dec_2_sums_memory() raises:
    check_src_table_grad(2)


def test_zero_grad_and_apply_sgd_touch_every_parameter() raises:
    # After a backward with nonzero d_logits, representative grads across every
    # layer type are nonzero; zero_grad drives them to exactly zero; a step with
    # nonzero grads moves every representative value. Covers each layer family:
    # the four embeddings, encoder + decoder sublayers (including cross-attn),
    # both final LayerNorms, and the head.
    var rng = Rng(5)
    var model = build_model(rng, 1)
    var src = sample_src()
    var tgt_in = sample_tgt_in()
    var tgt = sample_tgt()
    var fwd = model.forward_cached(src, tgt_in)
    var d_logits = cross_entropy_rows_backward(fwd.logits, tgt)
    model.zero_grad()
    model.backward(fwd.cache, d_logits)

    # A representative grad from each family is nonzero after backward.
    assert_true(grad_nonzero(model.src_tok.table.grad))
    assert_true(grad_nonzero(model.src_pos.table.grad))
    assert_true(grad_nonzero(model.tgt_tok.table.grad))
    assert_true(grad_nonzero(model.tgt_pos.table.grad))
    assert_true(grad_nonzero(model.encoder[0].attn.qkv.weight.grad))
    assert_true(grad_nonzero(model.encoder[0].mlp.up.weight.grad))
    assert_true(grad_nonzero(model.enc_ln_f.weight.grad))
    assert_true(grad_nonzero(model.decoder[0].self_attn.qkv.weight.grad))
    assert_true(grad_nonzero(model.decoder[0].cross_attn.kv.weight.grad))
    assert_true(grad_nonzero(model.decoder[0].mlp.down.weight.grad))
    assert_true(grad_nonzero(model.dec_ln_f.weight.grad))
    assert_true(grad_nonzero(model.head.weight.grad))

    # Snapshot representative values, take a step, confirm each moved.
    var before_src = model.src_tok.table.value[0, 0]
    var before_head = model.head.weight.value[0, 0]
    var before_ckv = model.decoder[0].cross_attn.kv.weight.value[0, 0]
    var before_enclnf = model.enc_ln_f.weight.value[0, 0]
    model.apply_sgd(0.1)
    assert_true(abs(model.src_tok.table.value[0, 0] - before_src) > 1e-12)
    assert_true(abs(model.head.weight.value[0, 0] - before_head) > 1e-12)
    assert_true(
        abs(model.decoder[0].cross_attn.kv.weight.value[0, 0] - before_ckv)
        > 1e-12
    )
    assert_true(abs(model.enc_ln_f.weight.value[0, 0] - before_enclnf) > 1e-12)

    # zero_grad clears every representative grad to exactly zero.
    model.zero_grad()
    assert_true(not grad_nonzero(model.src_tok.table.grad))
    assert_true(not grad_nonzero(model.decoder[0].cross_attn.kv.weight.grad))
    assert_true(not grad_nonzero(model.head.weight.grad))


def grad_nonzero(g: Tensor2D) -> Bool:
    for i in range(g.rows):
        for j in range(g.cols):
            if abs(g[i, j]) > 1e-12:
                return True
    return False


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
