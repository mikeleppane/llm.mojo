# Tests for global-norm gradient clipping: GPT.grad_norm (the whole-model L2),
# GPT.scale_grads (the clip multiply), and clip_grad_norm (the composition).
#
# The norm is GLOBAL — one vector norm over every gradient entry in the model,
# not a per-tensor norm — so the hand-computed cases set entries on SEVERAL
# different parameters (wte, ln_f.bias, a block bias) and check the single
# combined norm. Clipping is a no-op below the threshold (bit-for-bit) and brings
# the norm to exactly the threshold above it.

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.config import GPTConfig
from llm.training.optimizer import clip_grad_norm
from llm.tensor.ops import cross_entropy_rows_backward
from llm.transformer.gpt import GPT
from llm.utils.random import Rng


def _tiny_gpt(seed: UInt64) raises -> GPT:
    # V=8, C=8, context=8, L=2, H=2, dropout 0.
    var cfg = GPTConfig(8, 8, 8, 2, 2, 0.0)
    var rng = Rng(seed)
    return GPT.init_random(cfg, rng)


def _backward_into(mut gpt: GPT) raises:
    # Real, nonzero gradients across the model (dropout off; no rng drawn).
    var ids = List[Int]()
    ids.append(1)
    ids.append(4)
    ids.append(2)
    ids.append(6)
    var targets = List[Int]()
    targets.append(4)
    targets.append(2)
    targets.append(6)
    targets.append(3)
    var rng = Rng(0)
    gpt.zero_grad()
    var fwd = gpt.forward_cached(ids, False, rng)
    var d_logits = cross_entropy_rows_backward(fwd.logits, targets)
    gpt.backward(fwd.cache, d_logits)


def test_grad_norm_hand_computed_across_tensors() raises:
    # Zero every gradient, then plant three entries on three DIFFERENT parameters:
    # 3 on wte, 4 on ln_f.bias, 12 on block 0's mlp.down.bias. The global norm is
    # sqrt(3^2 + 4^2 + 12^2) = sqrt(169) = 13 — a per-tensor norm would report
    # 3, 4, or 12, never 13.
    var gpt = _tiny_gpt(1)
    gpt.zero_grad()
    gpt.wte.table.grad[0, 0] = 3.0
    gpt.ln_f.bias.grad[0, 0] = 4.0
    gpt.blocks[0].mlp.down.bias.grad[0, 0] = 12.0
    assert_almost_equal(gpt.grad_norm(), 13.0, atol=1e-12)


def test_clip_below_threshold_is_exact_noop() raises:
    # Norm 0.5 (from 0.3 and 0.4) is below clip 1.0, so clip_grad_norm returns the
    # norm and leaves every gradient byte-for-byte unchanged.
    var gpt = _tiny_gpt(1)
    gpt.zero_grad()
    gpt.wte.table.grad[0, 0] = 0.3
    gpt.ln_f.bias.grad[0, 0] = 0.4
    var norm = clip_grad_norm(gpt, 1.0)
    assert_almost_equal(norm, 0.5, atol=1e-12)
    # Unchanged EXACTLY (no multiply happened).
    assert_true(
        gpt.wte.table.grad[0, 0] == 0.3, "wte grad changed below thresh"
    )
    assert_true(
        gpt.ln_f.bias.grad[0, 0] == 0.4, "ln_f grad changed below thresh"
    )


def test_clip_above_threshold_scales_to_clip() raises:
    # Norm 5 (from 3 and 4) exceeds clip 1.0: every grad is scaled by 1/5, the
    # post-clip global norm is exactly the clip, and the returned norm is the
    # PRE-clip value.
    var gpt = _tiny_gpt(1)
    gpt.zero_grad()
    gpt.wte.table.grad[0, 0] = 3.0
    gpt.ln_f.bias.grad[0, 0] = 4.0
    var norm = clip_grad_norm(gpt, 1.0)
    assert_almost_equal(norm, 5.0, atol=1e-12)  # pre-clip
    assert_almost_equal(gpt.grad_norm(), 1.0, atol=1e-12)  # post-clip == clip
    # Direction preserved: each entry scaled by clip/norm = 0.2.
    assert_almost_equal(gpt.wte.table.grad[0, 0], 0.6, atol=1e-12)
    assert_almost_equal(gpt.ln_f.bias.grad[0, 0], 0.8, atol=1e-12)


def test_clip_zero_grad_no_blowup() raises:
    # All grads zero: norm 0 never exceeds a positive clip, so no division occurs
    # and the gradients stay exactly zero (no NaN from 0/0).
    var gpt = _tiny_gpt(1)
    gpt.zero_grad()
    var norm = clip_grad_norm(gpt, 1.0)
    assert_true(norm == 0.0, "zero-grad norm not zero")
    assert_true(gpt.wte.table.grad[0, 0] == 0.0, "zero grad perturbed")
    assert_true(gpt.grad_norm() == 0.0, "grad_norm nonzero after zero clip")


def test_scale_grads_touches_every_grad() raises:
    # scale_grads must multiply EVERY gradient. With real grads, exporting before
    # and after a scale by 2.0, every entry must be exactly doubled — a skipped
    # parameter would keep its original value.
    var gpt = _tiny_gpt(2)
    _backward_into(gpt)
    var before = gpt.export_gradients()
    gpt.scale_grads(2.0)
    var after = gpt.export_gradients()
    assert_true(len(before) == len(after), "grad tensor count drift")
    for k in range(len(before)):
        for i in range(before[k].rows):
            for j in range(before[k].cols):
                assert_almost_equal(
                    after[k][i, j], 2.0 * before[k][i, j], atol=1e-15
                )


def test_clip_grad_norm_rejects_nonpositive() raises:
    var gpt = _tiny_gpt(1)
    gpt.zero_grad()
    with assert_raises(contains="max_norm must be positive"):
        _ = clip_grad_norm(gpt, 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
