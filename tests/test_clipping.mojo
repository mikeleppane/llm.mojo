"""Tests for global-norm gradient clipping: GPT.grad_norm (whole-model L2),
GPT.scale_grads (the clip multiply), and clip_grad_norm (the composition).

The norm is global (one vector norm over every gradient entry, not per-tensor), so
the hand-computed cases plant entries on several parameters and check the single
combined norm. Clipping is a bit-for-bit no-op below the threshold and brings the
norm to exactly the threshold above it.
"""

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
    """A tiny GPT: V=8, context=8, C=8, L=2, H=2, dropout 0."""
    var cfg = GPTConfig(8, 8, 8, 2, 2, 0.0)
    var rng = Rng(seed)
    return GPT.init_random(cfg, rng)


def _backward_into(mut gpt: GPT) raises:
    """Plant real, nonzero gradients across the model (dropout off; no rng drawn).
    """
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
    """Entries 3, 4, 12 planted on three different parameters give a global norm
    sqrt(3^2+4^2+12^2)=13, which a per-tensor norm could never report."""
    var gpt = _tiny_gpt(1)
    gpt.zero_grad()
    gpt.wte.table.grad[0, 0] = 3.0
    gpt.ln_f.bias.grad[0, 0] = 4.0
    gpt.blocks[0].mlp.down.bias.grad[0, 0] = 12.0
    assert_almost_equal(gpt.grad_norm(), 13.0, atol=1e-12)


def test_clip_below_threshold_is_exact_noop() raises:
    """Norm 0.5 below clip 1.0: clip_grad_norm returns the norm and leaves every
    gradient byte-for-byte unchanged."""
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
    """Norm 5 exceeds clip 1.0: every grad scales by 1/5, the post-clip norm equals
    the clip, and the returned norm is the pre-clip value."""
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
    """All-zero grads: norm 0 never exceeds a positive clip, so no division occurs
    and the gradients stay exactly zero (no NaN from 0/0)."""
    var gpt = _tiny_gpt(1)
    gpt.zero_grad()
    var norm = clip_grad_norm(gpt, 1.0)
    assert_true(norm == 0.0, "zero-grad norm not zero")
    assert_true(gpt.wte.table.grad[0, 0] == 0.0, "zero grad perturbed")
    assert_true(gpt.grad_norm() == 0.0, "grad_norm nonzero after zero clip")


def test_scale_grads_touches_every_grad() raises:
    """Scaling multiplies every gradient: after scale_grads by 2.0, every entry is
    exactly doubled (a skipped parameter would keep its value)."""
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
    """Rejects a non-positive max_norm in clip_grad_norm."""
    var gpt = _tiny_gpt(1)
    gpt.zero_grad()
    with assert_raises(contains="max_norm must be positive"):
        _ = clip_grad_norm(gpt, 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
