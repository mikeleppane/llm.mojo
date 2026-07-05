# Tests for training checkpoints — bit-exact save/load and the resume gate.
#
# Checkpoints must round-trip EXACTLY (bit patterns, not tolerances): a resumed
# run has to be indistinguishable from an uninterrupted one. So the round-trip
# and resume checks assert exact equality, and the header-validation checks pin
# that a wrong-shaped, truncated, or mis-magicked file fails loudly instead of
# loading garbage.
#
# The capstone here is the resume gate: train n steps straight, versus train k,
# checkpoint, load into a FRESH model, train n-k more — the parameters must be
# BIT-IDENTICAL. Run on the overfit-batch setup (a fixed batch, no loader state),
# with the lr driven by the schedule off the step index so a mis-restored step
# counter would pick the wrong lr and diverge.

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.config import GPTConfig
from llm.tensor.ops import cross_entropy_rows_backward
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.training.checkpoint import (
    f64_to_hex,
    hex_to_f64,
    load_checkpoint,
    save_checkpoint,
)
from llm.training.optimizer import clip_grad_norm
from llm.training.schedule import lr_at
from llm.transformer.gpt import GPT
from llm.utils.random import Rng

comptime B1 = 0.9
comptime B2 = 0.95
comptime EPS = 1e-8
comptime WD = 0.1


def _tiny_gpt(seed: UInt64) raises -> GPT:
    var cfg = GPTConfig(8, 8, 8, 2, 2, 0.0)  # V, ctx, C, L, H, dropout
    var rng = Rng(seed)
    return GPT.init_random(cfg, rng)


def _ids() raises -> List[Int]:
    var out = List[Int]()
    out.append(1)
    out.append(4)
    out.append(2)
    out.append(6)
    out.append(3)
    return out^


def _targets() raises -> List[Int]:
    var out = List[Int]()
    out.append(4)
    out.append(2)
    out.append(6)
    out.append(3)
    out.append(0)
    return out^


def _zeros_state(gpt: GPT) raises -> List[Tensor2D]:
    # A parallel list of zeros, one per parameter, shaped by the walk.
    var shapes = gpt.parameter_shapes()
    var out = List[Tensor2D]()
    for k in range(len(shapes)):
        out.append(zeros_2d(shapes[k].rows, shapes[k].cols))
    return out^


def _adamw_step(
    mut gpt: GPT,
    mut m: List[Tensor2D],
    mut v: List[Tensor2D],
    t: Int,
    lr: Float64,
) raises:
    # One full optimizer step on the fixed overfit batch: zero_grad -> cached
    # forward (dropout 0, deterministic) -> loss backward -> clip -> AdamW. This
    # is exactly the per-step body train_gpt will run.
    var rng = Rng(0)  # dropout 0 draws nothing; a named var is still required
    gpt.zero_grad()
    var fwd = gpt.forward_cached(_ids(), False, rng)
    var d_logits = cross_entropy_rows_backward(fwd.logits, _targets())
    gpt.backward(fwd.cache, d_logits)
    _ = clip_grad_norm(gpt, 1.0)
    gpt.apply_adamw(m, v, t, lr, B1, B2, EPS, WD)


def _assert_params_bit_identical(a: GPT, b: GPT, msg: String) raises:
    var pa = a.export_parameters()
    var pb = b.export_parameters()
    assert_true(len(pa) == len(pb), msg + ": tensor count")
    for k in range(len(pa)):
        for i in range(pa[k].rows):
            for j in range(pa[k].cols):
                assert_true(
                    pa[k][i, j] == pb[k][i, j], msg + " at " + String(k)
                )


def test_hex_roundtrip_edge_values() raises:
    # The bit-pattern round-trip is exact for zero, negative zero, subnormals,
    # and large/small magnitudes — the values a decimal print might mangle.
    var vals = List[Float64]()
    vals.append(0.0)
    vals.append(-0.0)
    vals.append(1.0)
    vals.append(-3.141592653589793)
    vals.append(1e-300)
    vals.append(1e300)
    vals.append(123456.789)
    for i in range(len(vals)):
        var s = f64_to_hex(vals[i])
        assert_true(s.byte_length() == 16, "hex not 16 chars")
        assert_true(
            hex_to_f64(s) == vals[i], "roundtrip drift at index " + String(i)
        )


def test_save_load_roundtrip_bit_exact() raises:
    # Train a few steps to fill params, m, v, t; save; load into a DIFFERENTLY
    # seeded fresh model; every restored number matches bit-for-bit.
    var gpt = _tiny_gpt(1)
    var m = _zeros_state(gpt)
    var v = _zeros_state(gpt)
    for s in range(1, 4):
        _adamw_step(gpt, m, v, s, 0.02)
    var rng_state: UInt64 = 0x0123456789ABCDEF
    var path = String("build/ckpt_roundtrip.ckpt")
    save_checkpoint(path, gpt, m, v, 3, rng_state)

    var fresh = _tiny_gpt(999)  # different init — load must overwrite all of it
    var state = load_checkpoint(path, fresh)

    _assert_params_bit_identical(gpt, fresh, "params")
    assert_true(state.t == 3, "step counter not restored")
    assert_true(state.rng_state == rng_state, "rng state not restored")
    assert_true(len(state.m) == len(m) and len(state.v) == len(v), "state len")
    for k in range(len(m)):
        for i in range(m[k].rows):
            for j in range(m[k].cols):
                assert_true(state.m[k][i, j] == m[k][i, j], "m drift")
                assert_true(state.v[k][i, j] == v[k][i, j], "v drift")


def test_load_rejects_bad_magic() raises:
    var path = String("build/ckpt_badmagic.ckpt")
    with open(path, "w") as f:
        f.write("NOTACHECKPOINT\n1\n0\n0000000000000000\n")
    var gpt = _tiny_gpt(1)
    with assert_raises(contains="bad magic"):
        _ = load_checkpoint(path, gpt)


def test_load_rejects_shape_mismatch() raises:
    # Save from a d_model=8 model, load into a d_model=16 model: same tensor
    # count, different shapes, so the per-tensor shape check must fire (at tensor
    # 0, wte, which the wider d_model reshapes).
    var small = _tiny_gpt(1)
    var m = _zeros_state(small)
    var v = _zeros_state(small)
    var path = String("build/ckpt_shape.ckpt")
    save_checkpoint(path, small, m, v, 0, 0)

    var big_cfg = GPTConfig(8, 8, 16, 2, 2, 0.0)  # d_model 16 instead of 8
    var big_rng = Rng(1)
    var big = GPT.init_random(big_cfg, big_rng)
    with assert_raises(contains="shape mismatch"):
        _ = load_checkpoint(path, big)


def test_load_rejects_parameter_count_mismatch() raises:
    # A different depth changes the tensor count outright.
    var two_layer = _tiny_gpt(1)
    var m = _zeros_state(two_layer)
    var v = _zeros_state(two_layer)
    var path = String("build/ckpt_count.ckpt")
    save_checkpoint(path, two_layer, m, v, 0, 0)

    var three_cfg = GPTConfig(8, 8, 8, 3, 2, 0.0)  # L=3 instead of 2
    var three_rng = Rng(1)
    var three = GPT.init_random(three_cfg, three_rng)
    with assert_raises(contains="parameter count mismatch"):
        _ = load_checkpoint(path, three)


def test_load_rejects_truncated() raises:
    # Chop the tail off a valid checkpoint: loading must report truncation, not
    # silently restore a partial model.
    var gpt = _tiny_gpt(1)
    var m = _zeros_state(gpt)
    var v = _zeros_state(gpt)
    var path = String("build/ckpt_full.ckpt")
    save_checkpoint(path, gpt, m, v, 0, 0)

    var content = open(path, "r").read()
    var raw = content.split("\n")
    # Keep only the first 40 lines (header + a few floats) — far short of the
    # full param+m+v payload.
    var kept = String("")
    for i in range(40):
        kept += String(raw[i]) + "\n"
    var tpath = String("build/ckpt_truncated.ckpt")
    with open(tpath, "w") as f:
        f.write(kept)

    var fresh = _tiny_gpt(1)
    with assert_raises(contains="truncated"):
        _ = load_checkpoint(tpath, fresh)


def test_save_rejects_misshaped_moment() raises:
    # A moment list with the right length but a wrong-shaped tensor would write a
    # payload that no longer lines up with the header shapes; save must reject it
    # rather than emit a file load would silently mis-read.
    var gpt = _tiny_gpt(1)
    var m = _zeros_state(gpt)
    var v = _zeros_state(gpt)
    m[0] = zeros_2d(1, 1)  # wrong shape for wte
    var path = String("build/ckpt_misshaped.ckpt")
    with assert_raises(contains="does not match parameter"):
        save_checkpoint(path, gpt, m, v, 0, 0)


def test_load_rejects_trailing_garbage() raises:
    # A file with extra value lines beyond params+m+v does not match this model;
    # load must reject it rather than ignore the tail.
    var gpt = _tiny_gpt(1)
    var m = _zeros_state(gpt)
    var v = _zeros_state(gpt)
    var path = String("build/ckpt_valid.ckpt")
    save_checkpoint(path, gpt, m, v, 0, 0)

    var content = open(path, "r").read()
    var gpath = String("build/ckpt_garbage.ckpt")
    with open(gpath, "w") as f:
        f.write(content + "0000000000000000\n")  # one extra value line

    var fresh = _tiny_gpt(1)
    with assert_raises(contains="extra value"):
        _ = load_checkpoint(gpath, fresh)


def test_resume_gate_bit_identical() raises:
    # THE resume gate. n steps straight must equal k steps + checkpoint + fresh
    # model + load + (n-k) steps, bit-for-bit. lr comes from the schedule off the
    # step index, so a mis-restored step counter would diverge here.
    var n = 6
    var k = 3
    var peak = 0.05
    var warmup = 2
    var min_lr = 0.005

    # Straight run.
    var full = _tiny_gpt(5)
    var m_full = _zeros_state(full)
    var v_full = _zeros_state(full)
    for s in range(1, n + 1):
        var lr = lr_at(s - 1, peak, warmup, n, min_lr)
        _adamw_step(full, m_full, v_full, s, lr)

    # Interrupted run, phase 1: k steps then checkpoint.
    var part = _tiny_gpt(5)  # same init as the straight run
    var m_part = _zeros_state(part)
    var v_part = _zeros_state(part)
    for s in range(1, k + 1):
        var lr = lr_at(s - 1, peak, warmup, n, min_lr)
        _adamw_step(part, m_part, v_part, s, lr)
    var path = String("build/ckpt_resume.ckpt")
    save_checkpoint(path, part, m_part, v_part, k, 0)

    # Phase 2: load into a fresh model, resume from the restored step counter.
    var resumed = _tiny_gpt(31)  # deliberately different init; load overwrites
    var state = load_checkpoint(path, resumed)
    assert_true(state.t == k, "resume step counter")
    var m_res = state.m.copy()
    var v_res = state.v.copy()
    for s in range(state.t + 1, n + 1):
        var lr = lr_at(s - 1, peak, warmup, n, min_lr)
        _adamw_step(resumed, m_res, v_res, s, lr)

    _assert_params_bit_identical(full, resumed, "resume gate")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
