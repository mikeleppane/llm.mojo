# Tests for the GPT parameter-walk-as-registry (parameter_shapes, decay flags,
# grad_norm, scale_grads, export/import, apply_adamw). The walk order is a
# load-bearing contract: the optimizer, gradient clipping, and the checkpoint all
# index the same documented traversal, and drift between walk methods is the
# named failure mode these tests exist to catch.
#
# The headline check is the against-oracle 2-step run: gpt.apply_adamw is compared
# to a FLAT reference optimizer that drives the oracle-verified adamw_update over
# the model's OWN published walk metadata (export_parameters, export_gradients,
# parameter_decay_flags) in a plain index loop. If apply_adamw visits parameters
# in a different order, or assigns a decay flag inconsistent with
# parameter_decay_flags, or threads its m/v state off-by-one, the two diverge.
# adamw_update is the only shared code, and it is independently pinned against a
# NumPy oracle in test_adamw.mojo.

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.config import GPTConfig
from llm.nn.optim import adamw_update
from llm.nn.parameter import Parameter
from llm.tensor.ops import cross_entropy_rows_backward
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.gpt import GPT
from llm.utils.random import Rng

comptime B1 = 0.9
comptime B2 = 0.95
comptime EPS = 1e-8


def _tiny_gpt(seed: UInt64) raises -> GPT:
    # V=8, C=8, context=8, L=2, H=2, dropout 0. 28 parameter tensors:
    # wte, wpe, 12 per block x 2, ln_f weight+bias.
    var cfg = GPTConfig(8, 8, 8, 2, 2, 0.0)
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


def _backward_into(mut gpt: GPT) raises:
    # Populate every gradient with a real backward pass (dropout off, so no rng
    # is drawn). Grads are nonzero across the model.
    var rng = Rng(0)
    gpt.zero_grad()
    var fwd = gpt.forward_cached(_ids(), False, rng)
    var d_logits = cross_entropy_rows_backward(fwd.logits, _targets())
    gpt.backward(fwd.cache, d_logits)


def _set_grad(mut p: Parameter, g: Tensor2D):
    for i in range(g.rows):
        for j in range(g.cols):
            p.grad[i, j] = g[i, j]


def test_parameter_shapes_matches_walk() raises:
    var gpt = _tiny_gpt(3)
    var shapes = gpt.parameter_shapes()
    # 28 tensors: 2 embeddings + 12*2 block params + 2 ln_f.
    assert_true(len(shapes) == 28, "expected 28 parameter tensors")
    assert_true(len(shapes) == gpt.parameter_tensor_count(), "count mismatch")
    # wte and wpe are both [V=8, C=8] / [context=8, C=8].
    assert_true(shapes[0].rows == 8 and shapes[0].cols == 8, "wte shape")
    assert_true(shapes[1].rows == 8 and shapes[1].cols == 8, "wpe shape")
    # The float total over the walk must reconcile with parameter_count_actual
    # (which counts wte exactly once) — a double-counted wte would inflate both.
    var total = 0
    for k in range(len(shapes)):
        total += shapes[k].rows * shapes[k].cols
    assert_true(
        total == gpt.parameter_count_actual(),
        "walked float total " + String(total) + " != parameter_count_actual",
    )


def test_decay_partition_inventory() raises:
    # The GPT-family selective-decay partition, pinned as an explicit inventory:
    # embeddings decay; every bias and LayerNorm vector does not. Per block: 4
    # decayed matrices (qkv, proj, mlp up, mlp down) and 8 undecayed (4 biases +
    # 4 LN vectors). Plus wte, wpe decayed and ln_f (weight, bias) undecayed.
    var gpt = _tiny_gpt(3)
    var flags = gpt.parameter_decay_flags()
    assert_true(len(flags) == 28, "flags length")

    var decayed = 0
    for k in range(len(flags)):
        if flags[k]:
            decayed += 1
    # 2 embeddings + 4 matrices * 2 blocks = 10 decayed; 18 undecayed.
    assert_true(
        decayed == 10, "expected 10 decayed tensors, got " + String(decayed)
    )

    # Embeddings decay; ln_f does not.
    assert_true(flags[0], "wte must decay")
    assert_true(flags[1], "wpe must decay")
    assert_true(not flags[26], "ln_f.weight must not decay")
    assert_true(not flags[27], "ln_f.bias must not decay")

    # Block 0 occupies indices 2..13 in walk order: ln1 w/b, qkv w/b, proj w/b,
    # ln2 w/b, up w/b, down w/b. Pin the exact matrix/vector pattern.
    assert_true(not flags[2], "ln1.weight")
    assert_true(not flags[3], "ln1.bias")
    assert_true(flags[4], "qkv.weight decays")
    assert_true(not flags[5], "qkv.bias")
    assert_true(flags[6], "proj.weight decays")
    assert_true(not flags[7], "proj.bias")
    assert_true(not flags[8], "ln2.weight")
    assert_true(not flags[9], "ln2.bias")
    assert_true(flags[10], "mlp.up.weight decays")
    assert_true(not flags[11], "mlp.up.bias")
    assert_true(flags[12], "mlp.down.weight decays")
    assert_true(not flags[13], "mlp.down.bias")


def test_apply_adamw_moves_every_parameter() raises:
    # With real (nonzero) gradients, one AdamW step moves EVERY parameter tensor.
    # A Parameter the walk skipped would come back byte-for-byte unchanged.
    var gpt = _tiny_gpt(3)
    _backward_into(gpt)
    var before = gpt.export_parameters()
    var shapes = gpt.parameter_shapes()
    var m = List[Tensor2D]()
    var v = List[Tensor2D]()
    for k in range(len(shapes)):
        m.append(zeros_2d(shapes[k].rows, shapes[k].cols))
        v.append(zeros_2d(shapes[k].rows, shapes[k].cols))

    gpt.apply_adamw(m, v, 1, 0.02, B1, B2, EPS, 0.1)
    var after = gpt.export_parameters()

    for k in range(len(after)):
        var moved = False
        for i in range(after[k].rows):
            for j in range(after[k].cols):
                if after[k][i, j] != before[k][i, j]:
                    moved = True
        assert_true(moved, "parameter " + String(k) + " did not move")
        # m/v advanced and kept their shape.
        assert_true(
            m[k].rows == shapes[k].rows and m[k].cols == shapes[k].cols,
            "m shape drift at " + String(k),
        )


def test_apply_adamw_length_mismatch_raises() raises:
    var gpt = _tiny_gpt(3)
    _backward_into(gpt)
    var m = List[Tensor2D]()  # empty — wrong length
    var v = List[Tensor2D]()
    with assert_raises(contains="m/v must have"):
        gpt.apply_adamw(m, v, 1, 0.02, B1, B2, EPS, 0.1)


def test_apply_adamw_matches_flat_walk_two_steps() raises:
    # The against-oracle. A flat reference optimizer applies adamw_update to
    # copies of the model's parameters, driven purely by the published walk
    # metadata (export order + decay flags), and must agree with gpt.apply_adamw
    # after each of two steps. Two steps so the second exercises non-zero m/v
    # (an off-by-one in m/v threading is invisible while all state is zero).
    var gpt = _tiny_gpt(3)
    var shapes = gpt.parameter_shapes()
    var flags = gpt.parameter_decay_flags()
    var n = len(shapes)

    # Flat reference: one Parameter per exported value, independent m/v.
    var ref_params = List[Parameter]()
    var init_vals = gpt.export_parameters()
    for k in range(n):
        ref_params.append(Parameter(init_vals[k].copy()))
    var m_ref = List[Tensor2D]()
    var v_ref = List[Tensor2D]()
    var m_model = List[Tensor2D]()
    var v_model = List[Tensor2D]()
    for k in range(n):
        m_ref.append(zeros_2d(shapes[k].rows, shapes[k].cols))
        v_ref.append(zeros_2d(shapes[k].rows, shapes[k].cols))
        m_model.append(zeros_2d(shapes[k].rows, shapes[k].cols))
        v_model.append(zeros_2d(shapes[k].rows, shapes[k].cols))

    var lr = 0.02
    var wd = 0.1
    for step in range(1, 3):
        _backward_into(gpt)  # grads for the model's CURRENT state
        var grads = gpt.export_gradients()  # walk order

        # Reference: index-driven, decay per the published flag.
        for k in range(n):
            _set_grad(ref_params[k], grads[k])
            var decay = wd if flags[k] else 0.0
            adamw_update(
                ref_params[k], m_ref[k], v_ref[k], step, lr, B1, B2, EPS, decay
            )

        # Model: threaded through the walk.
        gpt.apply_adamw(m_model, v_model, step, lr, B1, B2, EPS, wd)

        # They must agree, tensor by tensor, after this step.
        var model_vals = gpt.export_parameters()
        for k in range(n):
            for i in range(model_vals[k].rows):
                for j in range(model_vals[k].cols):
                    assert_almost_equal(
                        model_vals[k][i, j],
                        ref_params[k].value[i, j],
                        atol=1e-12,
                    )


def test_walk_methods_are_stable() raises:
    # Two walks agree: the order and shapes are a fixed function of the model, not
    # of call history. A drifting walk would return different lists per call.
    var gpt = _tiny_gpt(7)
    var a = gpt.parameter_shapes()
    var b = gpt.parameter_shapes()
    assert_true(len(a) == len(b), "shape count drift")
    for k in range(len(a)):
        assert_true(a[k].rows == b[k].rows and a[k].cols == b[k].cols, "shape")
    var fa = gpt.parameter_decay_flags()
    var fb = gpt.parameter_decay_flags()
    for k in range(len(fa)):
        assert_true(fa[k] == fb[k], "decay flag drift at " + String(k))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
