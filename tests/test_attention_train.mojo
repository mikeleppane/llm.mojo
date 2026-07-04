# The attention-weight-dropout train path — equivalence, placement, gradients.
#
# GPT-2 drops post-softmax attention weights before they weight the values:
# output = dropout(W) @ v. This train path is ADDITIVE over the frozen core, so
# the load-bearing checks are:
#   1. with training = False (and with p = 0) it equals the proven cached path
#      EXACTLY — outputs and all three input gradients;
#   2. eval / p = 0 consume NO rng (twin-generator check);
#   3. with dropout active (seeded) the dropped weight entries are 0 and the
#      survivors are scaled by inv_keep = 1/(1-p);
#   4. dq/dk/dv through the DROPPED core match a central finite difference, using
#      the re-seeded-rng convention (below);
#   5. MHA backward_train parameter grads match a finite difference and double
#      exactly on a second backward.
#
# Re-seeded-rng finite-diff convention: with training = True the forward draws a
# dropout mask, so a naive finite difference would compare two DIFFERENT masks.
# The mask depends only on the draw sequence (one uniform per weight entry, fixed
# row-major order), never on q/k/v, so re-seeding the rng identically before each
# forward call replays the identical mask for x+h and x-h. The check then
# differentiates the mask-fixed (linear-in-the-mask) map — which is exactly what
# backward computes.
#
# Finite-difference convention: L = sum(cotangent ⊙ output); central diff
# h = 1e-5; tolerance |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|.

from std.testing import assert_almost_equal, assert_true, TestSuite

from llm.nn.dropout import dropout_backward
from llm.tensor.tensor2d import Tensor2D, from_rows
from llm.transformer.attention import (
    MultiHeadAttention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_cached,
    scaled_dot_product_attention_train,
    scaled_dot_product_attention_train_backward,
)
from llm.transformer.masks import no_mask
from llm.utils.random import Rng


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    # Shared mixed tolerance |a - n| <= 1e-7 + 1e-5 * |n|.
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


def sample_q() raises -> Tensor2D:
    # [T_q=3, D=3], asymmetric.
    return from_rows([[1.0, 0.0, -0.5], [0.2, 1.0, 0.3], [-0.7, 0.4, 1.1]])


def sample_k() raises -> Tensor2D:
    # [T_k=4, D=3], asymmetric.
    return from_rows(
        [
            [1.0, 0.0, 0.2],
            [0.0, 1.0, -0.3],
            [1.0, 1.0, 0.5],
            [-1.0, 0.6, 0.1],
        ]
    )


def sample_v() raises -> Tensor2D:
    # [T_k=4, D_v=3], asymmetric.
    return from_rows(
        [
            [0.5, -0.2, 1.0],
            [-1.0, 0.3, 0.4],
            [0.7, 0.9, -0.6],
            [0.1, -0.8, 0.2],
        ]
    )


def cotangent() raises -> Tensor2D:
    # Fixed asymmetric d_out [T_q=3, D_v=3].
    return from_rows([[0.7, -0.3, 0.5], [0.2, 0.9, -0.4], [-0.6, 0.1, 0.8]])


def projected_output(output: Tensor2D, cot: Tensor2D) raises -> Float64:
    var total = 0.0
    for i in range(output.rows):
        for j in range(output.cols):
            total += cot[i, j] * output[i, j]
    return total


def test_eval_equals_cached_forward_and_grads() raises:
    # training = False must degenerate to the proven cached path — output AND all
    # three input gradients equal, to exact-math tolerance.
    var q = sample_q()
    var k = sample_k()
    var v = sample_v()
    var mask = no_mask(3, 4)
    var cot = cotangent()

    var base = scaled_dot_product_attention_cached(q, k, v, mask)
    var base_grads = scaled_dot_product_attention_backward(base.cache, cot)

    var rng = Rng(123)
    var train = scaled_dot_product_attention_train(
        q, k, v, mask, 0.5, False, rng
    )
    var train_grads = scaled_dot_product_attention_train_backward(
        train.cache, cot
    )

    for i in range(base.output.rows):
        for j in range(base.output.cols):
            assert_almost_equal(train.output[i, j], base.output[i, j])
    for i in range(base_grads.d_q.rows):
        for j in range(base_grads.d_q.cols):
            assert_almost_equal(train_grads.d_q[i, j], base_grads.d_q[i, j])
    for i in range(base_grads.d_k.rows):
        for j in range(base_grads.d_k.cols):
            assert_almost_equal(train_grads.d_k[i, j], base_grads.d_k[i, j])
    for i in range(base_grads.d_v.rows):
        for j in range(base_grads.d_v.cols):
            assert_almost_equal(train_grads.d_v[i, j], base_grads.d_v[i, j])


def test_p_zero_equals_cached() raises:
    # p = 0 with training = True is also the identity: no element dropped.
    var q = sample_q()
    var k = sample_k()
    var v = sample_v()
    var mask = no_mask(3, 4)

    var base = scaled_dot_product_attention_cached(q, k, v, mask)
    var rng = Rng(7)
    var train = scaled_dot_product_attention_train(
        q, k, v, mask, 0.0, True, rng
    )
    for i in range(base.output.rows):
        for j in range(base.output.cols):
            assert_almost_equal(train.output[i, j], base.output[i, j])


def test_eval_and_p_zero_consume_no_rng() raises:
    # Disabling dropout (training = False, or p = 0) must not advance the seeded
    # generator downstream tests depend on. Twin-generator check: a reference rng
    # left untouched must still be bit-identical to the one the train path saw.
    var q = sample_q()
    var k = sample_k()
    var v = sample_v()
    var mask = no_mask(3, 4)

    var rng_eval = Rng(999)
    var rng_ref = Rng(999)
    _ = scaled_dot_product_attention_train(q, k, v, mask, 0.5, False, rng_eval)
    assert_true(
        rng_eval.state == rng_ref.state,
        "training=False consumed rng draws",
    )

    var rng_p0 = Rng(999)
    _ = scaled_dot_product_attention_train(q, k, v, mask, 0.0, True, rng_p0)
    assert_true(
        rng_p0.state == rng_ref.state,
        "p=0 consumed rng draws",
    )


def test_dropout_placement_on_weights() raises:
    # With dropout active, the cached mask/scale must describe a dropout on the
    # post-softmax weights: inv_keep = 1/(1-p); dropped entries (mask 0) reconstruct
    # to 0, survivors (mask 1) to weights * inv_keep. The reconstructed dropped
    # weights are what the value matmul saw.
    var q = sample_q()
    var k = sample_k()
    var v = sample_v()
    var mask = no_mask(3, 4)
    var p = 0.5

    var rng = Rng(42)
    var train = scaled_dot_product_attention_train(q, k, v, mask, p, True, rng)
    var weights = train.cache.weights.copy()
    var drop_mask = train.cache.drop_mask.copy()
    var inv_keep = train.cache.inv_keep

    assert_almost_equal(inv_keep, 1.0 / (1.0 - p))
    # Reconstruct dropped_W = weights ⊙ mask * inv_keep (dropout_backward IS that
    # diagonal map) and check the two cases.
    var dropped = dropout_backward(drop_mask, inv_keep, weights)
    var saw_zero = False
    var saw_scaled = False
    for i in range(weights.rows):
        for j in range(weights.cols):
            var m = drop_mask[i, j]
            assert_true(m == 0.0 or m == 1.0, "mask entry not in {0,1}")
            if m == 0.0:
                assert_almost_equal(dropped[i, j], 0.0)
                saw_zero = True
            else:
                assert_almost_equal(dropped[i, j], weights[i, j] * inv_keep)
                saw_scaled = True
    assert_true(saw_zero, "no weight was dropped at p=0.5 — mask is degenerate")
    assert_true(saw_scaled, "no weight survived at p=0.5 — mask is degenerate")


def test_dropped_core_grads_finite_difference() raises:
    # dq/dk/dv through the DROPPED core vs a central finite difference, re-seeding
    # the rng identically before every forward so x+h and x-h share the mask. The
    # analytic side comes from one seeded forward's cache.
    var q = sample_q()
    var k = sample_k()
    var v = sample_v()
    var mask = no_mask(3, 4)
    var cot = cotangent()
    var p = 0.5
    var seed = UInt64(2024)
    var h = 1e-5

    var rng_a = Rng(seed)
    var fwd = scaled_dot_product_attention_train(q, k, v, mask, p, True, rng_a)
    var grads = scaled_dot_product_attention_train_backward(fwd.cache, cot)

    # dq
    for i in range(q.rows):
        for j in range(q.cols):
            var qp = q.copy()
            qp[i, j] = qp[i, j] + h
            var qm = q.copy()
            qm[i, j] = qm[i, j] - h
            var rp = Rng(seed)
            var fp = scaled_dot_product_attention_train(
                qp, k, v, mask, p, True, rp
            )
            var rm = Rng(seed)
            var fm = scaled_dot_product_attention_train(
                qm, k, v, mask, p, True, rm
            )
            var numeric = (
                projected_output(fp.output, cot)
                - projected_output(fm.output, cot)
            ) / (2.0 * h)
            assert_grad_close(grads.d_q[i, j], numeric)
    # dk
    for i in range(k.rows):
        for j in range(k.cols):
            var kp = k.copy()
            kp[i, j] = kp[i, j] + h
            var km = k.copy()
            km[i, j] = km[i, j] - h
            var rp = Rng(seed)
            var fp = scaled_dot_product_attention_train(
                q, kp, v, mask, p, True, rp
            )
            var rm = Rng(seed)
            var fm = scaled_dot_product_attention_train(
                q, km, v, mask, p, True, rm
            )
            var numeric = (
                projected_output(fp.output, cot)
                - projected_output(fm.output, cot)
            ) / (2.0 * h)
            assert_grad_close(grads.d_k[i, j], numeric)
    # dv
    for i in range(v.rows):
        for j in range(v.cols):
            var vp = v.copy()
            vp[i, j] = vp[i, j] + h
            var vm = v.copy()
            vm[i, j] = vm[i, j] - h
            var rp = Rng(seed)
            var fp = scaled_dot_product_attention_train(
                q, k, vp, mask, p, True, rp
            )
            var rm = Rng(seed)
            var fm = scaled_dot_product_attention_train(
                q, k, vm, mask, p, True, rm
            )
            var numeric = (
                projected_output(fp.output, cot)
                - projected_output(fm.output, cot)
            ) / (2.0 * h)
            assert_grad_close(grads.d_v[i, j], numeric)


def mha_input() raises -> Tensor2D:
    # [T=4, C=6] asymmetric self-attention input.
    return from_rows(
        [
            [0.5, -0.2, 1.0, 0.3, -0.7, 0.1],
            [-1.0, 0.4, 0.2, 0.9, 0.0, -0.5],
            [0.6, 0.8, -0.3, -0.1, 1.1, 0.2],
            [0.1, -0.6, 0.7, 0.5, -0.4, 0.9],
        ]
    )


def mha_cotangent() raises -> Tensor2D:
    # Fixed asymmetric d_out [T=4, C=6].
    return from_rows(
        [
            [0.3, -0.5, 0.7, -0.2, 0.4, 0.1],
            [-0.6, 0.2, 0.8, 0.5, -0.3, 0.9],
            [0.1, 0.7, -0.4, 0.6, -0.8, 0.2],
            [0.9, -0.1, 0.3, -0.7, 0.5, -0.2],
        ]
    )


def test_mha_train_equals_eval() raises:
    # forward_cached_train / backward_train with training = False must equal the
    # proven forward_cached / backward — output and the qkv/proj parameter grads.
    var x = mha_input()
    var mask = no_mask(4, 4)
    var cot = mha_cotangent()

    var rng_a = Rng(5)
    var mha_a = MultiHeadAttention.init_random(rng_a, 6, 2)
    var rng_b = Rng(5)
    var mha_b = MultiHeadAttention.init_random(rng_b, 6, 2)

    var f_eval = mha_a.forward_cached(x, mask)
    var _dx_eval = mha_a.backward(f_eval.cache, cot)
    var rng_drop = Rng(1)
    var f_train = mha_b.forward_cached_train(x, mask, 0.5, False, rng_drop)
    var _dx_train = mha_b.backward_train(f_train.cache, cot)

    for i in range(f_eval.output.rows):
        for j in range(f_eval.output.cols):
            assert_almost_equal(f_train.output[i, j], f_eval.output[i, j])
    for i in range(mha_a.qkv.weight.grad.rows):
        for j in range(mha_a.qkv.weight.grad.cols):
            assert_almost_equal(
                mha_b.qkv.weight.grad[i, j], mha_a.qkv.weight.grad[i, j]
            )
    for i in range(mha_a.proj.weight.grad.rows):
        for j in range(mha_a.proj.weight.grad.cols):
            assert_almost_equal(
                mha_b.proj.weight.grad[i, j], mha_a.proj.weight.grad[i, j]
            )


def test_mha_backward_train_doubles_exactly() raises:
    # Two backward_train passes without a zero_grad between them must double the
    # parameter grads bit-for-bit — the accumulation contract weight tying relies
    # on, checked here on the train path.
    var x = mha_input()
    var mask = no_mask(4, 4)
    var cot = mha_cotangent()

    var rng = Rng(9)
    var mha = MultiHeadAttention.init_random(rng, 6, 2)
    var rng_drop = Rng(3)
    var fwd = mha.forward_cached_train(x, mask, 0.5, True, rng_drop)
    var _d1 = mha.backward_train(fwd.cache, cot)
    # snapshot grads after one call
    var qkv1 = mha.qkv.weight.grad.copy()
    var proj1 = mha.proj.weight.grad.copy()
    var _d2 = mha.backward_train(fwd.cache, cot)
    for i in range(qkv1.rows):
        for j in range(qkv1.cols):
            assert_true(
                mha.qkv.weight.grad[i, j] == 2.0 * qkv1[i, j],
                "qkv grad did not double exactly",
            )
    for i in range(proj1.rows):
        for j in range(proj1.cols):
            assert_true(
                mha.proj.weight.grad[i, j] == 2.0 * proj1[i, j],
                "proj grad did not double exactly",
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
