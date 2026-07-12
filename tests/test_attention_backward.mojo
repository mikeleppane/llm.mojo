"""Finite-difference and no-leak tests for attention backward.

The core (scaled_dot_product_attention_backward) is checked for dq/dk/dv under
no_mask and causal_mask, and with T_q != T_k (the cross-attention shape). Masking
adds a decisive check: a key blocked from every query must receive zero gradient,
because the softmax backward multiplies by a ~0 weight. MHA (forward_cached/
backward) is checked end to end: dx by finite difference and the fused-qkv and
proj parameter grads against a rebuilt forward.

Finite-difference convention: L = sum(cotangent ⊙ output); central diff h = 1e-5;
mixed tolerance |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|. Core T=4, D=3;
MHA T=4, C=6, H=2 (D=3).
"""

from std.testing import assert_almost_equal, assert_true, TestSuite

from llm.nn.linear import Linear
from llm.nn.parameter import Parameter
from llm.transformer.attention import (
    MultiHeadAttention,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_cached,
)
from llm.transformer.masks import causal_mask, key_padding_mask, no_mask
from llm.tensor.tensor2d import Tensor2D, from_rows
from llm.utils.random import Rng


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    """Assert |analytic - numeric| <= 1e-7 + 1e-5 * |numeric| (mixed tolerance).
    """
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


def sum_product(cot: Tensor2D, output: Tensor2D) -> Float64:
    var total = 0.0
    for i in range(output.rows):
        for j in range(output.cols):
            total += cot[i, j] * output[i, j]
    return total


def core_loss(
    q: Tensor2D, k: Tensor2D, v: Tensor2D, mask: Tensor2D, cot: Tensor2D
) raises -> Float64:
    return sum_product(cot, scaled_dot_product_attention(q, k, v, mask).output)


def check_core(
    q: Tensor2D, k: Tensor2D, v: Tensor2D, mask: Tensor2D, cot: Tensor2D
) raises:
    """Finite-difference dq, dk, dv for the given q, k, v, mask vs the analytic
    core backward."""
    var fwd = scaled_dot_product_attention_cached(q, k, v, mask)
    var grads = scaled_dot_product_attention_backward(fwd.cache, cot)
    var h = 1e-5

    for i in range(q.rows):
        for j in range(q.cols):
            var plus = q.copy()
            plus[i, j] = plus[i, j] + h
            var minus = q.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                core_loss(plus, k, v, mask, cot)
                - core_loss(minus, k, v, mask, cot)
            ) / (2.0 * h)
            assert_grad_close(grads.d_q[i, j], numeric)

    for i in range(k.rows):
        for j in range(k.cols):
            var plus = k.copy()
            plus[i, j] = plus[i, j] + h
            var minus = k.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                core_loss(q, plus, v, mask, cot)
                - core_loss(q, minus, v, mask, cot)
            ) / (2.0 * h)
            assert_grad_close(grads.d_k[i, j], numeric)

    for i in range(v.rows):
        for j in range(v.cols):
            var plus = v.copy()
            plus[i, j] = plus[i, j] + h
            var minus = v.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                core_loss(q, k, plus, mask, cot)
                - core_loss(q, k, minus, mask, cot)
            ) / (2.0 * h)
            assert_grad_close(grads.d_v[i, j], numeric)


def square_qkv() raises -> List[Tensor2D]:
    """q, k, v each [T=4, D=3], asymmetric; returned as [q, k, v]."""
    var q = from_rows(
        [[0.5, -1.0, 2.0], [-0.3, 0.8, -2.1], [1.1, 0.0, -0.7], [0.2, 1.3, 0.4]]
    )
    var k = from_rows(
        [[0.9, 0.1, -0.6], [1.2, -0.4, 0.3], [-0.8, 0.5, 1.0], [0.0, -1.1, 0.7]]
    )
    var v = from_rows(
        [[0.3, -0.5, 0.8], [-1.0, 0.6, 0.2], [0.7, 0.9, -0.4], [0.1, -0.2, 1.4]]
    )
    var out = List[Tensor2D]()
    out.append(q^)
    out.append(k^)
    out.append(v^)
    return out^


def square_cotangent() raises -> Tensor2D:
    """Fixed asymmetric d_out [T=4, D_v=3]."""
    return from_rows(
        [[0.7, -0.2, 1.3], [0.1, 0.9, -1.1], [-0.6, 0.3, 0.2], [0.4, -0.5, 0.8]]
    )


def test_core_backward_no_mask() raises:
    """Core dq/dk/dv match finite differences with no mask."""
    var qkv = square_qkv()
    check_core(qkv[0], qkv[1], qkv[2], no_mask(4, 4), square_cotangent())


def test_core_backward_causal_mask() raises:
    """Core dq/dk/dv match finite differences under a causal mask."""
    var qkv = square_qkv()
    check_core(qkv[0], qkv[1], qkv[2], causal_mask(4), square_cotangent())


def test_core_backward_cross_shape() raises:
    """Core grads match finite differences at the cross shape T_q=3, T_k=5, D=3,
    D_v=4."""
    var q = from_rows([[0.5, -1.0, 2.0], [-0.3, 0.8, -2.1], [1.1, 0.0, -0.7]])
    var k = from_rows(
        [
            [0.9, 0.1, -0.6],
            [1.2, -0.4, 0.3],
            [-0.8, 0.5, 1.0],
            [0.0, -1.1, 0.7],
            [0.6, 0.2, -0.9],
        ]
    )
    var v = from_rows(
        [
            [0.3, -0.5, 0.8, 0.2],
            [-1.0, 0.6, 0.2, -0.7],
            [0.7, 0.9, -0.4, 1.1],
            [0.1, -0.2, 1.4, 0.5],
            [-0.6, 0.4, 0.0, -0.3],
        ]
    )
    var cot = from_rows(
        [[0.7, -0.2, 1.3, -0.5], [0.1, 0.9, -1.1, 0.4], [-0.6, 0.3, 0.2, -0.8]]
    )
    check_core(q, k, v, no_mask(3, 5), cot)


def test_no_gradient_leaks_into_blocked_key() raises:
    """A key blocked from every query gets ~0 value and key gradient (its softmax
    weight column is ~0)."""
    var qkv = square_qkv()
    var keep = [True, True, False, True]  # key 2 blocked
    var mask = key_padding_mask(keep, 4)  # [4, 4]
    var fwd = scaled_dot_product_attention_cached(qkv[0], qkv[1], qkv[2], mask)
    var grads = scaled_dot_product_attention_backward(
        fwd.cache, square_cotangent()
    )
    for j in range(grads.d_k.cols):
        assert_almost_equal(grads.d_k[2, j], 0.0, atol=1e-6)
    for j in range(grads.d_v.cols):
        assert_almost_equal(grads.d_v[2, j], 0.0, atol=1e-6)


# --- MHA ---


def build_mha(
    qkv_w: Tensor2D,
    qkv_b: Tensor2D,
    proj_w: Tensor2D,
    proj_b: Tensor2D,
    n_heads: Int,
) raises -> MultiHeadAttention:
    var qkv = Linear(Parameter(qkv_w.copy()), Parameter(qkv_b.copy()))
    var proj = Linear(Parameter(proj_w.copy()), Parameter(proj_b.copy()))
    return MultiHeadAttention(qkv^, proj^, n_heads)


def mha_input() raises -> Tensor2D:
    """Asymmetric MHA input [T=4, C=6]."""
    return from_rows(
        [
            [1.0, 0.5, -1.0, 0.3, 0.8, -0.2],
            [0.2, -0.4, 0.9, -1.1, 0.1, 0.7],
            [-0.7, 1.2, 0.1, 0.6, -0.9, 0.4],
            [0.5, -0.3, 0.2, 1.0, -0.6, 0.1],
        ]
    )


def mha_cotangent() raises -> Tensor2D:
    return from_rows(
        [
            [0.7, -0.2, 1.3, -0.5, 0.4, 0.1],
            [0.1, 0.9, -1.1, 0.4, -0.3, 0.6],
            [-0.6, 0.3, 0.2, -0.8, 0.5, -0.1],
            [0.2, -0.7, 0.9, 0.3, -0.4, 0.8],
        ]
    )


def mha_base_weights() raises -> List[Tensor2D]:
    """Deterministic seeded weights for C=6, H=2: [qkv_w [18,6], qkv_b [1,18],
    proj_w [6,6], proj_b [1,6]]."""
    var rng = Rng(23)
    var base = MultiHeadAttention.init_random(rng, 6, 2)
    var out = List[Tensor2D]()
    out.append(base.qkv.weight.value.copy())
    out.append(base.qkv.bias.value.copy())
    out.append(base.proj.weight.value.copy())
    out.append(base.proj.bias.value.copy())
    return out^


def mha_projected(
    mha: MultiHeadAttention, x: Tensor2D, mask: Tensor2D, cot: Tensor2D
) raises -> Float64:
    return sum_product(cot, mha.forward(x, mask))


def test_mha_backward_d_x() raises:
    """MHA d_x matches a central finite difference."""
    var w = mha_base_weights()
    var mha = build_mha(w[0], w[1], w[2], w[3], 2)
    var x = mha_input()
    var cot = mha_cotangent()
    var mask = causal_mask(4)
    var fwd = mha.forward_cached(x.copy(), mask)
    var d_x = mha.backward(fwd.cache, cot)

    var h = 1e-5
    for i in range(x.rows):
        for j in range(x.cols):
            var plus = x.copy()
            plus[i, j] = plus[i, j] + h
            var minus = x.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                mha_projected(mha, plus, mask, cot)
                - mha_projected(mha, minus, mask, cot)
            ) / (2.0 * h)
            assert_grad_close(d_x[i, j], numeric)


def test_mha_backward_parameter_grads() raises:
    """MHA qkv/proj weight and bias grads match finite differences."""
    var w = mha_base_weights()
    var mha = build_mha(w[0], w[1], w[2], w[3], 2)
    var x = mha_input()
    var cot = mha_cotangent()
    var mask = causal_mask(4)
    mha.qkv.weight.zero_grad()
    mha.qkv.bias.zero_grad()
    mha.proj.weight.zero_grad()
    mha.proj.bias.zero_grad()
    var fwd = mha.forward_cached(x.copy(), mask)
    _ = mha.backward(fwd.cache, cot)

    var h = 1e-5
    # Fused qkv weight [3C, C] = [18, 6].
    for r in range(mha.qkv.weight.value.rows):
        for c in range(mha.qkv.weight.value.cols):
            var wp = w[0].copy()
            wp[r, c] = wp[r, c] + h
            var wm = w[0].copy()
            wm[r, c] = wm[r, c] - h
            var numeric = (
                mha_projected(build_mha(wp, w[1], w[2], w[3], 2), x, mask, cot)
                - mha_projected(
                    build_mha(wm, w[1], w[2], w[3], 2), x, mask, cot
                )
            ) / (2.0 * h)
            assert_grad_close(mha.qkv.weight.grad[r, c], numeric)
    # proj weight [C, C] = [6, 6].
    for r in range(mha.proj.weight.value.rows):
        for c in range(mha.proj.weight.value.cols):
            var wp = w[2].copy()
            wp[r, c] = wp[r, c] + h
            var wm = w[2].copy()
            wm[r, c] = wm[r, c] - h
            var numeric = (
                mha_projected(build_mha(w[0], w[1], wp, w[3], 2), x, mask, cot)
                - mha_projected(
                    build_mha(w[0], w[1], wm, w[3], 2), x, mask, cot
                )
            ) / (2.0 * h)
            assert_grad_close(mha.proj.weight.grad[r, c], numeric)
    # qkv bias [1, 18] and proj bias [1, 6].
    for c in range(mha.qkv.bias.value.cols):
        var bp = w[1].copy()
        bp[0, c] = bp[0, c] + h
        var bm = w[1].copy()
        bm[0, c] = bm[0, c] - h
        var numeric = (
            mha_projected(build_mha(w[0], bp, w[2], w[3], 2), x, mask, cot)
            - mha_projected(build_mha(w[0], bm, w[2], w[3], 2), x, mask, cot)
        ) / (2.0 * h)
        assert_grad_close(mha.qkv.bias.grad[0, c], numeric)
    for c in range(mha.proj.bias.value.cols):
        var bp = w[3].copy()
        bp[0, c] = bp[0, c] + h
        var bm = w[3].copy()
        bm[0, c] = bm[0, c] - h
        var numeric = (
            mha_projected(build_mha(w[0], w[1], w[2], bp, 2), x, mask, cot)
            - mha_projected(build_mha(w[0], w[1], w[2], bm, 2), x, mask, cot)
        ) / (2.0 * h)
        assert_grad_close(mha.proj.bias.grad[0, c], numeric)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
