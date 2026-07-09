# Tests for CrossMultiHeadAttention — the lab's separate-q + fused-kv cross
# attention. The forward golden (Case CA) comes from
# tests/oracles/encdec_reference.py, run once and frozen here; it uses T_q=3 !=
# T_k=4 so a transposed mask or a [T_k, ...] output shape fails it. The rest are
# structural (parameter count 4C²+4C, shape contract, init determinism, invalid
# config) and the finite-difference backward checks for BOTH input gradients
# (d_x through q, d_memory through the fused kv) and all six parameter grads,
# plus exact accumulation doubling.
#
# Finite-difference convention (Part XI, inline per file — no shared closure):
#   projected scalar loss L = sum(cotangent ⊙ forward), central diff h = 1e-5,
#   mixed tolerance |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|. A failing
#   check indicts the wiring first, the test second, the tolerance never.

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.lab.cross_attention import CrossMultiHeadAttention
from llm.nn.linear import Linear
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.attention import scaled_dot_product_attention
from llm.transformer.masks import no_mask
from llm.utils.random import Rng


def mat(flat: List[Float64], rows: Int, cols: Int) raises -> Tensor2D:
    # Build a [rows, cols] tensor from a flat row-major list (the shape the
    # oracle prints). Raises if the list length disagrees with rows*cols.
    if len(flat) != rows * cols:
        raise Error("mat: length mismatch")
    var t = zeros_2d(rows, cols)
    for r in range(rows):
        for c in range(cols):
            t[r, c] = flat[r * cols + c]
    return t^


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    # Part XI mixed tolerance |a - n| <= 1e-7 + 1e-5 * |n|.
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


def assert_equal_exact(a: Float64, b: Float64) raises:
    assert_true(
        a == b,
        String("expected exact equality, got ")
        + String(a)
        + " vs "
        + String(b),
    )


# --- Case CA golden data (frozen from tests/oracles/encdec_reference.py) ---


def ca_x() raises -> Tensor2D:
    return mat(
        [
            -0.3204,
            0.3832,
            0.0393,
            -0.114,
            0.3136,
            -0.3927,
            0.0519,
            -0.0889,
            0.3177,
            0.0061,
            0.3372,
            -0.1616,
        ],
        3,
        4,
    )


def ca_mem() raises -> Tensor2D:
    return mat(
        [
            -0.1471,
            0.5311,
            0.7023,
            -0.1135,
            -0.1597,
            -0.1448,
            -0.0068,
            -0.5188,
            -0.0503,
            -0.4941,
            0.4217,
            -0.3924,
            0.1293,
            0.0904,
            -0.2824,
            -0.0354,
        ],
        4,
        4,
    )


def ca_weights() raises -> List[Tensor2D]:
    # [q_w, q_b, kv_w, kv_b, proj_w, proj_b] as tensors (biases as [1, out]).
    var out = List[Tensor2D]()
    out.append(
        mat(
            [
                -0.0804,
                0.3443,
                -0.0653,
                -0.4186,
                0.2351,
                -0.0356,
                0.2257,
                0.0698,
                0.0131,
                0.0048,
                -0.317,
                0.3282,
                -0.3255,
                0.4082,
                -0.1748,
                -0.1512,
            ],
            4,
            4,
        )
    )
    out.append(mat([-0.0092, -0.0708, 0.1278, 0.0641], 1, 4))
    out.append(
        mat(
            [
                -0.1021,
                -0.3029,
                0.2731,
                -0.2868,
                0.2284,
                -0.0093,
                0.1623,
                0.4378,
                -0.1523,
                0.7986,
                0.1249,
                -0.2649,
                0.045,
                0.1626,
                0.0127,
                0.4899,
                -0.6157,
                0.0279,
                0.0754,
                -0.1525,
                0.0493,
                -0.4259,
                -0.0518,
                -0.3613,
                -0.1536,
                -0.1946,
                0.3935,
                0.0355,
                -0.1871,
                0.2737,
                0.2496,
                -0.1131,
            ],
            8,
            4,
        )
    )
    out.append(
        mat(
            [
                0.4064,
                0.1175,
                -0.7628,
                -0.1361,
                -0.0793,
                -0.5755,
                0.0967,
                0.3015,
            ],
            1,
            8,
        )
    )
    out.append(
        mat(
            [
                0.0241,
                -0.0904,
                -0.5151,
                -0.1268,
                0.022,
                0.2417,
                -0.4677,
                0.5209,
                0.0552,
                -0.2255,
                0.0717,
                0.0048,
                0.6042,
                0.1035,
                0.196,
                0.3175,
            ],
            4,
            4,
        )
    )
    out.append(mat([0.4657, 0.0233, 0.2724, -0.1803], 1, 4))
    return out^


def build_cmha(
    w: List[Tensor2D], n_heads: Int
) raises -> CrossMultiHeadAttention:
    # Assemble a cross-MHA from [q_w, q_b, kv_w, kv_b, proj_w, proj_b], copied in
    # so a finite-difference loop can perturb any entry and rebuild.
    var q = Linear(Parameter(w[0].copy()), Parameter(w[1].copy()))
    var kv = Linear(Parameter(w[2].copy()), Parameter(w[3].copy()))
    var proj = Linear(Parameter(w[4].copy()), Parameter(w[5].copy()))
    return CrossMultiHeadAttention(q^, kv^, proj^, n_heads)


def cotangent() raises -> Tensor2D:
    # Fixed asymmetric d_out [T_q=3, C=4].
    return mat(
        [0.7, -0.2, 1.3, -0.5, 0.1, 0.9, -1.1, 0.4, -0.6, 0.3, 0.2, -0.8], 3, 4
    )


def projected(
    layer: CrossMultiHeadAttention,
    x: Tensor2D,
    mem: Tensor2D,
    cot: Tensor2D,
) raises -> Float64:
    var y = layer.forward(x, mem, no_mask(x.rows, mem.rows))
    var total = 0.0
    for i in range(y.rows):
        for j in range(y.cols):
            total += cot[i, j] * y[i, j]
    return total


def test_case_ca_forward_oracle() raises:
    # Golden Case CA: cross-MHA forward, T_q=3, T_k=4, H=2, C=4, no mask. The
    # cross shape (T_q != T_k) proves the separate q vs k/v lengths work; the
    # output must be [T_q, C] = [3, 4].
    var layer = build_cmha(ca_weights(), 2)
    var x = ca_x()
    var mem = ca_mem()
    var y = layer.forward(x, mem, no_mask(3, 4))
    assert_true(y.rows == 3 and y.cols == 4)
    var expected = [
        0.3680494806579866,
        0.030307425750209664,
        0.39731986451514445,
        -0.06109958630471926,
        0.3678789896222456,
        0.02502263885337376,
        0.3992904024800289,
        -0.06469207094118723,
        0.3684369429710258,
        0.024798218249970774,
        0.39876365082199616,
        -0.06463070251933514,
    ]
    for i in range(3):
        for j in range(4):
            assert_almost_equal(y[i, j], expected[i * 4 + j], atol=1e-12)


def test_shape_contract() raises:
    # Output is [T_q, C] regardless of T_k; a swapped-length bug would surface as
    # [T_k, C]. Here T_q=2, T_k=5.
    var rng = Rng(3)
    var layer = CrossMultiHeadAttention.init_random(rng, 4, 2)
    var x = zeros_2d(2, 4)
    var mem = zeros_2d(5, 4)
    var y = layer.forward(x, mem, no_mask(2, 5))
    assert_equal(y.rows, 2)
    assert_equal(y.cols, 4)


def test_parameter_count_is_4c2_plus_4c() raises:
    # The lab's one pinned count: q(C²+C) + kv(2C²+2C) + proj(C²+C) = 4C²+4C, the
    # same total as self-MHA, summed over the layer's REAL Parameter tensors.
    var c = 8
    var rng = Rng(1)
    var layer = CrossMultiHeadAttention.init_random(rng, c, 2)
    var total = (
        layer.q.weight.value.size()
        + layer.q.bias.value.size()
        + layer.kv.weight.value.size()
        + layer.kv.bias.value.size()
        + layer.proj.weight.value.size()
        + layer.proj.bias.value.size()
    )
    assert_equal(total, 4 * c * c + 4 * c)


def test_init_determinism() raises:
    # Same seed reproduces the same layer (draw order q, kv, proj is pinned).
    var a = Rng(42)
    var b = Rng(42)
    var la = CrossMultiHeadAttention.init_random(a, 4, 2)
    var lb = CrossMultiHeadAttention.init_random(b, 4, 2)
    for r in range(la.kv.weight.value.rows):
        for c in range(la.kv.weight.value.cols):
            assert_equal(la.kv.weight.value[r, c], lb.kv.weight.value[r, c])


def test_invalid_config_raises() raises:
    var rng = Rng(1)
    with assert_raises():
        _ = CrossMultiHeadAttention.init_random(rng, 4, 0)  # n_heads = 0
    with assert_raises():
        _ = CrossMultiHeadAttention.init_random(rng, 6, 4)  # 6 % 4 != 0


def test_single_head_equals_core_and_projections() raises:
    # With H=1 the head loop does nothing structural: the layer is exactly
    # proj(sdpa(q(x), k, v)) with k, v the two halves of kv(memory). Recomputing
    # that path by hand and matching pins that the fused-kv split feeds the core
    # the right columns.
    var rng = Rng(9)
    var layer = CrossMultiHeadAttention.init_random(rng, 4, 1)
    var x = ca_x()
    var mem = ca_mem()
    var y = layer.forward(x, mem, no_mask(3, 4))

    var q_all = layer.q.forward(x)  # [3, 4]
    var kv_all = layer.kv.forward(mem)  # [4, 8]
    var k_all = zeros_2d(4, 4)
    var v_all = zeros_2d(4, 4)
    for r in range(4):
        for c in range(4):
            k_all[r, c] = kv_all[r, c]
            v_all[r, c] = kv_all[r, c + 4]
    var core = scaled_dot_product_attention(q_all, k_all, v_all, no_mask(3, 4))
    var expected = layer.proj.forward(core.output)
    for i in range(3):
        for j in range(4):
            assert_almost_equal(y[i, j], expected[i, j], atol=1e-12)


def test_d_x_matches_finite_difference() raises:
    var w = ca_weights()
    var layer = build_cmha(w, 2)
    var x = ca_x()
    var mem = ca_mem()
    var cot = cotangent()
    var fwd = layer.forward_cached(x.copy(), mem.copy(), no_mask(3, 4))
    var grads = layer.backward(fwd.cache, cot)

    var h = 1e-5
    for i in range(x.rows):
        for j in range(x.cols):
            var plus = x.copy()
            plus[i, j] = plus[i, j] + h
            var minus = x.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                projected(layer, plus, mem, cot)
                - projected(layer, minus, mem, cot)
            ) / (2.0 * h)
            assert_grad_close(grads.d_x[i, j], numeric)


def test_d_memory_matches_finite_difference() raises:
    var w = ca_weights()
    var layer = build_cmha(w, 2)
    var x = ca_x()
    var mem = ca_mem()
    var cot = cotangent()
    var fwd = layer.forward_cached(x.copy(), mem.copy(), no_mask(3, 4))
    var grads = layer.backward(fwd.cache, cot)

    var h = 1e-5
    for i in range(mem.rows):
        for j in range(mem.cols):
            var plus = mem.copy()
            plus[i, j] = plus[i, j] + h
            var minus = mem.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                projected(layer, x, plus, cot) - projected(layer, x, minus, cot)
            ) / (2.0 * h)
            assert_grad_close(grads.d_memory[i, j], numeric)


def finite_diff_param(
    w: List[Tensor2D], which: Int, r: Int, c: Int, cot: Tensor2D
) raises -> Float64:
    # Central difference of the projected loss wrt one parameter entry. `which`
    # indexes [q_w, q_b, kv_w, kv_b, proj_w, proj_b].
    var h = 1e-5
    var wp = List[Tensor2D]()
    var wm = List[Tensor2D]()
    for i in range(len(w)):
        wp.append(w[i].copy())
        wm.append(w[i].copy())
    wp[which][r, c] = wp[which][r, c] + h
    wm[which][r, c] = wm[which][r, c] - h
    var x = ca_x()
    var mem = ca_mem()
    var plus = projected(build_cmha(wp, 2), x, mem, cot)
    var minus = projected(build_cmha(wm, 2), x, mem, cot)
    return (plus - minus) / (2.0 * h)


def test_parameter_grads_match_finite_difference() raises:
    var w = ca_weights()
    var layer = build_cmha(w, 2)
    var x = ca_x()
    var mem = ca_mem()
    var cot = cotangent()
    layer.zero_grad()
    var fwd = layer.forward_cached(x.copy(), mem.copy(), no_mask(3, 4))
    _ = layer.backward(fwd.cache, cot)

    # which: 0=q_w, 1=q_b, 2=kv_w, 3=kv_b, 4=proj_w, 5=proj_b
    var grads = [
        layer.q.weight.grad.copy(),
        layer.q.bias.grad.copy(),
        layer.kv.weight.grad.copy(),
        layer.kv.bias.grad.copy(),
        layer.proj.weight.grad.copy(),
        layer.proj.bias.grad.copy(),
    ]
    for which in range(6):
        for r in range(grads[which].rows):
            for c in range(grads[which].cols):
                var numeric = finite_diff_param(w, which, r, c, cot)
                assert_grad_close(grads[which][r, c], numeric)


def test_backward_accumulates_exactly() raises:
    # Two backward calls without a zero_grad() between them exactly double every
    # parameter grad — the += contract the d_memory summing and any later weight
    # tying rely on.
    var w = ca_weights()
    var layer = build_cmha(w, 2)
    var x = ca_x()
    var mem = ca_mem()
    var cot = cotangent()
    layer.zero_grad()
    var fwd = layer.forward_cached(x.copy(), mem.copy(), no_mask(3, 4))
    _ = layer.backward(fwd.cache, cot)
    var qw1 = layer.q.weight.grad.copy()
    var kvw1 = layer.kv.weight.grad.copy()
    var pw1 = layer.proj.weight.grad.copy()
    var kvb1 = layer.kv.bias.grad.copy()
    _ = layer.backward(fwd.cache, cot)
    for r in range(qw1.rows):
        for c in range(qw1.cols):
            assert_equal_exact(layer.q.weight.grad[r, c], 2.0 * qw1[r, c])
    for r in range(kvw1.rows):
        for c in range(kvw1.cols):
            assert_equal_exact(layer.kv.weight.grad[r, c], 2.0 * kvw1[r, c])
    for r in range(pw1.rows):
        for c in range(pw1.cols):
            assert_equal_exact(layer.proj.weight.grad[r, c], 2.0 * pw1[r, c])
    for c in range(kvb1.cols):
        assert_equal_exact(layer.kv.bias.grad[0, c], 2.0 * kvb1[0, c])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
