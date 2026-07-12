"""Tests for DecoderBlock: the pre-LN block with masked self-attention, cross-attention over memory, and an MLP (three residuals).

Covers the forward golden (Case DEC), two behavioral wiring checks (causal
masking, memory is read), and finite-difference checks for d_x, d_memory, and
all twenty parameter grads. Finite-difference convention: projected scalar loss
L = sum(cotangent * forward), central diff h = 1e-5, mixed absolute/relative
tolerance |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|.
"""

from std.testing import assert_almost_equal, assert_true, TestSuite

from llm.lab.blocks import DecoderBlock
from llm.lab.cross_attention import CrossMultiHeadAttention
from llm.nn.layernorm import LayerNorm
from llm.nn.linear import Linear
from llm.nn.mlp import MLP
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.attention import MultiHeadAttention
from llm.transformer.masks import causal_mask, no_mask


def mat(flat: List[Float64], rows: Int, cols: Int) raises -> Tensor2D:
    if len(flat) != rows * cols:
        raise Error("mat: length mismatch")
    var t = zeros_2d(rows, cols)
    for r in range(rows):
        for c in range(cols):
            t[r, c] = flat[r * cols + c]
    return t^


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


def dec_x() raises -> Tensor2D:
    return mat(
        [
            0.0801,
            -0.199,
            0.4209,
            0.2131,
            0.0912,
            0.0916,
            -0.2849,
            0.0032,
            -0.2264,
            -0.5317,
            0.423,
            0.4476,
        ],
        3,
        4,
    )


def dec_mem() raises -> Tensor2D:
    return mat(
        [
            -0.118,
            0.3183,
            0.2437,
            0.3488,
            -0.2912,
            -0.063,
            0.1847,
            0.1322,
            0.2374,
            -0.081,
            0.2569,
            -0.102,
            0.1621,
            -0.0927,
            0.384,
            0.0363,
        ],
        4,
        4,
    )


def dec_ln1_w() raises -> Tensor2D:
    return mat(
        [
            -0.0187,
            -0.3825,
            -0.0077,
            0.1364,
        ],
        1,
        4,
    )


def dec_ln1_b() raises -> Tensor2D:
    return mat(
        [
            0.2475,
            -0.095,
            -0.0146,
            -0.2826,
        ],
        1,
        4,
    )


def dec_sqkv_w() raises -> Tensor2D:
    return mat(
        [
            -0.2219,
            0.0871,
            0.5006,
            0.0598,
            0.3077,
            0.023,
            -0.4214,
            -0.0276,
            -0.3387,
            0.2002,
            0.2571,
            0.0394,
            -0.2014,
            0.4069,
            0.2957,
            -0.042,
            -0.5832,
            0.6452,
            -0.2145,
            -0.1194,
            0.2837,
            -0.3395,
            -0.2089,
            0.0641,
            -0.0915,
            -0.2221,
            -0.6638,
            0.0711,
            0.1636,
            0.0114,
            -0.1029,
            -0.8326,
            0.2992,
            -0.1119,
            -0.1215,
            0.4319,
            -0.0501,
            0.0271,
            0.2082,
            0.1648,
            -0.1855,
            -0.4349,
            0.269,
            0.1177,
            0.1808,
            -0.0616,
            0.0878,
            -0.426,
        ],
        12,
        4,
    )


def dec_sqkv_b() raises -> Tensor2D:
    return mat(
        [
            0.0579,
            0.9513,
            -0.3071,
            -0.1018,
            -0.4297,
            0.1221,
            0.1311,
            0.1936,
            -0.0427,
            -0.5259,
            -0.1044,
            0.3112,
        ],
        1,
        12,
    )


def dec_sproj_w() raises -> Tensor2D:
    return mat(
        [
            0.193,
            -0.2876,
            -0.6538,
            -0.405,
            0.0012,
            -0.3544,
            0.1192,
            -0.4434,
            0.378,
            -0.5667,
            -0.1792,
            0.107,
            -0.8128,
            -0.3791,
            0.0677,
            0.5824,
        ],
        4,
        4,
    )


def dec_sproj_b() raises -> Tensor2D:
    return mat(
        [
            0.2249,
            0.1331,
            -0.4916,
            -0.0527,
        ],
        1,
        4,
    )


def dec_ln2_w() raises -> Tensor2D:
    return mat(
        [
            0.3665,
            -0.2741,
            -0.3678,
            0.0017,
        ],
        1,
        4,
    )


def dec_ln2_b() raises -> Tensor2D:
    return mat(
        [
            0.1696,
            -0.6664,
            -0.2991,
            -0.2827,
        ],
        1,
        4,
    )


def dec_cq_w() raises -> Tensor2D:
    return mat(
        [
            -0.0567,
            0.1463,
            0.1098,
            0.3379,
            -0.273,
            -0.2758,
            -0.2872,
            0.1219,
            0.0095,
            -0.0215,
            -0.1941,
            -0.8115,
            0.2208,
            -0.1648,
            -0.1983,
            -0.1695,
        ],
        4,
        4,
    )


def dec_cq_b() raises -> Tensor2D:
    return mat(
        [
            0.0955,
            0.0505,
            -0.0199,
            0.1803,
        ],
        1,
        4,
    )


def dec_ckv_w() raises -> Tensor2D:
    return mat(
        [
            0.1161,
            0.089,
            -0.06,
            -0.3098,
            0.0557,
            0.0177,
            -0.1692,
            -0.3252,
            0.3883,
            -0.2137,
            0.4165,
            0.162,
            -0.4393,
            -0.1049,
            0.4686,
            0.306,
            0.3279,
            -0.4601,
            0.354,
            0.4078,
            0.0076,
            -0.2165,
            -0.0293,
            -0.0935,
            0.3517,
            0.2593,
            0.1474,
            0.1634,
            -0.2731,
            -0.3725,
            -0.0378,
            -0.3402,
        ],
        8,
        4,
    )


def dec_ckv_b() raises -> Tensor2D:
    return mat(
        [
            0.1539,
            0.5677,
            0.1379,
            -0.1749,
            0.2511,
            -0.2509,
            0.2029,
            -0.0884,
        ],
        1,
        8,
    )


def dec_cproj_w() raises -> Tensor2D:
    return mat(
        [
            -0.2247,
            -0.2752,
            -0.0761,
            -0.3377,
            0.0004,
            -0.1011,
            0.4698,
            0.4059,
            -0.1067,
            -0.2255,
            -0.6233,
            -0.0716,
            0.382,
            0.3871,
            0.5686,
            0.2658,
        ],
        4,
        4,
    )


def dec_cproj_b() raises -> Tensor2D:
    return mat(
        [
            -0.021,
            0.3518,
            -0.1572,
            0.0944,
        ],
        1,
        4,
    )


def dec_ln3_w() raises -> Tensor2D:
    return mat(
        [
            0.2022,
            0.0386,
            -0.4099,
            -0.2062,
        ],
        1,
        4,
    )


def dec_ln3_b() raises -> Tensor2D:
    return mat(
        [
            -0.4475,
            -0.6192,
            -0.6743,
            0.1109,
        ],
        1,
        4,
    )


def dec_up_w() raises -> Tensor2D:
    return mat(
        [
            0.0939,
            -0.1183,
            0.042,
            -0.3671,
            0.1615,
            -0.2349,
            0.1073,
            -0.0737,
            0.2547,
            0.0052,
            0.327,
            0.2744,
            0.0398,
            0.3694,
            -0.0414,
            -0.1622,
            -0.0334,
            0.2743,
            0.0237,
            -0.1194,
            0.3297,
            -0.3163,
            0.3708,
            -0.2848,
            0.3408,
            -0.3089,
            -0.1406,
            -0.0246,
            -0.1335,
            0.0959,
            -0.5173,
            -0.2411,
        ],
        8,
        4,
    )


def dec_up_b() raises -> Tensor2D:
    return mat(
        [
            0.1451,
            0.118,
            -0.6356,
            -0.0133,
            -0.1928,
            0.313,
            0.2559,
            -0.1744,
        ],
        1,
        8,
    )


def dec_down_w() raises -> Tensor2D:
    return mat(
        [
            0.1185,
            0.2558,
            -0.547,
            0.4946,
            0.0602,
            -0.2134,
            -0.3281,
            -0.1009,
            0.2461,
            0.4923,
            -0.1292,
            0.0563,
            -0.6791,
            0.2527,
            -0.0562,
            -0.136,
            -0.0155,
            0.0169,
            0.2135,
            -0.1209,
            0.1082,
            -0.2538,
            0.0239,
            0.1107,
            0.5342,
            0.4146,
            -0.2886,
            -0.1418,
            0.0209,
            -0.278,
            0.0373,
            -0.0457,
        ],
        4,
        8,
    )


def dec_down_b() raises -> Tensor2D:
    return mat(
        [
            0.0695,
            -0.6009,
            0.0183,
            0.5604,
        ],
        1,
        4,
    )


def dec_output() raises -> Tensor2D:
    return mat(
        [
            0.5593629380661179,
            -0.00554276052364322,
            -0.048687523907450334,
            1.6236746706530072,
            0.45702175765638475,
            0.29959741179101695,
            -0.7683428587923122,
            1.396810155911717,
            0.1728235504045402,
            -0.37478872740451835,
            -0.017500195329752236,
            1.8471635223066705,
        ],
        3,
        4,
    )


def dec_weights() raises -> List[Tensor2D]:
    """The 20 parameter tensors in fixed order: ln1(w,b), self-attn(qkv w,b; proj w,b), ln2(w,b), cross-attn(q w,b; kv w,b; proj w,b), ln3(w,b), mlp(up w,b; down w,b).
    """
    var out = List[Tensor2D]()
    out.append(dec_ln1_w())
    out.append(dec_ln1_b())
    out.append(dec_sqkv_w())
    out.append(dec_sqkv_b())
    out.append(dec_sproj_w())
    out.append(dec_sproj_b())
    out.append(dec_ln2_w())
    out.append(dec_ln2_b())
    out.append(dec_cq_w())
    out.append(dec_cq_b())
    out.append(dec_ckv_w())
    out.append(dec_ckv_b())
    out.append(dec_cproj_w())
    out.append(dec_cproj_b())
    out.append(dec_ln3_w())
    out.append(dec_ln3_b())
    out.append(dec_up_w())
    out.append(dec_up_b())
    out.append(dec_down_w())
    out.append(dec_down_b())
    return out^


def build_dec_block(w: List[Tensor2D]) raises -> DecoderBlock:
    var ln1 = LayerNorm(Parameter(w[0].copy()), Parameter(w[1].copy()))
    var self_attn = MultiHeadAttention(
        Linear(Parameter(w[2].copy()), Parameter(w[3].copy())),
        Linear(Parameter(w[4].copy()), Parameter(w[5].copy())),
        2,
    )
    var ln2 = LayerNorm(Parameter(w[6].copy()), Parameter(w[7].copy()))
    var cross_attn = CrossMultiHeadAttention(
        Linear(Parameter(w[8].copy()), Parameter(w[9].copy())),
        Linear(Parameter(w[10].copy()), Parameter(w[11].copy())),
        Linear(Parameter(w[12].copy()), Parameter(w[13].copy())),
        2,
    )
    var ln3 = LayerNorm(Parameter(w[14].copy()), Parameter(w[15].copy()))
    var mlp = MLP(
        Linear(Parameter(w[16].copy()), Parameter(w[17].copy())),
        Linear(Parameter(w[18].copy()), Parameter(w[19].copy())),
    )
    return DecoderBlock(ln1^, self_attn^, ln2^, cross_attn^, ln3^, mlp^)


def cotangent() raises -> Tensor2D:
    return mat(
        [0.7, -0.2, 1.3, -0.5, 0.1, 0.9, -1.1, 0.4, -0.6, 0.3, 0.2, -0.8], 3, 4
    )


def projected(
    block: DecoderBlock, x: Tensor2D, mem: Tensor2D, cot: Tensor2D
) raises -> Float64:
    var y = block.forward(
        x, mem, causal_mask(x.rows), no_mask(x.rows, mem.rows)
    )
    var total = 0.0
    for i in range(y.rows):
        for j in range(y.cols):
            total += cot[i, j] * y[i, j]
    return total


def test_forward_oracle() raises:
    """Golden Case DEC: pre-LN decoder block, T_tgt=3, T_src=4, C=4, H=2, causal self-mask, no-mask cross.
    """
    var block = build_dec_block(dec_weights())
    var x = dec_x()
    var mem = dec_mem()
    var y = block.forward(x, mem, causal_mask(3), no_mask(3, 4))
    var expected = dec_output()
    assert_true(y.rows == 3 and y.cols == 4)
    for i in range(3):
        for j in range(4):
            assert_almost_equal(y[i, j], expected[i, j], atol=1e-12)


def test_causality_earlier_rows_unchanged() raises:
    """Perturbing the decoder input at position j leaves output rows < j unchanged (causal self-attention plus prefix-only cross queries).
    """
    # Perturb x at position 1; row 0 must be identical.
    var block = build_dec_block(dec_weights())
    var x = dec_x()
    var mem = dec_mem()
    var base = block.forward(x, mem, causal_mask(3), no_mask(3, 4))
    var xp = x.copy()
    xp[1, 2] = xp[1, 2] + 0.5
    var perturbed = block.forward(xp, mem, causal_mask(3), no_mask(3, 4))
    for j in range(4):
        assert_almost_equal(perturbed[0, j], base[0, j], atol=1e-14)
    # And a later row (row 2) DOES change — the perturbation is not inert.
    var changed = False
    for j in range(4):
        if abs(perturbed[2, j] - base[2, j]) > 1e-9:
            changed = True
    assert_true(changed)


def test_memory_is_read() raises:
    """Perturbing memory changes the output: the decoder actually reads the encoder through cross-attention.
    """
    var block = build_dec_block(dec_weights())
    var x = dec_x()
    var mem = dec_mem()
    var base = block.forward(x, mem, causal_mask(3), no_mask(3, 4))
    var memp = mem.copy()
    memp[0, 0] = memp[0, 0] + 0.5
    var perturbed = block.forward(x, memp, causal_mask(3), no_mask(3, 4))
    var changed = False
    for i in range(3):
        for j in range(4):
            if abs(perturbed[i, j] - base[i, j]) > 1e-9:
                changed = True
    assert_true(changed)


def test_d_x_matches_finite_difference() raises:
    """Gradient d_x matches a central finite difference of the projected loss.
    """
    var block = build_dec_block(dec_weights())
    var x = dec_x()
    var mem = dec_mem()
    var cot = cotangent()
    var fwd = block.forward_cached(x, mem, causal_mask(3), no_mask(3, 4))
    var grads = block.backward(fwd.cache, cot)
    var h = 1e-5
    for i in range(x.rows):
        for j in range(x.cols):
            var plus = x.copy()
            plus[i, j] = plus[i, j] + h
            var minus = x.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                projected(block, plus, mem, cot)
                - projected(block, minus, mem, cot)
            ) / (2.0 * h)
            assert_grad_close(grads.d_x[i, j], numeric)


def test_d_memory_matches_finite_difference() raises:
    """Gradient d_memory (flowing only through cross-attention) matches a central finite difference.
    """
    var block = build_dec_block(dec_weights())
    var x = dec_x()
    var mem = dec_mem()
    var cot = cotangent()
    var fwd = block.forward_cached(x, mem, causal_mask(3), no_mask(3, 4))
    var grads = block.backward(fwd.cache, cot)
    var h = 1e-5
    for i in range(mem.rows):
        for j in range(mem.cols):
            var plus = mem.copy()
            plus[i, j] = plus[i, j] + h
            var minus = mem.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                projected(block, x, plus, cot) - projected(block, x, minus, cot)
            ) / (2.0 * h)
            assert_grad_close(grads.d_memory[i, j], numeric)


def finite_diff_param(
    w: List[Tensor2D], which: Int, r: Int, c: Int, cot: Tensor2D
) raises -> Float64:
    var h = 1e-5
    var wp = List[Tensor2D]()
    var wm = List[Tensor2D]()
    for i in range(len(w)):
        wp.append(w[i].copy())
        wm.append(w[i].copy())
    wp[which][r, c] = wp[which][r, c] + h
    wm[which][r, c] = wm[which][r, c] - h
    var x = dec_x()
    var mem = dec_mem()
    var plus = projected(build_dec_block(wp), x, mem, cot)
    var minus = projected(build_dec_block(wm), x, mem, cot)
    return (plus - minus) / (2.0 * h)


def test_parameter_grads_match_finite_difference() raises:
    """All twenty parameter gradients match a central finite difference."""
    var w = dec_weights()
    var block = build_dec_block(w)
    var x = dec_x()
    var mem = dec_mem()
    var cot = cotangent()
    block.zero_grad()
    var fwd = block.forward_cached(x, mem, causal_mask(3), no_mask(3, 4))
    _ = block.backward(fwd.cache, cot)
    var grads = [
        block.ln1.weight.grad.copy(),
        block.ln1.bias.grad.copy(),
        block.self_attn.qkv.weight.grad.copy(),
        block.self_attn.qkv.bias.grad.copy(),
        block.self_attn.proj.weight.grad.copy(),
        block.self_attn.proj.bias.grad.copy(),
        block.ln2.weight.grad.copy(),
        block.ln2.bias.grad.copy(),
        block.cross_attn.q.weight.grad.copy(),
        block.cross_attn.q.bias.grad.copy(),
        block.cross_attn.kv.weight.grad.copy(),
        block.cross_attn.kv.bias.grad.copy(),
        block.cross_attn.proj.weight.grad.copy(),
        block.cross_attn.proj.bias.grad.copy(),
        block.ln3.weight.grad.copy(),
        block.ln3.bias.grad.copy(),
        block.mlp.up.weight.grad.copy(),
        block.mlp.up.bias.grad.copy(),
        block.mlp.down.weight.grad.copy(),
        block.mlp.down.bias.grad.copy(),
    ]
    for which in range(20):
        for r in range(grads[which].rows):
            for c in range(grads[which].cols):
                var numeric = finite_diff_param(w, which, r, c, cot)
                assert_grad_close(grads[which][r, c], numeric)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
