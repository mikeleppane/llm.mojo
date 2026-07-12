"""Tests for EncoderBlock, the pre-LN residual block (self-attention + MLP). The forward golden (Case ENC, from tests/oracles/encdec_reference.py) is wiring-sensitive, a zeroed-sublayer block collapsing to the identity pins the skip connections, and finite-difference checks verify d_x and every parameter grad. Gradient checks project the scalar loss L = sum(cotangent (.) forward), central diff h = 1e-5, mixed absolute/relative tolerance 1e-7 + 1e-5*|n|."""

from std.testing import assert_almost_equal, assert_true, TestSuite

from llm.lab.blocks import EncoderBlock
from llm.nn.layernorm import LayerNorm
from llm.nn.linear import Linear
from llm.nn.mlp import MLP
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.attention import MultiHeadAttention
from llm.transformer.masks import no_mask


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


def enc_x() raises -> Tensor2D:
    return mat(
        [
            0.1093,
            -0.0381,
            0.2259,
            0.113,
            -0.178,
            0.2669,
            -0.0299,
            0.1949,
            -0.0529,
            0.3772,
            0.2343,
            -0.2069,
        ],
        3,
        4,
    )


def enc_ln1_w() raises -> Tensor2D:
    return mat(
        [
            -0.2705,
            -0.0497,
            0.4348,
            0.0422,
        ],
        1,
        4,
    )


def enc_ln1_b() raises -> Tensor2D:
    return mat(
        [
            -0.305,
            -0.079,
            0.5544,
            -0.9879,
        ],
        1,
        4,
    )


def enc_qkv_w() raises -> Tensor2D:
    return mat(
        [
            -0.1368,
            -0.2606,
            0.2633,
            -0.4619,
            -0.0035,
            0.0551,
            -0.6245,
            -0.2506,
            0.0104,
            -0.0112,
            -0.1049,
            -0.4253,
            -0.4175,
            0.061,
            -0.0099,
            0.5159,
            -0.3091,
            0.1653,
            0.1727,
            -0.1153,
            -0.334,
            0.4276,
            0.1429,
            0.274,
            0.0494,
            0.2217,
            -0.1239,
            0.3473,
            0.0312,
            -0.0826,
            -0.4436,
            -0.2226,
            0.2682,
            0.3029,
            0.3019,
            -0.1713,
            0.277,
            0.4637,
            0.3188,
            -0.3509,
            0.257,
            0.4687,
            0.126,
            0.1385,
            -0.3178,
            0.3536,
            0.1279,
            0.2135,
        ],
        12,
        4,
    )


def enc_qkv_b() raises -> Tensor2D:
    return mat(
        [
            -0.6039,
            0.1236,
            -0.0068,
            0.4738,
            -0.3536,
            0.0212,
            -0.1243,
            -0.2146,
            -0.326,
            0.6107,
            0.0837,
            -0.2479,
        ],
        1,
        12,
    )


def enc_proj_w() raises -> Tensor2D:
    return mat(
        [
            -0.3052,
            0.0449,
            0.3388,
            -0.1377,
            0.088,
            0.0215,
            0.6042,
            0.2124,
            0.1298,
            -0.0899,
            0.0138,
            -0.1212,
            0.2957,
            0.4645,
            0.0314,
            -0.0546,
        ],
        4,
        4,
    )


def enc_proj_b() raises -> Tensor2D:
    return mat(
        [
            -0.4757,
            -0.4029,
            0.7169,
            -0.1554,
        ],
        1,
        4,
    )


def enc_ln2_w() raises -> Tensor2D:
    return mat(
        [
            0.6249,
            0.2519,
            0.3444,
            -0.0585,
        ],
        1,
        4,
    )


def enc_ln2_b() raises -> Tensor2D:
    return mat(
        [
            0.2759,
            -0.1073,
            0.3865,
            0.1222,
        ],
        1,
        4,
    )


def enc_up_w() raises -> Tensor2D:
    return mat(
        [
            0.3504,
            -0.1386,
            -0.2915,
            0.2288,
            0.0148,
            0.0073,
            0.2281,
            0.1128,
            0.2251,
            0.207,
            0.3053,
            -0.1566,
            -0.0965,
            0.0857,
            0.2047,
            -0.3202,
            0.0449,
            -0.2963,
            -0.271,
            0.2422,
            -0.0476,
            -0.3192,
            0.0392,
            -0.1537,
            0.4241,
            0.17,
            -0.2506,
            0.0306,
            0.2182,
            -0.1982,
            -0.0838,
            -0.2328,
        ],
        8,
        4,
    )


def enc_up_b() raises -> Tensor2D:
    return mat(
        [
            0.551,
            -0.0915,
            0.3114,
            0.0701,
            0.0184,
            -0.3816,
            -0.2464,
            -0.3662,
        ],
        1,
        8,
    )


def enc_down_w() raises -> Tensor2D:
    return mat(
        [
            0.4296,
            -0.2593,
            0.5132,
            -0.1783,
            0.2749,
            0.1185,
            0.0284,
            -0.2295,
            -0.0819,
            0.3456,
            -0.0925,
            -0.0462,
            0.0699,
            -0.8987,
            -0.0761,
            -0.3069,
            -0.1021,
            -0.1663,
            0.1281,
            -0.2196,
            -0.3914,
            0.2653,
            0.5288,
            -0.2941,
            0.4106,
            0.6904,
            -0.2192,
            0.0906,
            0.2464,
            -0.4174,
            0.0371,
            0.2934,
        ],
        4,
        8,
    )


def enc_down_b() raises -> Tensor2D:
    return mat(
        [
            0.8671,
            0.034,
            -0.1577,
            0.587,
        ],
        1,
        4,
    )


def enc_output() raises -> Tensor2D:
    return mat(
        [
            0.7861673382890391,
            -0.3716128323001341,
            0.6540231477843903,
            1.1350059637662377,
            0.4297549352189258,
            -0.048378130464080654,
            0.401739339203921,
            1.1842382762689216,
            0.5675259395500014,
            0.06983408030812374,
            0.6737399777661135,
            0.7822305255427804,
        ],
        3,
        4,
    )


def enc_weights() raises -> List[Tensor2D]:
    """The 12 parameter tensors of the block in fixed order: ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b, up_w, up_b, down_w, down_b.
    """
    var out = List[Tensor2D]()
    out.append(enc_ln1_w())
    out.append(enc_ln1_b())
    out.append(enc_qkv_w())
    out.append(enc_qkv_b())
    out.append(enc_proj_w())
    out.append(enc_proj_b())
    out.append(enc_ln2_w())
    out.append(enc_ln2_b())
    out.append(enc_up_w())
    out.append(enc_up_b())
    out.append(enc_down_w())
    out.append(enc_down_b())
    return out^


def build_enc_block(w: List[Tensor2D]) raises -> EncoderBlock:
    """Assemble an EncoderBlock from the 12 weight tensors, copied in so a finite-difference loop can perturb any entry and rebuild.
    """
    var ln1 = LayerNorm(Parameter(w[0].copy()), Parameter(w[1].copy()))
    var attn = MultiHeadAttention(
        Linear(Parameter(w[2].copy()), Parameter(w[3].copy())),
        Linear(Parameter(w[4].copy()), Parameter(w[5].copy())),
        2,
    )
    var ln2 = LayerNorm(Parameter(w[6].copy()), Parameter(w[7].copy()))
    var mlp = MLP(
        Linear(Parameter(w[8].copy()), Parameter(w[9].copy())),
        Linear(Parameter(w[10].copy()), Parameter(w[11].copy())),
    )
    return EncoderBlock(ln1^, attn^, ln2^, mlp^)


def cotangent() raises -> Tensor2D:
    return mat(
        [0.7, -0.2, 1.3, -0.5, 0.1, 0.9, -1.1, 0.4, -0.6, 0.3, 0.2, -0.8], 3, 4
    )


def projected(
    block: EncoderBlock, x: Tensor2D, cot: Tensor2D
) raises -> Float64:
    var y = block.forward(x, no_mask(x.rows, x.rows))
    var total = 0.0
    for i in range(y.rows):
        for j in range(y.cols):
            total += cot[i, j] * y[i, j]
    return total


def test_forward_oracle() raises:
    """Forward matches golden Case ENC: pre-LN encoder block, T=3, C=4, H=2, hidden=8, no mask.
    """
    var block = build_enc_block(enc_weights())
    var x = enc_x()
    var y = block.forward(x, no_mask(3, 3))
    var expected = enc_output()
    assert_true(y.rows == 3 and y.cols == 4)
    for i in range(3):
        for j in range(4):
            assert_almost_equal(y[i, j], expected[i, j], atol=1e-12)


def test_zeroed_sublayers_are_identity() raises:
    """With both sublayers zeroed the block is the identity (out = x), isolating the skip connections.
    """
    # With both sublayers' weights and biases zero, attn(ln1(x)) = 0 and
    # mlp(ln2(a)) = 0, so out = x + 0 + 0 = x. If either residual dropped its
    # skip term, the output would collapse toward zero instead of reproducing x.
    var ln1 = LayerNorm.init_default(4)
    var attn = MultiHeadAttention(
        Linear(Parameter(zeros_2d(12, 4)), Parameter(zeros_2d(1, 12))),
        Linear(Parameter(zeros_2d(4, 4)), Parameter(zeros_2d(1, 4))),
        2,
    )
    var ln2 = LayerNorm.init_default(4)
    var mlp = MLP(
        Linear(Parameter(zeros_2d(8, 4)), Parameter(zeros_2d(1, 8))),
        Linear(Parameter(zeros_2d(4, 8)), Parameter(zeros_2d(1, 4))),
    )
    var block = EncoderBlock(ln1^, attn^, ln2^, mlp^)
    var x = enc_x()
    var y = block.forward(x, no_mask(3, 3))
    for i in range(3):
        for j in range(4):
            assert_almost_equal(y[i, j], x[i, j], atol=1e-12)


def test_d_x_matches_finite_difference() raises:
    """The input gradient d_x matches a central finite difference of the projected loss.
    """
    var block = build_enc_block(enc_weights())
    var x = enc_x()
    var cot = cotangent()
    var fwd = block.forward_cached(x, no_mask(3, 3))
    var d_x = block.backward(fwd.cache, cot)
    var h = 1e-5
    for i in range(x.rows):
        for j in range(x.cols):
            var plus = x.copy()
            plus[i, j] = plus[i, j] + h
            var minus = x.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                projected(block, plus, cot) - projected(block, minus, cot)
            ) / (2.0 * h)
            assert_grad_close(d_x[i, j], numeric)


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
    var x = enc_x()
    var plus = projected(build_enc_block(wp), x, cot)
    var minus = projected(build_enc_block(wm), x, cot)
    return (plus - minus) / (2.0 * h)


def test_parameter_grads_match_finite_difference() raises:
    """Every one of the 12 parameter grads matches a central finite difference.
    """
    var w = enc_weights()
    var block = build_enc_block(w)
    var x = enc_x()
    var cot = cotangent()
    block.zero_grad()
    var fwd = block.forward_cached(x, no_mask(3, 3))
    _ = block.backward(fwd.cache, cot)
    var grads = [
        block.ln1.weight.grad.copy(),
        block.ln1.bias.grad.copy(),
        block.attn.qkv.weight.grad.copy(),
        block.attn.qkv.bias.grad.copy(),
        block.attn.proj.weight.grad.copy(),
        block.attn.proj.bias.grad.copy(),
        block.ln2.weight.grad.copy(),
        block.ln2.bias.grad.copy(),
        block.mlp.up.weight.grad.copy(),
        block.mlp.up.bias.grad.copy(),
        block.mlp.down.weight.grad.copy(),
        block.mlp.down.bias.grad.copy(),
    ]
    for which in range(12):
        for r in range(grads[which].rows):
            for c in range(grads[which].cols):
                var numeric = finite_diff_param(w, which, r, c, cot)
                assert_grad_close(grads[which][r, c], numeric)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
