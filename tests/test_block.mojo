# Tests for TransformerBlock — GPT-2's pre-LN decoder block (self-attention).
#
# The forward golden comes from tests/oracles/gpt_reference.py and is
# WIRING-SENSITIVE by construction: a post-LN block (ln(x + sublayer(x))) or a
# LayerNorm applied to the residual sum produces different numbers and fails it.
# Behavioral causality under causal_mask pins that a query never attends to the
# future. The finite-difference checks verify d_x and every parameter grad (with
# dropout off); the residual backward rule d_x = d_out + branch_backward(d_out)
# is what they measure. Two placement checks pin GPT-2's dropout: the skip path
# is never dropped (a zeroed-sublayer block reproduces x EXACTLY even under
# training with high p), and the branch IS dropped (training output differs from
# inference).
#
# Weights come from the shared `fill` pattern (identical to gpt_reference.py's
# `fill`), so the goldens are an independent oracle: the reference math lives in
# NumPy, the implementation in src/, and a mismatch indicts the wiring.
#
# Finite-difference convention (inline): projected scalar loss
# L = sum(cotangent (.) forward), central diff h = 1e-5, mixed tolerance
# |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|.

from std.testing import assert_almost_equal, assert_true, TestSuite

from llm.nn.layernorm import LayerNorm
from llm.nn.linear import Linear
from llm.nn.mlp import MLP
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.attention import MultiHeadAttention
from llm.transformer.block import TransformerBlock
from llm.transformer.masks import causal_mask
from llm.utils.random import Rng


def fill(rows: Int, cols: Int, base: Int) raises -> Tensor2D:
    # The oracle's deterministic asymmetric pattern, entry at flat index k:
    #   v = (((k + base) * 37 + 11) mod 101) / 100 - 0.5
    # Integer modular arithmetic then /100 is exact in Float64, so this is
    # bit-identical to gpt_reference.py's `fill`. MUST match it exactly.
    var t = zeros_2d(rows, cols)
    for r in range(rows):
        for c in range(cols):
            var k = r * cols + c
            t[r, c] = Float64(((k + base) * 37 + 11) % 101) / 100.0 - 0.5
    return t^


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


# Config: C=4, H=2, d_hidden=6, T=3 — matches gpt_reference.py's block case.
def block_weights() raises -> List[Tensor2D]:
    # The 12 parameter tensors in fixed order, from `fill` with the oracle's
    # bases: [ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b, up_w,
    # up_b, down_w, down_b].
    var out = List[Tensor2D]()
    out.append(fill(1, 4, 10))  # ln1_w
    out.append(fill(1, 4, 20))  # ln1_b
    out.append(fill(12, 4, 100))  # qkv_w [3C, C]
    out.append(fill(1, 12, 200))  # qkv_b
    out.append(fill(4, 4, 300))  # proj_w
    out.append(fill(1, 4, 400))  # proj_b
    out.append(fill(1, 4, 30))  # ln2_w
    out.append(fill(1, 4, 40))  # ln2_b
    out.append(fill(6, 4, 500))  # up_w [d_hidden, C]
    out.append(fill(1, 6, 600))  # up_b
    out.append(fill(4, 6, 700))  # down_w [C, d_hidden]
    out.append(fill(1, 4, 800))  # down_b
    return out^


def build_block(w: List[Tensor2D]) raises -> TransformerBlock:
    # Assemble a TransformerBlock from the 12 tensors, copied in so a
    # finite-difference loop can perturb any entry and rebuild.
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
    return TransformerBlock(ln1^, attn^, ln2^, mlp^)


def block_x() raises -> Tensor2D:
    # Block input [T=3, C=4] = fill(3, 4, 0), matching the oracle.
    return fill(3, 4, 0)


def cotangent() raises -> Tensor2D:
    # Fixed asymmetric d_out [T=3, C=4].
    var t = zeros_2d(3, 4)
    var vals = [0.7, -0.2, 1.3, -0.5, 0.1, 0.9, -1.1, 0.4, -0.6, 0.3, 0.2, -0.8]
    for r in range(3):
        for c in range(4):
            t[r, c] = vals[r * 4 + c]
    return t^


def projected(
    block: TransformerBlock, x: Tensor2D, cot: Tensor2D
) raises -> Float64:
    var y = block.forward(x, causal_mask(x.rows))
    var total = 0.0
    for i in range(y.rows):
        for j in range(y.cols):
            total += cot[i, j] * y[i, j]
    return total


def block_out_golden() raises -> Tensor2D:
    # Frozen from tests/oracles/gpt_reference.py (`block_out`), the pre-LN block
    # forward at C=4 H=2 d_hidden=6 T=3 under a causal mask.
    var t = zeros_2d(3, 4)
    var vals = [
        -0.4379580143353259,
        -0.43586067182177907,
        0.8647097813821825,
        -0.08390743947789026,
        0.016256922041622157,
        0.21913427854018372,
        0.3872375167312304,
        0.2236864684210838,
        -0.5461489934960506,
        -0.4768202834115578,
        0.714299111081661,
        -0.13876694816100837,
    ]
    for r in range(3):
        for c in range(4):
            t[r, c] = vals[r * 4 + c]
    return t^


def test_forward_oracle_golden() raises:
    # Pre-LN wiring-sensitive golden — post-LN or LN-on-the-sum fails it.
    var block = build_block(block_weights())
    var y = block.forward(block_x(), causal_mask(3))
    var expected = block_out_golden()
    assert_true(y.rows == 3 and y.cols == 4)
    for i in range(3):
        for j in range(4):
            assert_almost_equal(y[i, j], expected[i, j], atol=1e-12)


def test_causality_under_causal_mask() raises:
    # Perturbing input row j must leave output rows < j unchanged: under the
    # causal mask a query at position i attends only to positions <= i, and the
    # MLP is position-wise, so output row i depends only on input rows 0..i.
    var block = build_block(block_weights())
    var x = block_x()
    var base = block.forward(x, causal_mask(3))
    var j = 1
    var perturbed = x.copy()
    for c in range(x.cols):
        perturbed[j, c] = perturbed[j, c] + 0.5
    var after = block.forward(perturbed, causal_mask(3))
    for i in range(j):  # rows strictly before j
        for c in range(x.cols):
            assert_almost_equal(after[i, c], base[i, c], atol=1e-12)
    # Sanity: row j itself DID change (the perturbation is not a no-op).
    var changed = False
    for c in range(x.cols):
        if abs(after[j, c] - base[j, c]) > 1e-9:
            changed = True
    assert_true(
        changed, "perturbing row j did not change row j — mask too strong"
    )


def test_forward_cached_eval_equals_forward() raises:
    # forward_cached(training=False) must equal the inference forward exactly, and
    # consume no rng.
    var block = build_block(block_weights())
    var x = block_x()
    var mask = causal_mask(3)
    var y_inf = block.forward(x, mask)
    var rng = Rng(5)
    var rng_ref = Rng(5)
    var fwd = block.forward_cached(x, mask, 0.5, False, rng)
    for i in range(3):
        for j in range(4):
            assert_almost_equal(fwd.output[i, j], y_inf[i, j], atol=1e-12)
    assert_true(rng.state == rng_ref.state, "eval forward_cached consumed rng")


def test_d_x_matches_finite_difference() raises:
    var block = build_block(block_weights())
    var x = block_x()
    var cot = cotangent()
    var mask = causal_mask(3)
    var rng = Rng(0)
    var fwd = block.forward_cached(x, mask, 0.0, False, rng)
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
    var x = block_x()
    var plus = projected(build_block(wp), x, cot)
    var minus = projected(build_block(wm), x, cot)
    return (plus - minus) / (2.0 * h)


def test_parameter_grads_match_finite_difference() raises:
    var w = block_weights()
    var block = build_block(w)
    var x = block_x()
    var cot = cotangent()
    var mask = causal_mask(3)
    block.zero_grad()
    var rng = Rng(0)
    var fwd = block.forward_cached(x, mask, 0.0, False, rng)
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


def test_exact_accumulation_doubling() raises:
    # Two backward passes without a zero_grad between them double the grads
    # bit-for-bit (the tied-weight accumulation contract, at the block level).
    var block = build_block(block_weights())
    var x = block_x()
    var cot = cotangent()
    var mask = causal_mask(3)
    block.zero_grad()
    var rng = Rng(0)
    var fwd = block.forward_cached(x, mask, 0.0, False, rng)
    _ = block.backward(fwd.cache, cot)
    var qkv1 = block.attn.qkv.weight.grad.copy()
    var down1 = block.mlp.down.weight.grad.copy()
    _ = block.backward(fwd.cache, cot)
    for i in range(qkv1.rows):
        for j in range(qkv1.cols):
            assert_true(
                block.attn.qkv.weight.grad[i, j] == 2.0 * qkv1[i, j],
                "qkv grad did not double exactly",
            )
    for i in range(down1.rows):
        for j in range(down1.cols):
            assert_true(
                block.mlp.down.weight.grad[i, j] == 2.0 * down1[i, j],
                "down grad did not double exactly",
            )


def test_residual_dropout_skip_never_dropped() raises:
    # Zeroed sublayers make both branches 0, so out = x + dropout(0) + dropout(0)
    # = x EXACTLY — even under training with high p. If the skip path were dropped
    # (or dropout applied to the residual sum), out would be a sparsified x.
    var ln1 = LayerNorm.init_default(4)
    var attn = MultiHeadAttention(
        Linear(Parameter(zeros_2d(12, 4)), Parameter(zeros_2d(1, 12))),
        Linear(Parameter(zeros_2d(4, 4)), Parameter(zeros_2d(1, 4))),
        2,
    )
    var ln2 = LayerNorm.init_default(4)
    var mlp = MLP(
        Linear(Parameter(zeros_2d(6, 4)), Parameter(zeros_2d(1, 6))),
        Linear(Parameter(zeros_2d(4, 6)), Parameter(zeros_2d(1, 4))),
    )
    var block = TransformerBlock(ln1^, attn^, ln2^, mlp^)
    var x = block_x()
    var rng = Rng(7)
    var fwd = block.forward_cached(x, causal_mask(3), 0.9, True, rng)
    for i in range(3):
        for j in range(4):
            assert_almost_equal(fwd.output[i, j], x[i, j], atol=1e-12)


def test_residual_dropout_branch_is_dropped() raises:
    # With real sublayers, training=True at high p must change the output relative
    # to inference — the branch dropout is active (the complement of the skip
    # test: dropout is not a silent no-op).
    var block = build_block(block_weights())
    var x = block_x()
    var mask = causal_mask(3)
    var y_inf = block.forward(x, mask)
    var rng = Rng(3)
    var fwd = block.forward_cached(x, mask, 0.9, True, rng)
    var differs = False
    for i in range(3):
        for j in range(4):
            if abs(fwd.output[i, j] - y_inf[i, j]) > 1e-6:
                differs = True
    assert_true(differs, "training=True output did not differ — dropout inert")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
