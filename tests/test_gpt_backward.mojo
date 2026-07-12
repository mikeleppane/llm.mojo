"""Model-level gradient checks by finite difference.

The weight-tied head means wte receives gradient through two paths — the head
matmul (d_logits^T @ h) and the embedding gather — that sum into its one
Parameter.grad; a finite difference of the wte-table grad checks both are present
and summing. The head path reaches every row of wte, so unused rows check the
head path alone and used rows check both together. wpe, a mid-block parameter,
and ln_f are finite-diff'd too, and backward doubles exactly on a second pass.
Weights are explicit `fill` values (dropout 0, no residual scaling), so no rng is
in play. Finite-difference convention: central diff h = 1e-5, mixed absolute/
relative tolerance |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|.
"""

from std.testing import assert_true, TestSuite

from llm.config import GPTConfig
from llm.nn.embedding import Embedding
from llm.nn.layernorm import LayerNorm
from llm.nn.linear import Linear
from llm.nn.mlp import MLP
from llm.nn.parameter import Parameter
from llm.tensor.ops import cross_entropy_rows_backward
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.attention import MultiHeadAttention
from llm.transformer.block import TransformerBlock
from llm.transformer.gpt import GPT
from llm.utils.random import Rng


def fill(rows: Int, cols: Int, base: Int) raises -> Tensor2D:
    """Deterministic sentinel weights, bit-identical to gpt_reference.py's `fill`.
    """
    var t = zeros_2d(rows, cols)
    for r in range(rows):
        for c in range(cols):
            var k = r * cols + c
            t[r, c] = Float64(((k + base) * 37 + 11) % 101) / 100.0 - 0.5
    return t^


def build_block_off(off: Int) raises -> TransformerBlock:
    var ln1 = LayerNorm(
        Parameter(fill(1, 4, 10 + off)), Parameter(fill(1, 4, 20 + off))
    )
    var attn = MultiHeadAttention(
        Linear(
            Parameter(fill(12, 4, 100 + off)), Parameter(fill(1, 12, 200 + off))
        ),
        Linear(
            Parameter(fill(4, 4, 300 + off)), Parameter(fill(1, 4, 400 + off))
        ),
        2,
    )
    var ln2 = LayerNorm(
        Parameter(fill(1, 4, 30 + off)), Parameter(fill(1, 4, 40 + off))
    )
    var mlp = MLP(
        Linear(
            Parameter(fill(6, 4, 500 + off)), Parameter(fill(1, 6, 600 + off))
        ),
        Linear(
            Parameter(fill(4, 6, 700 + off)), Parameter(fill(1, 4, 800 + off))
        ),
    )
    return TransformerBlock(ln1^, attn^, ln2^, mlp^)


def build_gpt() raises -> GPT:
    """A tiny GPT (V=5, C=4, H=2, L=2, context_length=8, dropout=0) from `fill` weights.
    """
    var cfg = GPTConfig(5, 8, 4, 2, 2, 0.0)
    var wte = Embedding(Parameter(fill(5, 4, 1000)))
    var wpe = Embedding(Parameter(fill(8, 4, 2000)))
    var blocks = List[TransformerBlock]()
    blocks.append(build_block_off(10000))
    blocks.append(build_block_off(20000))
    var ln_f = LayerNorm(
        Parameter(fill(1, 4, 3000)), Parameter(fill(1, 4, 4000))
    )
    return GPT(cfg^, wte^, wpe^, blocks^, ln_f^)


def ids() raises -> List[Int]:
    var out = List[Int]()
    out.append(1)
    out.append(3)
    out.append(4)
    return out^


def targets() raises -> List[Int]:
    var out = List[Int]()
    out.append(3)
    out.append(4)
    out.append(2)
    return out^


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


def analytic_backward(mut gpt: GPT) raises:
    """Run one loss backward, leaving grads populated (inference-equivalent cached path, dropout 0).
    """
    var rng = Rng(0)
    var fwd = gpt.forward_cached(ids(), False, rng)
    var d_logits = cross_entropy_rows_backward(fwd.logits, targets())
    gpt.backward(fwd.cache, d_logits)


def test_wte_table_grad_finite_difference() raises:
    """Every entry of the tied wte table matches finite difference, both paths summing: unused-id rows (0, 2) exercise the head path alone, used-id rows (1, 3, 4) head + gather together.
    """
    var gpt = build_gpt()
    gpt.zero_grad()
    analytic_backward(gpt)
    var analytic = gpt.wte.table.grad.copy()

    var h = 1e-5
    for vi in range(5):
        for c in range(4):
            var gp = build_gpt()
            gp.wte.table.value[vi, c] = gp.wte.table.value[vi, c] + h
            var lp = gp.loss(ids(), targets())
            var gm = build_gpt()
            gm.wte.table.value[vi, c] = gm.wte.table.value[vi, c] - h
            var lm = gm.loss(ids(), targets())
            var numeric = (lp - lm) / (2.0 * h)
            assert_grad_close(analytic[vi, c], numeric)


def test_wpe_grad_finite_difference() raises:
    """The wpe table (gather path only) matches finite difference on used position rows (0, 1, 2); unused rows are exactly zero.
    """
    var gpt = build_gpt()
    gpt.zero_grad()
    analytic_backward(gpt)
    var analytic = gpt.wpe.table.grad.copy()

    var h = 1e-5
    for pos in range(3):  # positions 0, 1, 2 used by T=3
        for c in range(4):
            var gp = build_gpt()
            gp.wpe.table.value[pos, c] = gp.wpe.table.value[pos, c] + h
            var lp = gp.loss(ids(), targets())
            var gm = build_gpt()
            gm.wpe.table.value[pos, c] = gm.wpe.table.value[pos, c] - h
            var lm = gm.loss(ids(), targets())
            var numeric = (lp - lm) / (2.0 * h)
            assert_grad_close(analytic[pos, c], numeric)
    # Unused positions carry no gradient.
    for pos in range(3, 8):
        for c in range(4):
            assert_true(
                analytic[pos, c] == 0.0,
                "unused position " + String(pos) + " got nonzero grad",
            )


def test_midblock_param_finite_difference() raises:
    """Block 0's fused qkv weight [12, 4], reached only through the whole block stack below ln_f, matches finite difference.
    """
    var gpt = build_gpt()
    gpt.zero_grad()
    analytic_backward(gpt)
    var analytic = gpt.blocks[0].attn.qkv.weight.grad.copy()

    var h = 1e-5
    for r in range(12):
        for c in range(4):
            var gp = build_gpt()
            gp.blocks[0].attn.qkv.weight.value[r, c] = (
                gp.blocks[0].attn.qkv.weight.value[r, c] + h
            )
            var lp = gp.loss(ids(), targets())
            var gm = build_gpt()
            gm.blocks[0].attn.qkv.weight.value[r, c] = (
                gm.blocks[0].attn.qkv.weight.value[r, c] - h
            )
            var lm = gm.loss(ids(), targets())
            var numeric = (lp - lm) / (2.0 * h)
            assert_grad_close(analytic[r, c], numeric)


def test_lnf_grad_finite_difference() raises:
    """The final LayerNorm's weight, the last parameter before the tied head, matches finite difference.
    """
    var gpt = build_gpt()
    gpt.zero_grad()
    analytic_backward(gpt)
    var analytic = gpt.ln_f.weight.grad.copy()

    var h = 1e-5
    for c in range(4):
        var gp = build_gpt()
        gp.ln_f.weight.value[0, c] = gp.ln_f.weight.value[0, c] + h
        var lp = gp.loss(ids(), targets())
        var gm = build_gpt()
        gm.ln_f.weight.value[0, c] = gm.ln_f.weight.value[0, c] - h
        var lm = gm.loss(ids(), targets())
        var numeric = (lp - lm) / (2.0 * h)
        assert_grad_close(analytic[0, c], numeric)


def test_model_exact_doubling() raises:
    """Two backward passes without an intervening zero_grad double every grad bit-for-bit, including the tied wte whose two paths combine into one delta per call.
    """
    var gpt = build_gpt()
    gpt.zero_grad()
    var rng = Rng(0)
    var fwd = gpt.forward_cached(ids(), False, rng)
    var d_logits = cross_entropy_rows_backward(fwd.logits, targets())
    gpt.backward(fwd.cache, d_logits)
    var wte1 = gpt.wte.table.grad.copy()
    var wpe1 = gpt.wpe.table.grad.copy()
    var qkv1 = gpt.blocks[1].attn.qkv.weight.grad.copy()
    var lnf1 = gpt.ln_f.weight.grad.copy()
    gpt.backward(fwd.cache, d_logits)
    for r in range(wte1.rows):
        for c in range(wte1.cols):
            assert_true(
                gpt.wte.table.grad[r, c] == 2.0 * wte1[r, c],
                "wte grad did not double exactly (tied-path accumulation)",
            )
    for r in range(wpe1.rows):
        for c in range(wpe1.cols):
            assert_true(gpt.wpe.table.grad[r, c] == 2.0 * wpe1[r, c], "wpe")
    for r in range(qkv1.rows):
        for c in range(qkv1.cols):
            assert_true(
                gpt.blocks[1].attn.qkv.weight.grad[r, c] == 2.0 * qkv1[r, c],
                "block qkv",
            )
    for c in range(lnf1.cols):
        assert_true(gpt.ln_f.weight.grad[0, c] == 2.0 * lnf1[0, c], "ln_f")


def test_backward_stale_cache_raises() raises:
    """A T=3 cache paired with T=2 d_logits is a shape mismatch backward rejects rather than reading out of bounds.
    """
    var gpt = build_gpt()
    var rng = Rng(0)
    var fwd = gpt.forward_cached(ids(), False, rng)  # T=3 cache
    var wrong = zeros_2d(2, 5)  # d_logits for T=2, mismatched
    var raised = False
    try:
        gpt.backward(fwd.cache, wrong)
    except:
        raised = True
    assert_true(raised, "backward accepted a mismatched d_logits")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
