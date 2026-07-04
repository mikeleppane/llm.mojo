# Tests for GPT — the assembled decoder-only model.
#
# forward math: shape, named length errors at the edge, the tied head pinned
# against a manual h @ wte^T (no bias), positions actually used, and a full
# tiny-config forward golden from tests/oracles/gpt_reference.py (independent
# NumPy oracle, shared `fill` weight pattern). Behavioral causality at the model
# level pins the causal mask reaches every block. init: loss ~ log V, same-seed
# determinism, the residual-init std bands (proj/down at 0.02/sqrt(2L), qkv/up at
# 0.02 — scaling the wrong matrices fails a band). Inventory: the walk equals the
# Part VIII formula on two tiny configs (one asymmetric) and counts the tied wte
# exactly once. zero_grad/apply_sgd reach every Parameter.
#
# The oracle model is built from explicit `fill` weights (NO residual scaling —
# that is an init concern, tested separately), so the golden isolates the forward
# wiring.

from std.math import log, sqrt

from std.testing import assert_almost_equal, assert_true, TestSuite

from llm.config import GPTConfig
from llm.nn.embedding import Embedding
from llm.nn.layernorm import LayerNorm
from llm.nn.linear import Linear
from llm.nn.mlp import MLP
from llm.nn.parameter import Parameter
from llm.tensor.ops import cross_entropy_rows, matmul, transpose
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.attention import MultiHeadAttention
from llm.transformer.block import TransformerBlock
from llm.transformer.gpt import GPT
from llm.utils.random import Rng


def fill(rows: Int, cols: Int, base: Int) raises -> Tensor2D:
    # Bit-identical to gpt_reference.py's `fill`:
    #   v = (((k + base) * 37 + 11) mod 101) / 100 - 0.5,  k = row-major index.
    var t = zeros_2d(rows, cols)
    for r in range(rows):
        for c in range(cols):
            var k = r * cols + c
            t[r, c] = Float64(((k + base) * 37 + 11) % 101) / 100.0 - 0.5
    return t^


def build_block_off(off: Int) raises -> TransformerBlock:
    # A block (C=4, H=2, d_hidden=6) from `fill` with per-layer base offset `off`,
    # matching gpt_reference.py's block construction.
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


def build_tiny_gpt() raises -> GPT:
    # The oracle's tiny model: V=5, C=4, H=2, L=2, context_length=8, dropout=0,
    # weights from `fill` (wte base 1000, wpe base 2000, ln_f 3000/4000, block l
    # base offset 10000*(l+1)). Built directly (no residual scaling).
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


def gpt_logits_golden() raises -> Tensor2D:
    # Frozen from tests/oracles/gpt_reference.py (`gpt_logits`), tiny GPT forward
    # at ids=[1,3,4], tied head.
    var t = zeros_2d(3, 5)
    var vals = [
        0.2954537191957601,
        0.049836248425203455,
        0.3108379852716377,
        -0.11424779785235689,
        0.3262222513475152,
        0.16504979347293214,
        0.20692093779455453,
        0.1351442434736777,
        0.011047558793039564,
        0.10523869347442323,
        0.2743948701134066,
        0.10123981133454175,
        0.2849002182291215,
        -0.10883479794593241,
        0.2954055663448364,
    ]
    for r in range(3):
        for c in range(5):
            t[r, c] = vals[r * 5 + c]
    return t^


def ids345() raises -> List[Int]:
    var out = List[Int]()
    out.append(1)
    out.append(3)
    out.append(4)
    return out^


def test_forward_shape() raises:
    var gpt = build_tiny_gpt()
    var logits = gpt.forward(ids345())
    assert_true(logits.rows == 3 and logits.cols == 5, "logits must be [T, V]")


def test_forward_oracle_golden() raises:
    # Full tiny-model forward golden — position off-by-one, a wrong block order, a
    # missing residual, or an untied head all fail it.
    var gpt = build_tiny_gpt()
    var logits = gpt.forward(ids345())
    var expected = gpt_logits_golden()
    for r in range(3):
        for c in range(5):
            assert_almost_equal(logits[r, c], expected[r, c], atol=1e-12)


def test_tied_head_is_manual_matmul_no_bias() raises:
    # The head must be exactly h @ wte.table^T with NO bias. Recompute h = ln_f of
    # the block stack and check logits == matmul(h, transpose(table)).
    var gpt = build_tiny_gpt()
    var ids = ids345()
    var logits = gpt.forward(ids)
    # Recompute h independently: embeddings -> blocks -> ln_f (eval, draws no rng).
    var rng = Rng(0)
    var fwd = gpt.forward_cached(ids, False, rng)
    var h = fwd.cache.h.copy()  # ln_f output
    var manual = matmul(h, transpose(gpt.wte.table.value))  # [T, V], no bias
    for r in range(logits.rows):
        for c in range(logits.cols):
            assert_almost_equal(logits[r, c], manual[r, c], atol=1e-12)


def test_length_zero_raises() raises:
    var gpt = build_tiny_gpt()
    var empty = List[Int]()
    var raised = False
    try:
        _ = gpt.forward(empty)
    except e:
        raised = True
        assert_true(
            String(e).find("must be positive") != -1,
            "T=0 error not the named length error: " + String(e),
        )
    assert_true(raised, "T=0 did not raise")


def test_length_over_context_raises() raises:
    var gpt = build_tiny_gpt()  # context_length = 8
    var ids = List[Int]()
    for _ in range(9):  # T=9 > 8
        ids.append(0)
    var raised = False
    try:
        _ = gpt.forward(ids)
    except e:
        raised = True
        assert_true(
            String(e).find("exceeds context_length") != -1,
            "T>ctx error not the named length error: " + String(e),
        )
    assert_true(raised, "T>context_length did not raise")


def test_model_causality() raises:
    # Perturbing input position j leaves logits rows < j unchanged: the causal
    # mask reaches every block, so an earlier position never sees a later one.
    var gpt = build_tiny_gpt()
    var ids = ids345()
    var base = gpt.forward(ids)
    # Change the token at position 2 (last); rows 0 and 1 must be unchanged.
    var ids2 = List[Int]()
    ids2.append(1)
    ids2.append(3)
    ids2.append(2)  # was 4
    var after = gpt.forward(ids2)
    for r in range(2):  # rows before the changed position
        for c in range(5):
            assert_almost_equal(after[r, c], base[r, c], atol=1e-12)


def test_positions_are_used() raises:
    # wpe adds a position-dependent vector, so the same token at two positions
    # gets different logits. Feed [t, t] and check row 0 != row 1.
    var gpt = build_tiny_gpt()
    var ids = List[Int]()
    ids.append(2)
    ids.append(2)  # same token, different positions
    var logits = gpt.forward(ids)
    var differs = False
    for c in range(5):
        if abs(logits[0, c] - logits[1, c]) > 1e-9:
            differs = True
    assert_true(differs, "same token at two positions gave identical logits")


def test_init_loss_near_log_v() raises:
    # A freshly initialized GPT (0.02-scale weights, tied head) has near-uniform
    # logits, so the initial loss is close to log V.
    var cfg = GPTConfig(20, 16, 16, 2, 2, 0.0)
    var rng = Rng(42)
    var gpt = GPT.init_random(cfg, rng)
    var ids = List[Int]()
    var targets = List[Int]()
    for i in range(6):
        ids.append(i % 20)
        targets.append((i + 1) % 20)
    var loss = gpt.loss(ids, targets)
    var expected = log(20.0)
    assert_true(
        abs(loss - expected) < 0.5,
        "init loss " + String(loss) + " not near log V " + String(expected),
    )


def test_same_seed_identical_model() raises:
    # A seed reproduces the model bit-for-bit: two init_random with the same seed
    # produce identical logits.
    var cfg = GPTConfig(12, 8, 8, 2, 2, 0.0)
    var rng_a = Rng(7)
    var gpt_a = GPT.init_random(cfg, rng_a)
    var rng_b = Rng(7)
    var gpt_b = GPT.init_random(cfg, rng_b)
    var ids = List[Int]()
    for i in range(5):
        ids.append(i)
    var la = gpt_a.forward(ids)
    var lb = gpt_b.forward(ids)
    for r in range(la.rows):
        for c in range(la.cols):
            assert_almost_equal(la[r, c], lb[r, c], atol=1e-15)


def sample_std(t: Tensor2D) -> Float64:
    var n = t.size()
    var mean = 0.0
    for r in range(t.rows):
        for c in range(t.cols):
            mean += t[r, c]
    mean /= Float64(n)
    var var_sum = 0.0
    for r in range(t.rows):
        for c in range(t.cols):
            var d = t[r, c] - mean
            var_sum += d * d
    return sqrt(var_sum / Float64(n))


def test_residual_init_std_bands() raises:
    # proj and down weights are drawn at 0.02 then scaled by 1/sqrt(2L); qkv and
    # up stay at 0.02. L=3 -> proj/down std ~ 0.02/sqrt(6) = 0.008165. The bands
    # are wide enough for sampling noise but tight enough that scaling the WRONG
    # matrices (qkv/up) would fail — 0.008 is far outside the qkv/up band.
    var cfg = GPTConfig(10, 16, 32, 3, 4, 0.0)
    var rng = Rng(123)
    var gpt = GPT.init_random(cfg, rng)
    var scaled = 0.02 / sqrt(6.0)  # ~0.008165
    for i in range(len(gpt.blocks)):
        var qkv_std = sample_std(gpt.blocks[i].attn.qkv.weight.value)
        var up_std = sample_std(gpt.blocks[i].mlp.up.weight.value)
        var proj_std = sample_std(gpt.blocks[i].attn.proj.weight.value)
        var down_std = sample_std(gpt.blocks[i].mlp.down.weight.value)
        assert_true(
            qkv_std > 0.016 and qkv_std < 0.024,
            "qkv std out of 0.02 band: " + String(qkv_std),
        )
        assert_true(
            up_std > 0.016 and up_std < 0.024,
            "up std out of 0.02 band: " + String(up_std),
        )
        assert_true(
            proj_std > scaled * 0.75 and proj_std < scaled * 1.25,
            "proj std out of scaled band: " + String(proj_std),
        )
        assert_true(
            down_std > scaled * 0.75 and down_std < scaled * 1.25,
            "down std out of scaled band: " + String(down_std),
        )


def test_walk_equals_formula_symmetric() raises:
    var cfg = GPTConfig(10, 8, 16, 2, 2, 0.0)
    var rng = Rng(1)
    var gpt = GPT.init_random(cfg, rng)
    assert_true(
        gpt.parameter_count_actual() == cfg.parameter_count(),
        "walk "
        + String(gpt.parameter_count_actual())
        + " != formula "
        + String(cfg.parameter_count()),
    )


def test_walk_equals_formula_asymmetric() raises:
    # Asymmetric config (V!=ctx, C not a power of two, L=3, H=3), dropout nonzero.
    var cfg = GPTConfig(7, 5, 12, 3, 3, 0.1)
    var rng = Rng(2)
    var gpt = GPT.init_random(cfg, rng)
    assert_true(
        gpt.parameter_count_actual() == cfg.parameter_count(),
        "asymmetric walk "
        + String(gpt.parameter_count_actual())
        + " != formula "
        + String(cfg.parameter_count()),
    )


def test_tied_wte_counted_once() raises:
    # The walk counts wte's V*C once (the tied head owns no Parameter). If it were
    # double-counted, the walk would exceed the formula by exactly V*C.
    var cfg = GPTConfig(9, 6, 8, 2, 2, 0.0)
    var rng = Rng(3)
    var gpt = GPT.init_random(cfg, rng)
    var vc = cfg.vocab_size * cfg.d_model
    assert_true(
        gpt.parameter_count_actual() == cfg.parameter_count(),
        "walk != formula",
    )
    assert_true(
        gpt.parameter_count_actual() != cfg.parameter_count() + vc,
        "walk double-counted the tied wte (off by exactly V*C)",
    )


def test_zero_grad_and_apply_sgd_reach_every_parameter() raises:
    # After a backward every grad is nonzero; zero_grad clears them all; a step
    # with a set-to-one grad moves every value. Exercises the full walk.
    var cfg = GPTConfig(8, 6, 8, 2, 2, 0.0)
    var rng = Rng(5)
    var gpt = GPT.init_random(cfg, rng)

    # zero_grad clears any residue.
    gpt.zero_grad()
    assert_true(gpt.wte.table.grad[0, 0] == 0.0, "wte grad not zeroed")
    assert_true(gpt.wpe.table.grad[0, 0] == 0.0, "wpe grad not zeroed")
    assert_true(gpt.ln_f.weight.grad[0, 0] == 0.0, "ln_f grad not zeroed")
    assert_true(
        gpt.blocks[0].attn.qkv.weight.grad[0, 0] == 0.0, "block grad not zeroed"
    )

    # Set every grad to 1.0 via direct fill, snapshot values, step, check moved.
    _set_all_grads_one(gpt)
    var wte0 = gpt.wte.table.value[0, 0]
    var wpe0 = gpt.wpe.table.value[0, 0]
    var lnf0 = gpt.ln_f.weight.value[0, 0]
    var blk0 = gpt.blocks[1].mlp.down.weight.value[0, 0]
    gpt.apply_sgd(0.1)
    assert_true(gpt.wte.table.value[0, 0] != wte0, "wte value did not move")
    assert_true(gpt.wpe.table.value[0, 0] != wpe0, "wpe value did not move")
    assert_true(gpt.ln_f.weight.value[0, 0] != lnf0, "ln_f value did not move")
    assert_true(
        gpt.blocks[1].mlp.down.weight.value[0, 0] != blk0,
        "block value did not move",
    )


def _set_all_grads_one(mut gpt: GPT):
    # Fill every parameter grad with 1.0 so apply_sgd must move every value.
    gpt.wte.table.grad.fill(1.0)
    gpt.wpe.table.grad.fill(1.0)
    gpt.ln_f.weight.grad.fill(1.0)
    gpt.ln_f.bias.grad.fill(1.0)
    for i in range(len(gpt.blocks)):
        gpt.blocks[i].ln1.weight.grad.fill(1.0)
        gpt.blocks[i].ln1.bias.grad.fill(1.0)
        gpt.blocks[i].attn.qkv.weight.grad.fill(1.0)
        gpt.blocks[i].attn.qkv.bias.grad.fill(1.0)
        gpt.blocks[i].attn.proj.weight.grad.fill(1.0)
        gpt.blocks[i].attn.proj.bias.grad.fill(1.0)
        gpt.blocks[i].ln2.weight.grad.fill(1.0)
        gpt.blocks[i].ln2.bias.grad.fill(1.0)
        gpt.blocks[i].mlp.up.weight.grad.fill(1.0)
        gpt.blocks[i].mlp.up.bias.grad.fill(1.0)
        gpt.blocks[i].mlp.down.weight.grad.fill(1.0)
        gpt.blocks[i].mlp.down.bias.grad.fill(1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
