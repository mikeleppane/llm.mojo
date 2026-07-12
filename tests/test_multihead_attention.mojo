"""Tests for MultiHeadAttention — GPT-2's fused-QKV multi-head self-attention.

These pin the structural contract (shape, parameter layout, count 4*C^2 + 4*C),
determinism, the invalid-config guards, and two behavioral properties that catch
a wrong head split: single-head equivalence (H=1 is the plain qkv -> core -> proj
composition) and causal locality (row 0 attends only to itself).
"""

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    TestSuite,
)

from llm.nn.linear import Linear
from llm.nn.parameter import Parameter
from llm.tensor.ops import slice_cols
from llm.tensor.tensor2d import Tensor2D, from_rows
from llm.transformer.attention import (
    MultiHeadAttention,
    scaled_dot_product_attention,
)
from llm.transformer.masks import causal_mask, no_mask
from llm.utils.random import Rng


def test_forward_shape_contract() raises:
    """`forward` maps [T, C] plus mask [T, T] to [T, C]."""
    var rng = Rng(42)
    var mha = MultiHeadAttention.init_random(rng, 4, 2)
    var x = from_rows(
        [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0], [-2.0, 1.0, 0.0, 3.0]]
    )  # [3, 4]
    var y = mha.forward(x, no_mask(3, 3))
    assert_equal(y.rows, 3)
    assert_equal(y.cols, 4)


def test_parameter_count_reconciles() raises:
    """The layer's real Parameter tensors sum to 4*C^2 + 4*C, with the layout spelled out.
    """
    # qkv is [3C, C] weight (3C^2) + [1, 3C] bias (3C); proj is [C, C] weight
    # (C^2) + [1, C] bias (C).
    var rng = Rng(1)
    var c = 8
    var mha = MultiHeadAttention.init_random(rng, c, 4)
    var actual = (
        mha.qkv.weight.value.size()
        + mha.qkv.bias.value.size()
        + mha.proj.weight.value.size()
        + mha.proj.bias.value.size()
    )
    var expected = 4 * c * c + 4 * c
    assert_equal(actual, expected)
    # spell the layout out so a wrong shape is localized, not just a wrong total
    assert_equal(mha.qkv.weight.value.rows, 3 * c)  # [3C, C]
    assert_equal(mha.qkv.weight.value.cols, c)
    assert_equal(mha.qkv.bias.value.cols, 3 * c)  # [1, 3C]
    assert_equal(mha.proj.weight.value.rows, c)  # [C, C]
    assert_equal(mha.proj.weight.value.cols, c)
    assert_equal(mha.proj.bias.value.cols, c)  # [1, C]


def test_init_random_is_deterministic() raises:
    """Same seed gives identical weights and identical forward output."""
    var rng_a = Rng(7)
    var rng_b = Rng(7)
    var a = MultiHeadAttention.init_random(rng_a, 4, 2)
    var b = MultiHeadAttention.init_random(rng_b, 4, 2)
    var x = from_rows([[1.0, 0.0, -1.0, 2.0], [0.0, 3.0, 1.0, -2.0]])  # [2, 4]
    var ya = a.forward(x, no_mask(2, 2))
    var yb = b.forward(x, no_mask(2, 2))
    for r in range(2):
        for col in range(4):
            assert_almost_equal(ya[r, col], yb[r, col], atol=1e-15)


def test_invalid_config_raises() raises:
    """`init_random` raises on zero heads, non-divisible width, and zero width.
    """
    var rng = Rng(0)
    with assert_raises(contains="n_heads must be positive"):
        _ = MultiHeadAttention.init_random(rng, 4, 0)  # zero heads
    with assert_raises(contains="divisible"):
        _ = MultiHeadAttention.init_random(rng, 6, 4)  # 6 % 4 != 0
    with assert_raises(contains="d_model must be positive"):
        _ = MultiHeadAttention.init_random(rng, 0, 1)  # zero width


def test_single_head_equals_manual_composition() raises:
    """With H=1 the layer equals qkv-projection -> core -> output-projection."""
    # This proves the head plumbing (split thirds, slice heads, run core, concat)
    # is the identity when H=1 — a wrong split axis or dropped third would diverge.
    var rng = Rng(99)
    var c = 4
    var mha = MultiHeadAttention.init_random(rng, c, 1)
    var x = from_rows(
        [[1.0, 2.0, -1.0, 0.5], [0.0, 1.0, 2.0, -3.0], [2.0, -2.0, 0.0, 1.0]]
    )  # [3, 4]
    var mask = causal_mask(3)

    # Manual: fused projection, split Q/K/V thirds, core, output projection.
    var qkv = mha.qkv.forward(x)  # [3, 3C]
    var q_all = slice_cols(qkv, 0, c)
    var k_all = slice_cols(qkv, c, 2 * c)
    var v_all = slice_cols(qkv, 2 * c, 3 * c)
    var core = scaled_dot_product_attention(q_all, k_all, v_all, mask)
    var manual = mha.proj.forward(core.output)  # [3, C]

    var y = mha.forward(x, mask)
    assert_equal(y.rows, 3)
    assert_equal(y.cols, c)
    for r in range(3):
        for col in range(c):
            assert_almost_equal(y[r, col], manual[r, col], atol=1e-12)


def test_multihead_forward_contiguous_split_oracle() raises:
    """H=2 forward matches the oracle's contiguous head split (Case E)."""
    # The single-head test can't distinguish a contiguous split from an
    # interleaved one, and causal locality is split-invariant at row 0 — so this
    # case pins the split axis and head ordering. Expected output (Case E in
    # tests/oracles/attention_reference.py) uses a CONTIGUOUS split ([0:2], [2:4]);
    # an interleaved split ([0,2]/[1,3]) or swapped heads yields different columns.
    var w_qkv = from_rows(
        [
            [0.10, -0.20, 0.30, 0.05],
            [0.40, 0.10, -0.10, 0.20],
            [-0.30, 0.20, 0.10, 0.15],
            [0.05, 0.25, -0.15, 0.30],
            [0.20, -0.10, 0.35, -0.05],
            [0.15, 0.30, 0.10, -0.20],
            [-0.25, 0.05, 0.20, 0.10],
            [0.30, -0.15, 0.05, 0.25],
            [0.10, 0.20, -0.30, 0.15],
            [-0.05, 0.35, 0.15, -0.10],
            [0.25, -0.20, 0.10, 0.30],
            [0.20, 0.10, -0.25, 0.05],
        ]
    )  # [3C=12, C=4]
    var b_qkv = from_rows(
        [
            [
                0.01,
                -0.02,
                0.03,
                0.00,
                0.02,
                -0.01,
                0.00,
                0.04,
                -0.03,
                0.01,
                0.02,
                -0.04,
            ]
        ]
    )  # [1, 12]
    var w_proj = from_rows(
        [
            [0.20, -0.10, 0.30, 0.05],
            [0.15, 0.25, -0.20, 0.10],
            [-0.05, 0.30, 0.10, -0.15],
            [0.25, -0.20, 0.05, 0.30],
        ]
    )  # [C=4, C=4]
    var b_proj = from_rows([[0.05, -0.05, 0.10, 0.00]])  # [1, 4]
    var mha = MultiHeadAttention(
        Linear(Parameter(w_qkv^), Parameter(b_qkv^)),
        Linear(Parameter(w_proj^), Parameter(b_proj^)),
        2,
    )
    var x = from_rows([[1.0, 2.0, -1.0, 0.5], [0.0, 1.0, 2.0, -3.0]])  # [2, 4]
    var y = mha.forward(x, causal_mask(2))
    # E_output, contiguous split, frozen from the oracle
    var expected = [
        0.18075000000000002,
        0.27125,
        0.09249999999999996,
        0.30575,
        -0.2027460178243602,
        0.2308733158146542,
        0.2728490081523048,
        -0.21152733293607015,
    ]
    assert_equal(y.rows, 2)
    assert_equal(y.cols, 4)
    for r in range(2):
        for col in range(4):
            assert_almost_equal(y[r, col], expected[r * 4 + col], atol=1e-12)


def test_causal_row0_ignores_later_positions() raises:
    """Under a causal mask, output row 0 is unchanged by permuting later rows.
    """
    # Query 0 attends only to key 0 and the qkv projection is position-wise, so
    # row 0 depends solely on input row 0 — locality falling out of causality.
    var rng = Rng(123)
    var mha = MultiHeadAttention.init_random(rng, 4, 2)
    var x = from_rows(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [-1.0, -2.0, -3.0, -4.0],
        ]
    )  # [4, 4]
    # same row 0, rows 1..3 permuted (3 -> 1 -> 2 order)
    var x_perm = from_rows(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, -2.0, -3.0, -4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )  # [4, 4]
    var mask = causal_mask(4)
    var y = mha.forward(x, mask)
    var y_perm = mha.forward(x_perm, mask)
    for col in range(4):
        assert_almost_equal(y[0, col], y_perm[0, col], atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
