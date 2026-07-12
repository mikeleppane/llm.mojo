"""Tests for scaled_dot_product_attention, the self/cross attention core.

Oracle goldens come from tests/oracles/attention_reference.py, run once and
frozen here. The rest are structural properties that catch the classic attention
bugs without an oracle: a wrong normalization axis, a multiplied-instead-of-added
mask, a forgotten or squared scale, and the fully-blocked-row NaN. Causality is
proven directly on AttentionResult.weights.
"""

from std.math import isnan, isinf

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.tensor.tensor2d import Tensor2D, from_rows, zeros_2d
from llm.transformer.attention import (
    AttentionResult,
    scaled_dot_product_attention,
)
from llm.transformer.masks import MASKED_SCORE, causal_mask, no_mask


def test_case_a_cross_shaped_oracle() raises:
    """Cross-shaped oracle golden (T_q=3, T_k=4): weights [3,4] and output [3,2]
    match attention_reference.py."""
    var q = from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # [3, 2]
    var k = from_rows(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
    )  # [4, 2]
    var v = from_rows(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]
    )  # [4, 2]
    var r = scaled_dot_product_attention(q, k, v, no_mask(3, 4))

    assert_true(r.weights.rows == 3 and r.weights.cols == 4)
    assert_true(r.output.rows == 3 and r.output.cols == 2)

    # weights [3, 4]
    var w = [
        0.3654723070154712,
        0.1802029521613177,
        0.3654723070154712,
        0.08885243380773993,
        0.1411563112428496,
        0.2862812295857168,
        0.2862812295857168,
        0.2862812295857168,
        0.22118101637021303,
        0.22118101637021303,
        0.44858053295644384,
        0.10905743430313006,
    ]
    for i in range(3):
        for j in range(4):
            assert_almost_equal(r.weights[i, j], w[i * 4 + j], atol=1e-12)
    # output [3, 2]
    var o = [
        0.9086494816464222,
        0.4568228253690489,
        1.0,
        0.2862812295857168,
        0.887876417932917,
        0.5607041150235268,
    ]
    for i in range(3):
        for j in range(2):
            assert_almost_equal(r.output[i, j], o[i * 2 + j], atol=1e-12)


def test_case_d_hand_worked_2x2() raises:
    """Hand-worked D=1 case (scale=1): softmax rows [0.269, 0.731] and [0.5, 0.5],
    output [4.462, 4.0]."""
    var q = from_rows([[1.0], [0.0]])
    var k = from_rows([[1.0], [2.0]])
    var v = from_rows([[3.0], [5.0]])
    var r = scaled_dot_product_attention(q, k, v, no_mask(2, 2))
    assert_almost_equal(r.weights[0, 0], 0.2689414213699951, atol=1e-12)
    assert_almost_equal(r.weights[0, 1], 0.7310585786300049, atol=1e-12)
    assert_almost_equal(r.weights[1, 0], 0.5, atol=1e-12)
    assert_almost_equal(r.weights[1, 1], 0.5, atol=1e-12)
    assert_almost_equal(r.output[0, 0], 4.46211715726001, atol=1e-12)
    assert_almost_equal(r.output[1, 0], 4.0, atol=1e-12)


def test_weight_rows_sum_to_one() raises:
    """Every query row's weights sum to 1 (the softmax invariant survives)."""
    var q = from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    var k = from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    var v = from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    var r = scaled_dot_product_attention(q, k, v, no_mask(3, 4))
    for i in range(3):
        var s = 0.0
        for j in range(4):
            s += r.weights[i, j]
        assert_almost_equal(s, 1.0, atol=1e-12)


def test_causal_weights_zero_above_diagonal() raises:
    """Under a causal mask, every weight strictly above the diagonal is 0."""
    var q = from_rows(
        [[1.0, 2.0], [0.5, -1.0], [3.0, 0.0], [-2.0, 1.5]]
    )  # [4, 2]
    var k = from_rows(
        [[0.0, 1.0], [1.0, 1.0], [2.0, -1.0], [-1.0, 0.5]]
    )  # [4, 2]
    var v = from_rows(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
    )  # [4, 2]
    var r = scaled_dot_product_attention(q, k, v, causal_mask(4))
    for i in range(4):
        for j in range(4):
            if j > i:
                assert_almost_equal(r.weights[i, j], 0.0, atol=1e-9)


def test_identical_values_pass_through() raises:
    """Identical v rows pass through: output equals that vector at every position
    (weights sum to 1)."""
    var q = from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # [3, 2]
    var k = from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [5.0, 5.0]])
    var same = from_rows(
        [[7.0, -2.0], [7.0, -2.0], [7.0, -2.0], [7.0, -2.0]]
    )  # [4, 2], all rows equal
    var r = scaled_dot_product_attention(q, k, same, no_mask(3, 4))
    for i in range(3):
        assert_almost_equal(r.output[i, 0], 7.0, atol=1e-12)
        assert_almost_equal(r.output[i, 1], -2.0, atol=1e-12)


def test_diagonal_only_mask_selects_self() raises:
    """A diagonal-only mask makes weights one-hot on self, so output row i == v row
    i (additive, not multiplicative, masking)."""
    var q = from_rows([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # [3, 2]
    var k = from_rows([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])  # [3, 2]
    var v = from_rows([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])  # [3, 2]
    var mask = zeros_2d(3, 3)
    for i in range(3):
        for j in range(3):
            if i != j:
                mask[i, j] = MASKED_SCORE
    var r = scaled_dot_product_attention(q, k, v, mask)
    for i in range(3):
        # weight is one-hot on the diagonal
        assert_almost_equal(r.weights[i, i], 1.0, atol=1e-9)
        # output row i equals v row i
        assert_almost_equal(r.output[i, 0], v[i, 0], atol=1e-9)
        assert_almost_equal(r.output[i, 1], v[i, 1], atol=1e-9)


def test_fully_blocked_row_is_finite_no_nan() raises:
    """A fully blocked query row stays finite (no NaN/inf) and sums to 1; here the
    scores tie (q=[0,0]) so it degrades to uniform 1/3. A finite MASKED_SCORE is
    what avoids the -inf - -inf = NaN a stable softmax would hit."""
    var q = from_rows([[0.0, 0.0], [0.0, 1.0]])  # [2, 2]; row 0 all zeros
    var k = from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # [3, 2]
    var v = from_rows([[2.0, 0.0], [0.0, 4.0], [6.0, 6.0]])  # [3, 2]
    var mask = zeros_2d(2, 3)
    for j in range(3):
        mask[0, j] = MASKED_SCORE  # block every key for query 0
    var r = scaled_dot_product_attention(q, k, v, mask)
    # row 0: equal underlying scores + full block -> uniform 1/3
    for j in range(3):
        assert_almost_equal(r.weights[0, j], 1.0 / 3.0, atol=1e-9)
    # output row 0 = mean of v rows = ( (2+0+6)/3, (0+4+6)/3 ) = (8/3, 10/3)
    assert_almost_equal(r.output[0, 0], 8.0 / 3.0, atol=1e-9)
    assert_almost_equal(r.output[0, 1], 10.0 / 3.0, atol=1e-9)
    # the guarantee that always holds: no NaN/inf anywhere, weights sum to 1
    for i in range(2):
        var s = 0.0
        for j in range(3):
            assert_true(
                not isnan(r.weights[i, j]) and not isinf(r.weights[i, j])
            )
            s += r.weights[i, j]
        assert_almost_equal(s, 1.0, atol=1e-9)
    for j in range(2):
        assert_true(not isnan(r.output[0, j]) and not isinf(r.output[0, j]))


def test_fully_blocked_nontied_row_stays_finite_not_uniform() raises:
    """A fully blocked row with non-tied scores stays finite and sums to 1, but is
    NOT uniform (additive masking degrades it to softmax of the unmasked scores).
    """
    var q = from_rows([[2.0, 1.0]])  # [1, 2]; non-degenerate query
    var k = from_rows([[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]])  # distinct dots
    var v = from_rows([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])  # [3, 2]
    var mask = zeros_2d(1, 3)
    for j in range(3):
        mask[0, j] = MASKED_SCORE  # fully block query 0
    var r = scaled_dot_product_attention(q, k, v, mask)
    var s = 0.0
    var all_equal = True
    for j in range(3):
        assert_true(not isnan(r.weights[0, j]) and not isinf(r.weights[0, j]))
        s += r.weights[0, j]
        if abs(r.weights[0, j] - 1.0 / 3.0) > 1e-9:
            all_equal = False
    assert_almost_equal(s, 1.0, atol=1e-12)
    # explicitly NOT uniform — the weights track the unmasked score differences
    assert_true(not all_equal)


def test_scale_is_inv_sqrt_dhead() raises:
    """The scale is 1/sqrt(d_head): with dots fixed at [8,2], D 2->4 shifts weights
    only via 1/sqrt(D). Oracle-frozen values."""
    var k2 = from_rows([[8.0, 7.0], [2.0, 3.0]])  # dots = 8, 2
    var v2 = from_rows([[1.0], [0.0]])
    var q2 = from_rows([[1.0, 0.0]])  # D=2
    var r2 = scaled_dot_product_attention(q2, k2, v2, no_mask(1, 2))
    assert_almost_equal(r2.weights[0, 0], 0.9858339641233116, atol=1e-12)
    assert_almost_equal(r2.weights[0, 1], 0.014166035876688408, atol=1e-12)

    var q4 = from_rows([[1.0, 0.0, 0.0, 0.0]])  # D=4
    var k4 = from_rows(
        [[8.0, 7.0, 5.0, 5.0], [2.0, 3.0, 9.0, 9.0]]
    )  # dots 8, 2
    var r4 = scaled_dot_product_attention(q4, k4, v2, no_mask(1, 2))
    assert_almost_equal(r4.weights[0, 0], 0.9525741268224334, atol=1e-12)
    assert_almost_equal(r4.weights[0, 1], 0.04742587317756679, atol=1e-12)


def test_scale_before_mask_order() raises:
    """Scores are scaled before the mask is added: small finite mask entries make
    the two orders diverge observably. Oracle-frozen values."""
    var q = from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    var k = from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    var v = from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    var mask = from_rows(
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -2.0, 0.0], [-3.0, 0.0, 0.0, 0.0]]
    )
    var r = scaled_dot_product_attention(q, k, v, mask)
    var w = [
        0.4124550590010894,
        0.07481515495260488,
        0.4124550590010894,
        0.10027472704521603,
        0.18759243105478166,
        0.38045901986587316,
        0.051489529213472,
        0.38045901986587316,
        0.0139421664228471,
        0.28003589847532273,
        0.5679450010969126,
        0.1380769340049176,
    ]
    for i in range(3):
        for j in range(4):
            assert_almost_equal(r.weights[i, j], w[i * 4 + j], atol=1e-12)
    var o = [
        1.025459572092611,
        0.3869954869084783,
        1.0,
        0.051489529213471996,
        0.8580410355295949,
        0.7099039655673177,
    ]
    for i in range(3):
        for j in range(2):
            assert_almost_equal(r.output[i, j], o[i * 2 + j], atol=1e-12)


def test_shape_mismatches_raise() raises:
    """Mismatched q/k width, v/k length, or mask shape each raise."""
    var q = from_rows([[1.0, 0.0], [0.0, 1.0]])  # [2, 2]
    var k = from_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # [3, 2]
    var v = from_rows([[1.0], [0.0], [1.0]])  # [3, 1]

    # q/k feature width mismatch
    var k_bad = from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    with assert_raises(contains="feature-width mismatch"):
        _ = scaled_dot_product_attention(q, k_bad, v, no_mask(2, 3))
    # v/k length mismatch
    var v_bad = from_rows([[1.0], [0.0]])  # only 2 rows, k has 3
    with assert_raises(contains="length mismatch"):
        _ = scaled_dot_product_attention(q, k, v_bad, no_mask(2, 3))
    # mask wrong shape
    with assert_raises(contains="mask must be"):
        _ = scaled_dot_product_attention(q, k, v, no_mask(2, 2))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
