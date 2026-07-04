# Tests for scaled_dot_product_attention — the self/cross attention core.
#
# The oracle goldens (Cases A-D) come from tests/oracles/attention_reference.py,
# run once and frozen here. The rest are structural properties that catch the
# classic attention bugs without an oracle: a wrong normalization axis, a
# multiplied-instead-of-added mask, a forgotten or squared scale, and the
# fully-blocked-row NaN. Nothing here recomputes the weights in a second path —
# causality is proven on AttentionResult.weights directly.

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
    # Golden from tests/oracles/attention_reference.py, Case A: T_q=3, T_k=4,
    # D=2, D_v=2, no mask. T_q != T_k is the cross-attention property; the output
    # shape must be [T_q, D_v] = [3, 2], not [T_k, ...].
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
    # Hand-worked, D=1 so scale = 1/sqrt(1) = 1 (also frozen as Case D):
    #   q=[[1],[0]], k=[[1],[2]], v=[[3],[5]]
    #   scores = q @ k^T = [[1, 2], [0, 0]]
    #   row 0 weights = softmax([1, 2]) = [1/(1+e), e/(1+e)]
    #                 = [0.26894142, 0.73105858]
    #   row 1 weights = softmax([0, 0]) = [0.5, 0.5]
    #   output row 0  = 0.26894142*3 + 0.73105858*5 = 4.46211716
    #   output row 1  = 0.5*3 + 0.5*5                = 4.0
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
    # The softmax invariant must survive all the plumbing: every query row's
    # weights sum to 1. Uses the cross-shaped Case A geometry.
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
    # With a causal mask, every weight strictly above the diagonal is 0 within
    # tolerance — proven directly on the returned weights, not inferred from the
    # output. T_q = T_k = 4. Uses arbitrary q/k/v; only the mask matters here.
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
    # If every row of v is the same vector, the output equals that vector at
    # every query position regardless of q/k/mask — because the weights sum to 1.
    # A wrong normalization axis (columns instead of rows) breaks this instantly.
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
    # A mask that blocks everything except the diagonal collapses each query's
    # weights to a one-hot on its own position, so output row i == v row i
    # exactly. Any multiplicative-mask confusion (mask * scores instead of
    # mask + scores) fails here loudly.
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
    # The load-bearing guarantee of a finite MASKED_SCORE: a query whose entire
    # row is blocked produces NO NaN/inf (with -inf, stable softmax would compute
    # -inf - -inf = NaN and poison everything). The weights stay finite and sum
    # to 1.
    #
    # A subtlety worth naming: additive masking is shift-invariant under softmax,
    # so adding -1e9 to EVERY key of a row cancels in the max-subtraction and the
    # row degrades to softmax(unmasked scores) — uniform ONLY when those scores
    # tie, not in general. Here query 0 is q = [0, 0], so every dot product (and
    # thus every score) is 0; the degraded row is genuinely uniform (1/3 each),
    # and we can pin that exactly. The finiteness assertions below are the part
    # that holds for ANY blocked row regardless of the underlying scores.
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
    # Companion to the tied case: when a fully-blocked row's underlying scores do
    # NOT tie, the row does not go uniform — additive masking is shift-invariant,
    # so it degrades to softmax(unmasked scores). The guarantee that still holds
    # is finiteness and rows summing to 1. Here query 0 has distinct dot products
    # against the keys, so its blocked-row weights are unequal but finite.
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
    # Golden from Case B: q has a single 1 in column 0, so dot(q, k_row) is
    # k_row[0] regardless of D. The two dot products stay fixed at 8 and 2 while
    # D goes 2 -> 4, so only the 1/sqrt(D) factor moves the weights:
    #   D=2: softmax([8, 2] / sqrt(2)) -> [0.98583396, 0.01416604]
    #   D=4: softmax([8, 2] / sqrt(4)) -> [0.95257413, 0.04742587]
    # A squared scale, a 1/sqrt(d_model) scale, or no scale gives other numbers.
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
    # Golden from Case C: same q/k/v as Case A but with a small finite mask. The
    # pinned order scales the raw scores THEN adds the mask; adding the mask
    # before scaling would divide these small entries by sqrt(2) too and give
    # different weights. Small finite entries (not -1e9) make the orders diverge
    # observably.
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
