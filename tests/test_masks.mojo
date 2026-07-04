# Tests for attention masks — additive [T_q, T_k] tensors.
#
# Masks are data, not baked-in behavior: 0.0 means "attend", MASKED_SCORE
# (a large finite negative) means "blocked", and masks compose by plain tensor
# addition (causal + padding = sum). These tests pin the exact entries of each
# builder and that composition keeps a cell blocked whenever either mask blocks
# it — the property the attention core relies on.

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_true,
    TestSuite,
)

from llm.tensor.ops import add
from llm.transformer.masks import (
    MASKED_SCORE,
    no_mask,
    causal_mask,
    key_padding_mask,
)


def test_no_mask_is_all_zeros() raises:
    var m = no_mask(2, 3)  # [T_q=2, T_k=3]
    assert_equal(m.rows, 2)
    assert_equal(m.cols, 3)
    for r in range(2):
        for c in range(3):
            assert_almost_equal(m[r, c], 0.0, atol=1e-12)


def test_causal_mask_hand_checked() raises:
    # causal_mask(4): 0 on and below the diagonal (attend to self and the past),
    # MASKED_SCORE strictly above (cannot see the future). Checked entry by entry.
    #   row 0: [ 0   M   M   M ]
    #   row 1: [ 0   0   M   M ]
    #   row 2: [ 0   0   0   M ]
    #   row 3: [ 0   0   0   0 ]
    var m = causal_mask(4)
    assert_equal(m.rows, 4)
    assert_equal(m.cols, 4)
    for i in range(4):
        for j in range(4):
            if j <= i:
                assert_almost_equal(m[i, j], 0.0, atol=1e-12)
            else:
                assert_almost_equal(m[i, j], MASKED_SCORE, atol=1e-3)


def test_key_padding_mask_blocks_false_columns() raises:
    # keep = [True, False, True, False]; T_q = 3. Every row blocks columns 1 and 3
    # (the padded keys) and leaves columns 0 and 2 open.
    var keep = [True, False, True, False]
    var m = key_padding_mask(keep, 3)  # [T_q=3, T_k=4]
    assert_equal(m.rows, 3)
    assert_equal(m.cols, 4)
    for r in range(3):
        assert_almost_equal(m[r, 0], 0.0, atol=1e-12)
        assert_almost_equal(m[r, 1], MASKED_SCORE, atol=1e-3)
        assert_almost_equal(m[r, 2], 0.0, atol=1e-12)
        assert_almost_equal(m[r, 3], MASKED_SCORE, atol=1e-3)


def test_causal_plus_padding_composition() raises:
    # Composition is the elementwise sum. A cell stays blocked if EITHER mask
    # blocks it: causal blocks the future, padding blocks column 1. Their sum is
    # 0 only where both are open, and <= MASKED_SCORE (still strongly negative)
    # wherever at least one blocks.
    #   causal(3):        padding(keep=[T,F,T], 3):
    #    0  M  M            0  M  0
    #    0  0  M            0  M  0
    #    0  0  0            0  M  0
    #   sum:
    #    0    2M   M
    #    0    M    M
    #    0    M    0
    var keep = [True, False, True]
    var m = add(causal_mask(3), key_padding_mask(keep, 3))
    # open only where both are 0:
    assert_almost_equal(m[0, 0], 0.0, atol=1e-12)
    assert_almost_equal(m[1, 0], 0.0, atol=1e-12)
    assert_almost_equal(m[2, 0], 0.0, atol=1e-12)
    assert_almost_equal(m[2, 2], 0.0, atol=1e-12)
    # blocked by both (double MASKED_SCORE):
    assert_almost_equal(m[0, 1], 2.0 * MASKED_SCORE, atol=1e-3)
    # blocked by one:
    assert_almost_equal(m[0, 2], MASKED_SCORE, atol=1e-3)
    assert_almost_equal(m[1, 1], MASKED_SCORE, atol=1e-3)
    assert_almost_equal(m[1, 2], MASKED_SCORE, atol=1e-3)
    assert_almost_equal(m[2, 1], MASKED_SCORE, atol=1e-3)
    # every entry stays <= 0 (no mask ever opens what another blocked):
    for r in range(3):
        for c in range(3):
            assert_true(m[r, c] <= 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
