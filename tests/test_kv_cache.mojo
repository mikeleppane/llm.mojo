# Tests for the KV cache and the cached decode step.
#
# The load-bearing test here is step-vs-forward EXACT parity at every prefix: the
# cached path must produce BIT-IDENTICAL logits to the uncached batch forward,
# not merely close. That exactness is earned — every stage of the eval forward is
# row-independent (LayerNorm per row, Linear per row, GELU/residual elementwise),
# and the one stage where positions interact, attention, reads cached K/V rows
# that are bit-identical to what the batch path would recompute, using the same
# frozen core in the same op order. So exact equality is the right assertion and
# any mismatch is a real bug — a position off-by-one, a slice bound, a reordered
# sum. Loosening to a tolerance would hide exactly that bug class; it is
# forbidden here. The doll-house uses TWO layers on purpose, so a cross-layer
# cache-indexing bug cannot hide behind a single-layer coincidence.

from std.testing import (
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.config import GPTConfig
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.gpt import GPT
from llm.transformer.kv_cache import KVCache
from llm.utils.random import Rng

# Doll-house GPT: V=11, context 8, d_model 8, 2 layers, 2 heads, dropout 0.
comptime DOLL_V = 11
comptime DOLL_T = 8
comptime DOLL_C = 8
comptime DOLL_L = 2
comptime DOLL_H = 2


def _doll_cfg() -> GPTConfig:
    return GPTConfig(DOLL_V, DOLL_T, DOLL_C, DOLL_L, DOLL_H, 0.0)


def _doll_gpt(seed: UInt64) raises -> GPT:
    var cfg = _doll_cfg()
    var rng = Rng(seed)
    return GPT.init_random(cfg, rng)


def _fixed_sequence() -> List[Int]:
    # A fixed 8-token sequence, every id in [0, V). Distinct-ish so a wrong token
    # or position would shift the logits visibly.
    return [3, 7, 0, 10, 4, 4, 9, 1]


# --- fresh: shapes, length, capacity ------------------------------------------


def test_fresh_shapes_length_capacity() raises:
    var cfg = _doll_cfg()
    var cache = KVCache.fresh(cfg)
    assert_equal(cache.length, 0)
    assert_equal(cache.capacity, DOLL_T)
    assert_equal(len(cache.k), DOLL_L)
    assert_equal(len(cache.v), DOLL_L)
    for i in range(DOLL_L):
        assert_equal(cache.k[i].rows, DOLL_T)  # [context_length, C]
        assert_equal(cache.k[i].cols, DOLL_C)
        assert_equal(cache.v[i].rows, DOLL_T)
        assert_equal(cache.v[i].cols, DOLL_C)


# --- check_compatible named raises --------------------------------------------


def test_check_compatible_wrong_layer_count_raises() raises:
    var cache = KVCache.fresh(_doll_cfg())
    var deeper = GPTConfig(DOLL_V, DOLL_T, DOLL_C, DOLL_L + 1, DOLL_H, 0.0)
    with assert_raises(contains="layers"):
        cache.check_compatible(deeper)


def test_check_compatible_wrong_width_raises() raises:
    # Same layer count and capacity, different d_model — must trip the per-layer
    # width guard (both stay divisible by n_heads).
    var cache = KVCache.fresh(_doll_cfg())
    var wider = GPTConfig(DOLL_V, DOLL_T, DOLL_C * 2, DOLL_L, DOLL_H, 0.0)
    with assert_raises(contains="width"):
        cache.check_compatible(wider)


def test_check_compatible_wrong_capacity_raises() raises:
    var cache = KVCache.fresh(_doll_cfg())
    var longer = GPTConfig(DOLL_V, DOLL_T * 2, DOLL_C, DOLL_L, DOLL_H, 0.0)
    with assert_raises(contains="capacity"):
        cache.check_compatible(longer)


def test_check_compatible_mismatched_kv_layer_count_raises() raises:
    # A publicly built cache (via @fieldwise_init) can carry more key buffers than
    # value buffers. check_compatible must surface that as the NAMED value-layer
    # error before the per-layer loop indexes v[i] into a bounds trap.
    var k = List[Tensor2D]()
    var v = List[Tensor2D]()
    k.append(zeros_2d(DOLL_T, DOLL_C))
    k.append(zeros_2d(DOLL_T, DOLL_C))  # 2 key buffers
    v.append(zeros_2d(DOLL_T, DOLL_C))  # only 1 value buffer
    var cache = KVCache(k^, v^, 0, DOLL_T)
    with assert_raises(contains="value layers"):
        cache.check_compatible(_doll_cfg())


# --- GPT.step full-cache guard and exact-capacity success ---------------------


def test_step_fills_to_capacity_then_raises_when_full() raises:
    # Feeding exactly context_length tokens succeeds; the next step raises NAMED.
    var gpt = _doll_gpt(11)
    var ids = _fixed_sequence()
    var cache = KVCache.fresh(_doll_cfg())
    for t in range(DOLL_T):
        _ = gpt.step(ids[t], cache)
    assert_equal(cache.length, DOLL_T)  # full, all positions consumed

    with assert_raises(contains="full"):
        _ = gpt.step(ids[0], cache)
    assert_equal(cache.length, DOLL_T)  # the failed step did not advance length


# --- THE CENTERPIECE: step-vs-forward exact parity at every prefix ------------


def test_step_matches_forward_at_every_prefix() raises:
    # For every prefix length t in 1..T, the batch forward's LAST logits row over
    # ids[0:t] must equal the t-th cached step's logits row EXACTLY, every one of
    # the V columns, no tolerance. Two layers, so a cross-layer cache-indexing bug
    # cannot hide. This is the part's proof that caching changes the algorithm,
    # not the arithmetic.
    var gpt = _doll_gpt(20260711)
    var ids = _fixed_sequence()
    var cache = KVCache.fresh(_doll_cfg())

    for t in range(1, DOLL_T + 1):
        # Cached step: feed token at position t-1, keep its logits row.
        var step_logits = gpt.step(ids[t - 1], cache)  # [1, V]
        assert_equal(step_logits.rows, 1)
        assert_equal(step_logits.cols, DOLL_V)

        # Batch forward over the whole prefix ids[0:t]; compare its last row.
        var prefix = List[Int]()
        for i in range(t):
            prefix.append(ids[i])
        var full_logits = gpt.forward(prefix)  # [t, V]
        var last = full_logits.rows - 1
        for c in range(DOLL_V):
            # Exact equality on purpose — the summation orders are identical, so
            # any drift is a real bug, not float noise.
            assert_equal(step_logits[0, c], full_logits[last, c])


# --- reset replays bit-identically --------------------------------------------


def test_reset_replays_bit_identically() raises:
    # Run a step sequence, capture the logits; reset; replay the same tokens. The
    # second pass must be bit-identical to the first — reset must leave no live
    # state behind, and the dead rows past length must never be read.
    var gpt = _doll_gpt(99)
    var ids = _fixed_sequence()
    var cache = KVCache.fresh(_doll_cfg())

    var pass1 = List[Float64]()  # flat [T, V] logits of the first pass
    for t in range(DOLL_T):
        var logits = gpt.step(ids[t], cache)
        for c in range(DOLL_V):
            pass1.append(logits[0, c])

    cache.reset()
    assert_equal(cache.length, 0)

    var idx = 0
    for t in range(DOLL_T):
        var logits = gpt.step(ids[t], cache)
        for c in range(DOLL_V):
            assert_equal(logits[0, c], pass1[idx])  # bit-identical replay
            idx += 1


# --- bad token id raises; length is left trustworthy --------------------------


def test_bad_token_id_raises_and_leaves_length_unchanged() raises:
    # A token id outside [0, V) raises via the embedding gather. length must not
    # advance on the raise path, so a caller can always trust it after a failed
    # step. The bad id (== V) is checked BEFORE any cache row is written.
    var gpt = _doll_gpt(5)
    var cache = KVCache.fresh(_doll_cfg())
    # Advance a couple of good steps first, so a nonzero length is on the line.
    var ids = _fixed_sequence()
    _ = gpt.step(ids[0], cache)
    _ = gpt.step(ids[1], cache)
    assert_equal(cache.length, 2)

    with assert_raises(contains="out of range"):
        _ = gpt.step(DOLL_V, cache)  # id == V is out of range
    assert_equal(cache.length, 2)  # unchanged by the failed step

    # And the cache is still usable: the next valid step advances normally.
    var logits = gpt.step(ids[2], cache)
    assert_equal(cache.length, 3)
    assert_equal(logits.cols, DOLL_V)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
