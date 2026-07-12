"""Tests for the autoregressive generate loop.

generate() is pure assembly over a tested forward and a tested sampler, so these
tests measure the assembly: length contract, determinism (and the greedy no-draw
invariant), stop-token semantics, the sliding-window context crop, and — the
capstone — that a model overfit on a repeating pattern speaks it back when
greedily generated.
"""

from std.math import log

from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from llm.config import GPTConfig
from llm.generation.generate import generate, generate_cached
from llm.generation.sampler import SamplerConfig
from llm.tensor.ops import (
    argmax,
    cross_entropy_rows,
    cross_entropy_rows_backward,
)
from llm.transformer.gpt import GPT
from llm.utils.random import Rng

# Tiny GPT for the loop-mechanics tests: 1 layer, d_model 8, 2 heads, context 8,
# V = 11, dropout 0. Cheap forward, deterministic (dropout off).
comptime TINY_V = 11
comptime TINY_C = 8


def _tiny_gpt(seed: UInt64) raises -> GPT:
    var cfg = GPTConfig(TINY_V, TINY_C, 8, 1, 2, 0.0)
    var rng = Rng(seed)
    return GPT.init_random(cfg, rng)


def _assert_ids_equal(got: List[Int], expected: List[Int]) raises:
    assert_equal(len(got), len(expected))
    for i in range(len(expected)):
        assert_equal(got[i], expected[i])


def _first_index(values: List[Int], target: Int) -> Int:
    for i in range(len(values)):
        if values[i] == target:
            return i
    return -1


# --- length contract ----------------------------------------------------------


def test_length_equals_budget_when_no_stop() raises:
    """Output length equals the token budget when no stop token fires."""
    var gpt = _tiny_gpt(1)
    var prompt: List[Int] = [3, 1, 4]
    var rng = Rng(0)
    var out = generate(
        gpt, prompt, 6, SamplerConfig.standard(), List[Int](), rng
    )
    assert_equal(len(out), 6)


def test_max_new_tokens_zero_returns_empty() raises:
    """A zero budget returns an empty list."""
    var gpt = _tiny_gpt(1)
    var prompt: List[Int] = [3, 1, 4]
    var rng = Rng(0)
    var out = generate(
        gpt, prompt, 0, SamplerConfig.standard(), List[Int](), rng
    )
    assert_equal(len(out), 0)


def test_negative_max_new_tokens_raises() raises:
    """A negative budget raises a named error."""
    var gpt = _tiny_gpt(1)
    var prompt: List[Int] = [3, 1, 4]
    var rng = Rng(0)
    with assert_raises(contains="max_new_tokens"):
        _ = generate(gpt, prompt, -1, SamplerConfig.greedy(), List[Int](), rng)


def test_empty_prompt_raises() raises:
    """An empty prompt raises a named error."""
    var gpt = _tiny_gpt(1)
    var rng = Rng(0)
    with assert_raises(contains="empty prompt"):
        _ = generate(
            gpt, List[Int](), 4, SamplerConfig.greedy(), List[Int](), rng
        )


# --- determinism --------------------------------------------------------------


def test_greedy_is_deterministic_and_draws_nothing() raises:
    """Greedy generation is reproducible and leaves rng.state bit-untouched."""
    var gpt = _tiny_gpt(2)
    var prompt: List[Int] = [7, 2]
    var rng = Rng(4242)
    var before = rng.state
    var a = generate(gpt, prompt, 5, SamplerConfig.greedy(), List[Int](), rng)
    assert_equal(rng.state, before)  # ZERO draws over the whole loop

    var rng2 = Rng(4242)
    var b = generate(gpt, prompt, 5, SamplerConfig.greedy(), List[Int](), rng2)
    _assert_ids_equal(a, b)


def test_equal_seeds_give_identical_sampled_runs() raises:
    """Equal seeds give identical sampled runs and consume the same draws."""
    var gpt = _tiny_gpt(2)
    var prompt: List[Int] = [7, 2]
    var cfg = SamplerConfig(1.0, 0, 1.0)
    var r1 = Rng(2024)
    var r2 = Rng(2024)
    var a = generate(gpt, prompt, 8, cfg, List[Int](), r1)
    var b = generate(gpt, prompt, 8, cfg, List[Int](), r2)
    _assert_ids_equal(a, b)
    assert_equal(r1.state, r2.state)  # same number of draws


# --- stop tokens --------------------------------------------------------------


def test_stop_token_appended_then_halts() raises:
    """The triggering stop token is appended, then the loop halts short of budget.
    """
    var gpt = _tiny_gpt(3)
    var prompt: List[Int] = [5, 9]
    var budget = 10

    # Greedy with no stop is deterministic; use its 3rd emitted token as the stop.
    var r1 = _rng()
    var full = generate(
        gpt, prompt, budget, SamplerConfig.greedy(), List[Int](), r1
    )
    var stop = full[2]
    var expected_len = _first_index(full, stop) + 1  # first occurrence, 1-based

    var stops: List[Int] = [stop]
    var r2 = _rng()
    var out = generate(gpt, prompt, budget, SamplerConfig.greedy(), stops, r2)
    assert_equal(len(out), expected_len)
    assert_equal(out[len(out) - 1], stop)  # ends with the stop token
    assert_true(len(out) < budget, "stop did not shorten the run")
    for i in range(len(out)):
        assert_equal(out[i], full[i])  # a prefix of the unstopped run


def test_stop_id_in_prompt_does_not_halt() raises:
    """A stop id occurring only in the prompt is ignored; only emitted tokens halt.
    """
    # Premise (asserted below): the greedy stream does not re-emit prompt[0], so
    # the stopped and unstopped runs coincide at full budget.
    var gpt = _tiny_gpt(3)
    var prompt: List[Int] = [5, 9]
    var budget = 6

    var r1 = _rng()
    var full = generate(
        gpt, prompt, budget, SamplerConfig.greedy(), List[Int](), r1
    )
    # prompt[0] = 5 is in the prompt; require it is NOT among the emitted tokens.
    assert_true(
        _first_index(full, prompt[0]) == -1,
        "test premise broke: prompt token was re-emitted",
    )
    var stops: List[Int] = [prompt[0]]
    var r2 = _rng()
    var out = generate(gpt, prompt, budget, SamplerConfig.greedy(), stops, r2)
    assert_equal(len(out), budget)  # ran to budget, prompt occurrence ignored
    _assert_ids_equal(out, full)


def test_empty_stop_list_runs_to_budget() raises:
    """An empty stop list runs to the full budget."""
    var gpt = _tiny_gpt(3)
    var prompt: List[Int] = [5, 9]
    var rng = _rng()
    var out = generate(gpt, prompt, 7, SamplerConfig.greedy(), List[Int](), rng)
    assert_equal(len(out), 7)


# --- context crop -------------------------------------------------------------


def test_context_crop_equivalence() raises:
    """Each emitted token matches a manual greedy forward over the hand-cropped last-context_length window (the off-by-one catcher).
    """
    var gpt = _tiny_gpt(5)
    var prompt: List[Int] = [1, 2, 3, 4, 5, 6, 7, 8]  # length == TINY_C
    assert_equal(len(prompt), TINY_C)

    var rng = _rng()
    var out = generate(gpt, prompt, 4, SamplerConfig.greedy(), List[Int](), rng)

    # Independent manual replay of the crop + greedy argmax. The oracle window is
    # built by DROPPING from the front until it fits — deliberately NOT the
    # `start = len - context_length` arithmetic generate.mojo uses — so an
    # off-by-one in that formula cannot be mirrored on both sides and pass.
    var seq = prompt.copy()
    var manual = List[Int]()
    for _ in range(4):
        var ctx = seq.copy()
        while len(ctx) > TINY_C:
            var trimmed = List[Int]()
            for i in range(1, len(ctx)):
                trimmed.append(ctx[i])
            ctx = trimmed^
        var logits = gpt.forward(ctx)
        var last = logits.rows - 1
        var row = List[Float64]()
        for c in range(logits.cols):
            row.append(logits[last, c])
        var nxt = argmax(row)
        manual.append(nxt)
        seq.append(nxt)

    _assert_ids_equal(out, manual)


# --- capstone: memorize, then speak -------------------------------------------


def test_capstone_memorize_then_speak() raises:
    """A tiny GPT overfit on a repeating cycle with plain SGD reproduces the memorized continuation exactly when greedily generated.

    Plain SGD (zero_grad / forward_cached / backward / apply_sgd, dropout 0)
    keeps a trainer bug from reading as a generation failure. The pattern is the
    cycle 1->2->3->4->5->1 (each token has a unique successor), trained as one
    fixed sequence.
    """
    var cfg = GPTConfig(
        6, 8, 8, 2, 2, 0.0
    )  # V=6, C=8, d_model 8, 2 heads, 2 layers
    var rng = Rng(7)
    var gpt = GPT.init_random(cfg, rng)

    var ids: List[Int] = [1, 2, 3, 4, 5, 1, 2, 3]  # length 8 = context
    var targets: List[Int] = [2, 3, 4, 5, 1, 2, 3, 4]  # next token of the cycle

    var final_loss = 0.0
    for _ in range(250):
        var fwd = gpt.forward_cached(ids, False, rng)
        var d_logits = cross_entropy_rows_backward(fwd.logits, targets)
        final_loss = cross_entropy_rows(fwd.logits, targets)
        gpt.zero_grad()
        gpt.backward(fwd.cache, d_logits)
        gpt.apply_sgd(0.5)

    # Loss far below the uniform-model baseline log V — the model memorized.
    assert_true(
        final_loss < 0.05,
        "capstone did not overfit: final loss " + String(final_loss),
    )

    # Greedy-generate 6 tokens from [1, 2]: the cycle continuation is 3,4,5,1,2,3.
    # The prompt (2) + 6 emitted stays within context_length (8), so every step's
    # tokens sit at positions the model was trained on — the crop boundary and its
    # position shift are exercised separately in test_context_crop_equivalence, not
    # here (an overfit model has no reason to generalize to unseen positions).
    var prompt: List[Int] = [1, 2]
    var gen_rng = _rng()
    var out = generate(
        gpt, prompt, 6, SamplerConfig.greedy(), List[Int](), gen_rng
    )
    var expected: List[Int] = [3, 4, 5, 1, 2, 3]
    _assert_ids_equal(out, expected)


def _rng() -> Rng:
    """A fresh seeded generator to bind to generate's `mut rng` argument."""
    # Greedy calls never draw from it, but a temporary cannot bind to a mut arg.
    return Rng(0)


# --- generate_cached: parity with generate, and its own contract --------------
#
# generate_cached must be the KV-cached twin of generate: same tokens, same rng
# stream, same contract. The parity tests use a TWO-layer doll-house so a
# cross-layer cache-indexing bug cannot hide, and check both the emitted ids AND
# rng.state (stream parity, not just output parity).


def _tiny_gpt2(seed: UInt64) raises -> GPT:
    """Two-layer doll-house GPT (V=11, context 8, d_model 8, 2 heads, dropout 0).
    """
    var cfg = GPTConfig(TINY_V, TINY_C, 8, 2, 2, 0.0)
    var rng = Rng(seed)
    return GPT.init_random(cfg, rng)


def test_cached_matches_uncached_greedy() raises:
    """Greedy: cached and uncached paths give identical tokens and untouched rng.state.
    """
    var gpt = _tiny_gpt2(2)
    var prompt: List[Int] = [7, 2, 5]
    var r_un = Rng(4242)
    var r_ca = Rng(4242)
    var un = generate(gpt, prompt, 5, SamplerConfig.greedy(), List[Int](), r_un)
    var ca = generate_cached(
        gpt, prompt, 5, SamplerConfig.greedy(), List[Int](), r_ca
    )
    _assert_ids_equal(ca, un)
    assert_equal(r_ca.state, r_un.state)  # both greedy: zero draws either way


def test_cached_matches_uncached_sampled_with_stream_parity() raises:
    """Sampled (top-k and top-p engaged): cached path matches tokens and rng.state, one draw per emitted token.
    """
    var gpt = _tiny_gpt2(3)
    var prompt: List[Int] = [1, 9, 4]
    var cfg = SamplerConfig(0.9, 5, 0.95)  # temperature, top_k=5, top_p=0.95
    var r_un = Rng(2024)
    var r_ca = Rng(2024)
    var un = generate(gpt, prompt, 5, cfg, List[Int](), r_un)
    var ca = generate_cached(gpt, prompt, 5, cfg, List[Int](), r_ca)
    _assert_ids_equal(ca, un)
    assert_equal(r_ca.state, r_un.state)  # same number of draws, same order


def test_cached_overflow_raises_up_front() raises:
    """A prompt plus budget exceeding context_length raises a named error before any generation.
    """
    var gpt = _tiny_gpt2(1)
    var prompt: List[Int] = [1, 2, 3, 4, 5]  # len 5, context is 8
    var rng = _rng()
    with assert_raises(contains="context_length"):
        _ = generate_cached(
            gpt, prompt, 4, SamplerConfig.greedy(), List[Int](), rng
        )  # 5 + 4 = 9 > 8


def test_cached_at_exactly_context_length_succeeds() raises:
    """The prompt-plus-budget == context_length boundary succeeds and matches the uncached path token-for-token.
    """
    var gpt = _tiny_gpt2(1)
    var prompt: List[Int] = [1, 2, 3, 4, 5]  # len 5
    var r_un = Rng(7)
    var r_ca = Rng(7)
    var un = generate(gpt, prompt, 3, SamplerConfig.greedy(), List[Int](), r_un)
    var ca = generate_cached(
        gpt, prompt, 3, SamplerConfig.greedy(), List[Int](), r_ca
    )  # 5 + 3 == 8
    assert_equal(len(ca), 3)
    _assert_ids_equal(ca, un)


def test_cached_empty_prompt_raises() raises:
    """An empty prompt raises a named error."""
    var gpt = _tiny_gpt2(1)
    var rng = _rng()
    with assert_raises(contains="empty prompt"):
        _ = generate_cached(
            gpt, List[Int](), 4, SamplerConfig.greedy(), List[Int](), rng
        )


def test_cached_negative_budget_raises() raises:
    """A negative budget raises a named error."""
    var gpt = _tiny_gpt2(1)
    var prompt: List[Int] = [3, 1, 4]
    var rng = _rng()
    with assert_raises(contains="max_new_tokens"):
        _ = generate_cached(
            gpt, prompt, -1, SamplerConfig.greedy(), List[Int](), rng
        )


def test_cached_zero_budget_is_noop_rng_untouched() raises:
    """A zero budget returns an empty list and leaves rng untouched, even for a sampled config.
    """
    var gpt = _tiny_gpt2(1)
    var prompt: List[Int] = [3, 1, 4]
    var cfg = SamplerConfig(0.9, 5, 0.95)
    var rng = Rng(555)
    var before = rng.state
    var out = generate_cached(gpt, prompt, 0, cfg, List[Int](), rng)
    assert_equal(len(out), 0)
    assert_equal(rng.state, before)  # bit-untouched


def test_cached_zero_budget_noop_ignores_overflow() raises:
    """A zero budget is a no-op for any prompt length, returning [] without raising overflow and leaving rng untouched, matching generate.
    """
    var gpt = _tiny_gpt2(1)
    var long_prompt = List[Int]()
    for i in range(TINY_C + 3):  # longer than context_length (8)
        long_prompt.append((i * 3) % TINY_V)
    var cfg = SamplerConfig(0.9, 5, 0.95)
    var r_ca = Rng(555)
    var before = r_ca.state
    var out_ca = generate_cached(gpt, long_prompt, 0, cfg, List[Int](), r_ca)
    assert_equal(len(out_ca), 0)
    assert_equal(r_ca.state, before)  # bit-untouched
    # generate agrees: 0 budget → [] regardless of prompt length.
    var r_un = Rng(555)
    var out_un = generate(gpt, long_prompt, 0, cfg, List[Int](), r_un)
    _assert_ids_equal(out_ca, out_un)


def test_cached_stop_token_appended_then_halts() raises:
    """Append-then-halt: the triggering stop token is last and the run is shorter than budget, matching generate.
    """
    var gpt = _tiny_gpt2(3)
    var prompt: List[Int] = [5, 9]
    var budget = (
        6  # prompt 2 + 6 == context 8 (the cached path cannot overflow)
    )

    var r1 = _rng()
    var full = generate_cached(
        gpt, prompt, budget, SamplerConfig.greedy(), List[Int](), r1
    )
    var stop = full[2]
    var expected_len = _first_index(full, stop) + 1

    var stops: List[Int] = [stop]
    var r2 = _rng()
    var out = generate_cached(
        gpt, prompt, budget, SamplerConfig.greedy(), stops, r2
    )
    assert_equal(len(out), expected_len)
    assert_equal(out[len(out) - 1], stop)  # ends with the stop token
    assert_true(len(out) < budget, "stop did not shorten the run")
    for i in range(len(out)):
        assert_equal(out[i], full[i])  # a prefix of the unstopped run


def test_cached_stop_id_in_prompt_does_not_halt() raises:
    """A stop id occurring only in the prompt does not halt generation."""
    var gpt = _tiny_gpt2(3)
    var prompt: List[Int] = [5, 9]
    var budget = 5

    var r1 = _rng()
    var full = generate_cached(
        gpt, prompt, budget, SamplerConfig.greedy(), List[Int](), r1
    )
    assert_true(
        _first_index(full, prompt[0]) == -1,
        "test premise broke: prompt token was re-emitted",
    )
    var stops: List[Int] = [prompt[0]]
    var r2 = _rng()
    var out = generate_cached(
        gpt, prompt, budget, SamplerConfig.greedy(), stops, r2
    )
    assert_equal(len(out), budget)  # ran to budget, prompt occurrence ignored
    _assert_ids_equal(out, full)


def test_generate_cached_threaded_is_deterministic() raises:
    """Generation stays reproducible end to end when the tied head's matmul crosses the parallel-kernel threshold.

    The config is sized so the head's per-step matmul_transpose_b ([1, 32] .
    [32000, 32]^T, ~1.0M multiply-adds) drives the parallel kernel; a tiny model
    would stay serial. Random weights are fine — reproducibility, not coherent
    text, is the property under test.
    """
    var cfg = GPTConfig(32000, 16, 32, 1, 2, 0.0)  # V, T, C, L, H, dropout
    var init_rng = Rng(7)
    var gpt = GPT.init_random(cfg, init_rng)
    var prompt: List[Int] = [5, 9, 2, 17]
    var stops = List[Int]()

    # Greedy: pure forward determinism through the threaded head, no draws.
    var g1 = Rng(123)
    var greedy1 = generate_cached(
        gpt, prompt, 6, SamplerConfig.greedy(), stops, g1
    )
    var g2 = Rng(123)
    var greedy2 = generate_cached(
        gpt, prompt, 6, SamplerConfig.greedy(), stops, g2
    )
    _assert_ids_equal(greedy1, greedy2)

    # Nucleus: the sampled stream must also reproduce under the same seed.
    var s1 = Rng(456)
    var nucleus1 = generate_cached(
        gpt, prompt, 6, SamplerConfig(1.0, 0, 0.9), stops, s1
    )
    var s2 = Rng(456)
    var nucleus2 = generate_cached(
        gpt, prompt, 6, SamplerConfig(1.0, 0, 0.9), stops, s2
    )
    _assert_ids_equal(nucleus1, nucleus2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
