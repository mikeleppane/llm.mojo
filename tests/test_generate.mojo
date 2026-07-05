# Tests for the autoregressive generate loop.
#
# generate() is pure assembly over a tested forward and a tested sampler, so these
# tests measure the ASSEMBLY: the length contract, determinism (and the greedy
# no-draw invariant carried up from sample_next), stop-token semantics
# (appended-then-halt, prompt occurrences ignored), the sliding-window context
# crop (the off-by-one catcher), and — as the capstone — that a model overfit on a
# repeating pattern actually SPEAKS that pattern back when greedily generated.
#
# The non-capstone tests run on a random tiny GPT: loop mechanics don't need a
# trained model, only a real forward. The capstone trains with plain Part XIII SGD
# (not the Part XIV trainer) so a trainer regression can never masquerade as a
# generation failure.

from std.math import log

from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from llm.config import GPTConfig
from llm.generation.generate import generate
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
    var gpt = _tiny_gpt(1)
    var prompt: List[Int] = [3, 1, 4]
    var rng = Rng(0)
    var out = generate(
        gpt, prompt, 6, SamplerConfig.standard(), List[Int](), rng
    )
    assert_equal(len(out), 6)


def test_max_new_tokens_zero_returns_empty() raises:
    var gpt = _tiny_gpt(1)
    var prompt: List[Int] = [3, 1, 4]
    var rng = Rng(0)
    var out = generate(
        gpt, prompt, 0, SamplerConfig.standard(), List[Int](), rng
    )
    assert_equal(len(out), 0)


def test_negative_max_new_tokens_raises() raises:
    var gpt = _tiny_gpt(1)
    var prompt: List[Int] = [3, 1, 4]
    var rng = Rng(0)
    with assert_raises(contains="max_new_tokens"):
        _ = generate(gpt, prompt, -1, SamplerConfig.greedy(), List[Int](), rng)


def test_empty_prompt_raises() raises:
    var gpt = _tiny_gpt(1)
    var rng = Rng(0)
    with assert_raises(contains="empty prompt"):
        _ = generate(
            gpt, List[Int](), 4, SamplerConfig.greedy(), List[Int](), rng
        )


# --- determinism --------------------------------------------------------------


def test_greedy_is_deterministic_and_draws_nothing() raises:
    # Greedy twice => identical ids, and rng.state is bit-untouched, so a greedy
    # generate can be dropped anywhere into a seeded pipeline.
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
    # The triggering stop token is APPENDED, then the loop halts — the output's
    # last element records why it stopped, and the run is shorter than the budget.
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
    # A stop id that occurs only in the PROMPT must not stop anything — only
    # emitted tokens are checked. Premise (asserted): the greedy stream does not
    # re-emit prompt[0], so the stopped and unstopped runs coincide at full budget.
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
    var gpt = _tiny_gpt(3)
    var prompt: List[Int] = [5, 9]
    var rng = _rng()
    var out = generate(gpt, prompt, 7, SamplerConfig.greedy(), List[Int](), rng)
    assert_equal(len(out), 7)


# --- context crop -------------------------------------------------------------


def test_context_crop_equivalence() raises:
    # Prompt at exactly context_length, generate 4 more. Each emitted token must
    # equal a manual greedy forward over the HAND-CROPPED last-context_length
    # window. This is the off-by-one catcher: a crop that kept context_length + 1
    # tokens would raise on the positional bounds; one that cropped from the front
    # or dropped a token would diverge here.
    var gpt = _tiny_gpt(5)
    var prompt: List[Int] = [1, 2, 3, 4, 5, 6, 7, 8]  # length == TINY_C
    assert_equal(len(prompt), TINY_C)

    var rng = _rng()
    var out = generate(gpt, prompt, 4, SamplerConfig.greedy(), List[Int](), rng)

    # Independent manual replay of the crop + greedy argmax.
    var seq = prompt.copy()
    var manual = List[Int]()
    for _ in range(4):
        var start = len(seq) - TINY_C
        if start < 0:
            start = 0
        var ctx = List[Int]()
        for i in range(start, len(seq)):
            ctx.append(seq[i])
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
    # Overfit a tiny GPT on a pure repeating cycle with PLAIN SGD (the Part XIII
    # surface: zero_grad / forward_cached / backward / apply_sgd, dropout 0), then
    # greedy-generate from a 2-token prompt and assert it reproduces the memorized
    # continuation EXACTLY. This is the end-to-end proof that generation connects
    # to what training learned; using plain SGD (not the Part XIV trainer) keeps a
    # trainer bug from reading as a generation failure.
    #
    # The pattern is the cycle 1->2->3->4->5->1 (each token has a unique
    # successor), trained as one fixed sequence.
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
    # A fresh seeded generator. Greedy calls never draw from it, but generate's
    # signature takes `mut rng`, and a temporary cannot bind to a mut argument.
    return Rng(0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
