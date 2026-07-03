# Tests for GPTConfig and TrainingConfig.
#
# Config validation gets both kinds of test: the failure paths (assert_raises on
# each invalid field) and the success path (a valid config's validate() must
# simply not raise). A failure-only suite would pass a validate() that rejects
# everything, so the success path is not optional. The derived helpers (d_head,
# parameter counts) are pinned to hand-computed values.

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.config import GPTConfig, TrainingConfig, check_gpt2_contract
from llm.tokenizer import GPT2_VOCAB_SIZE


def test_d_head() raises:
    var cfg = GPTConfig(4096, 128, 256, 4, 4, 0.0)
    assert_equal(cfg.d_head(), 64)


def test_token_embedding_parameter_count() raises:
    var cfg = GPTConfig(4096, 128, 256, 4, 4, 0.0)
    assert_equal(cfg.token_embedding_parameter_count(), 1048576)


def test_approx_parameter_count() raises:
    # embed = 4096*256 = 1048576; per_layer = (4+8)*256*256 = 786432;
    # total = 1048576 + 4*786432 = 4194304.
    var cfg = GPTConfig(4096, 128, 256, 4, 4, 0.0)
    assert_equal(cfg.approx_parameter_count(), 4194304)


def test_gpt2_preset_values() raises:
    # The reference architecture, pinned field by field so a stray edit to any
    # one number is caught on its own rather than hidden inside a total.
    var cfg = GPTConfig.gpt2_124m()
    assert_equal(cfg.vocab_size, 50257)
    assert_equal(cfg.context_length, 1024)
    assert_equal(cfg.d_model, 768)
    assert_equal(cfg.n_layers, 12)
    assert_equal(cfg.n_heads, 12)
    # dropout is the only Float64 field; 0.1 is GPT-2's training-time value.
    assert_almost_equal(cfg.dropout, 0.1)


def test_gpt2_preset_validates() raises:
    # The preset is a valid config: divisible head split, positive dims.
    GPTConfig.gpt2_124m().validate()


def test_gpt2_preset_d_head() raises:
    # 768 / 12 = 64, GPT-2's per-head width.
    assert_equal(GPTConfig.gpt2_124m().d_head(), 64)


def test_gpt2_parameter_count_exact() raises:
    # The headline: the exact GPT-2-layout total, the independently published
    # 124M figure. A single missing bias vector would move this number.
    assert_equal(GPTConfig.gpt2_124m().parameter_count(), 124_439_808)


def test_parameter_count_per_layer_delta() raises:
    # Adding one Transformer block adds exactly the documented per-block cost
    # 12*C^2 + 13*C, so the delta tests the formula's structure, not just its
    # total. C = 768 -> 12*589824 + 13*768 = 7_087_872.
    var base = GPTConfig(50257, 1024, 768, 12, 12, 0.1)
    var one_more = GPTConfig(50257, 1024, 768, 13, 12, 0.1)
    var delta = one_more.parameter_count() - base.parameter_count()
    assert_equal(delta, 12 * 768 * 768 + 13 * 768)
    assert_equal(delta, 7_087_872)


def test_parameter_count_embedding_share() raises:
    # The token-embedding term of the exact breakdown (V*C) must equal the
    # standalone token_embedding_parameter_count(), so the two never drift.
    var cfg = GPTConfig.gpt2_124m()
    var embed = cfg.token_embedding_parameter_count()
    assert_equal(embed, 50257 * 768)
    # The embedding is genuinely a summand of the exact count: reconstruct the
    # total from its documented parts (token + positional embeddings, L blocks
    # at 12C^2+13C, final LayerNorm 2C, tied head 0) and it must match exactly.
    var c = 768
    var reconstructed = embed + 1024 * c + 12 * (12 * c * c + 13 * c) + 2 * c
    assert_equal(cfg.parameter_count(), reconstructed)


def test_gpt2_vocab_matches_tokenizer() raises:
    # config must not import tokenizer (no such edge in the layering graph), so
    # the preset carries its own vocab literal; this test — which may import
    # both — pins the two to the same value.
    assert_equal(GPTConfig.gpt2_124m().vocab_size, GPT2_VOCAB_SIZE)


def test_gpt2_comptime_contract() raises:
    # Calling this forces the compiler to evaluate the compile-time assertion
    # that gpt2_124m().parameter_count() == 124_439_808. If the preset or the
    # arithmetic drifts, this test file stops compiling; the runtime
    # test_gpt2_parameter_count_exact above is the belt to this compile-time brace.
    check_gpt2_contract()


def test_valid_config_does_not_raise() raises:
    var cfg = GPTConfig(4096, 128, 256, 4, 4, 0.0)
    cfg.validate()  # success path: must not raise


def test_invalid_head_split_raises() raises:
    # 250 is not divisible by 6.
    var cfg = GPTConfig(4096, 128, 250, 4, 6, 0.0)
    with assert_raises(contains="divisible"):
        cfg.validate()


def test_nonpositive_dim_raises() raises:
    var cfg = GPTConfig(0, 128, 256, 4, 4, 0.0)
    with assert_raises(contains="vocab_size"):
        cfg.validate()


def test_dropout_out_of_range_raises() raises:
    var negative = GPTConfig(4096, 128, 256, 4, 4, -0.5)
    with assert_raises(contains="dropout"):
        negative.validate()
    var too_large = GPTConfig(4096, 128, 256, 4, 4, 1.5)
    with assert_raises(contains="dropout"):
        too_large.validate()


def test_dropout_boundary_values() raises:
    # 0.0 is valid (no dropout); values in [0, 1) pass.
    GPTConfig(4096, 128, 256, 4, 4, 0.0).validate()
    GPTConfig(4096, 128, 256, 4, 4, 0.9).validate()


def test_writable_summary() raises:
    var cfg = GPTConfig(4096, 128, 256, 4, 4, 0.0)
    var text = String.write(cfg)
    assert_true("vocab_size=4096" in text)
    assert_true("d_model=256" in text)


def test_training_config_valid() raises:
    var tc = TrainingConfig(32, 0.0003, 1000, 42)
    tc.validate()  # success path: must not raise


def test_training_config_rejects_zero_lr() raises:
    var tc = TrainingConfig(32, 0.0, 1000, 42)
    with assert_raises(contains="learning_rate"):
        tc.validate()


def test_training_config_rejects_bad_batch() raises:
    var tc = TrainingConfig(0, 0.0003, 1000, 42)
    with assert_raises(contains="batch_size"):
        tc.validate()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
