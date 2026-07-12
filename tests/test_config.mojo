"""Tests for GPTConfig and TrainingConfig.

Validation gets both failure paths (assert_raises per invalid field) and the
success path (a valid config's validate() must not raise), since a failure-only
suite would pass a validate() that rejects everything. Derived helpers (d_head,
parameter counts) are pinned to hand-computed values.
"""

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
    """The d_head helper is d_model / n_heads (256 / 4 = 64)."""
    var cfg = GPTConfig(4096, 128, 256, 4, 4, 0.0)
    assert_equal(cfg.d_head(), 64)


def test_token_embedding_parameter_count() raises:
    """Token-embedding count is V*C (4096 * 256 = 1048576)."""
    var cfg = GPTConfig(4096, 128, 256, 4, 4, 0.0)
    assert_equal(cfg.token_embedding_parameter_count(), 1048576)


def test_approx_parameter_count() raises:
    """Approx count: embed 4096*256 + 4 layers of (4+8)*256*256 = 4194304."""
    var cfg = GPTConfig(4096, 128, 256, 4, 4, 0.0)
    assert_equal(cfg.approx_parameter_count(), 4194304)


def test_gpt2_preset_values() raises:
    """The GPT-2 preset is pinned field by field so a stray edit to any one number
    is caught on its own."""
    var cfg = GPTConfig.gpt2_124m()
    assert_equal(cfg.vocab_size, 50257)
    assert_equal(cfg.context_length, 1024)
    assert_equal(cfg.d_model, 768)
    assert_equal(cfg.n_layers, 12)
    assert_equal(cfg.n_heads, 12)
    # dropout is the only Float64 field; 0.1 is GPT-2's training-time value.
    assert_almost_equal(cfg.dropout, 0.1)


def test_gpt2_preset_validates() raises:
    """The GPT-2 preset is a valid config (divisible head split, positive dims).
    """
    GPTConfig.gpt2_124m().validate()


def test_gpt2_preset_d_head() raises:
    """The GPT-2 preset's per-head width is 768 / 12 = 64."""
    assert_equal(GPTConfig.gpt2_124m().d_head(), 64)


def test_gpt2_parameter_count_exact() raises:
    """The exact GPT-2-layout total matches the published 124M figure; a single
    missing bias vector would move it."""
    assert_equal(GPTConfig.gpt2_124m().parameter_count(), 124_439_808)


def test_parameter_count_per_layer_delta() raises:
    """Adding one block adds exactly 12*C^2 + 13*C, testing the formula's structure:
    at C=768 the delta is 7_087_872."""
    var base = GPTConfig(50257, 1024, 768, 12, 12, 0.1)
    var one_more = GPTConfig(50257, 1024, 768, 13, 12, 0.1)
    var delta = one_more.parameter_count() - base.parameter_count()
    assert_equal(delta, 12 * 768 * 768 + 13 * 768)
    assert_equal(delta, 7_087_872)


def test_parameter_count_embedding_share() raises:
    """The token-embedding term (V*C) equals token_embedding_parameter_count(), and
    the exact count reconstructs from its documented parts."""
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
    """The preset carries its own vocab literal (config does not import tokenizer);
    this test pins it to GPT2_VOCAB_SIZE."""
    assert_equal(GPTConfig.gpt2_124m().vocab_size, GPT2_VOCAB_SIZE)


def test_gpt2_comptime_contract() raises:
    """The compile-time assertion gpt2_124m().parameter_count() == 124_439_808: if
    it drifts, this file stops compiling."""
    check_gpt2_contract()


def test_valid_config_does_not_raise() raises:
    """A valid config's validate() does not raise (the success path)."""
    var cfg = GPTConfig(4096, 128, 256, 4, 4, 0.0)
    cfg.validate()  # success path: must not raise


def test_invalid_head_split_raises() raises:
    """A d_model not divisible by n_heads raises (250 is not divisible by 6)."""
    var cfg = GPTConfig(4096, 128, 250, 4, 6, 0.0)
    with assert_raises(contains="divisible"):
        cfg.validate()


def test_nonpositive_dim_raises() raises:
    """A non-positive dimension raises (vocab_size 0)."""
    var cfg = GPTConfig(0, 128, 256, 4, 4, 0.0)
    with assert_raises(contains="vocab_size"):
        cfg.validate()


def test_dropout_out_of_range_raises() raises:
    """A dropout outside [0, 1) raises, both below 0 and >= 1."""
    var negative = GPTConfig(4096, 128, 256, 4, 4, -0.5)
    with assert_raises(contains="dropout"):
        negative.validate()
    var too_large = GPTConfig(4096, 128, 256, 4, 4, 1.5)
    with assert_raises(contains="dropout"):
        too_large.validate()


def test_dropout_boundary_values() raises:
    """Dropout values in [0, 1) validate (0.0 means no dropout)."""
    GPTConfig(4096, 128, 256, 4, 4, 0.0).validate()
    GPTConfig(4096, 128, 256, 4, 4, 0.9).validate()


def test_writable_summary() raises:
    """The config's written summary includes its field values."""
    var cfg = GPTConfig(4096, 128, 256, 4, 4, 0.0)
    var text = String.write(cfg)
    assert_true("vocab_size=4096" in text)
    assert_true("d_model=256" in text)


def test_training_config_valid() raises:
    """A valid TrainingConfig's validate() does not raise (the success path)."""
    var tc = TrainingConfig(32, 0.0003, 1000, 42)
    tc.validate()  # success path: must not raise


def test_training_config_rejects_zero_lr() raises:
    """A zero learning rate raises."""
    var tc = TrainingConfig(32, 0.0, 1000, 42)
    with assert_raises(contains="learning_rate"):
        tc.validate()


def test_training_config_rejects_bad_batch() raises:
    """A non-positive batch size raises."""
    var tc = TrainingConfig(0, 0.0003, 1000, 42)
    with assert_raises(contains="batch_size"):
        tc.validate()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
