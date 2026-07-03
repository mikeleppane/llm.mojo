# Tests for GPTConfig and TrainingConfig.
#
# Config validation gets both kinds of test: the failure paths (assert_raises on
# each invalid field) and the success path (a valid config's validate() must
# simply not raise). A failure-only suite would pass a validate() that rejects
# everything, so the success path is not optional. The derived helpers (d_head,
# parameter counts) are pinned to hand-computed values.

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from llm.config import GPTConfig, TrainingConfig


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
