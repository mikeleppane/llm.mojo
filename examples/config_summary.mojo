"""Print a model and training configuration summary.

Demonstrates that GPTConfig is Writable and that the derived helpers and
validate() are wired up.

Run:
    pixi run mojo run -I src examples/config_summary.mojo
"""

from llm.config import GPTConfig, TrainingConfig


def main() raises:
    """Build, validate, and print a GPTConfig and a TrainingConfig."""
    var cfg = GPTConfig(
        vocab_size=4096,
        context_length=128,
        d_model=256,
        n_layers=4,
        n_heads=4,
        dropout=0.0,
    )
    cfg.validate()
    print(cfg)  # works because GPTConfig is Writable
    print("d_head:", cfg.d_head())
    print("token embedding params:", cfg.token_embedding_parameter_count())
    print("approx total params:", cfg.approx_parameter_count())

    var train = TrainingConfig(
        batch_size=32,
        learning_rate=0.0003,
        max_steps=1000,
        seed=42,
    )
    train.validate()
    print("training:", train.max_steps, "steps at lr", train.learning_rate)
