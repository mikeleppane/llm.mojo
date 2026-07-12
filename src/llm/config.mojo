"""Model and training configuration.

Two small value structs carry the knobs the rest of the project reads.
`GPTConfig` describes the model's shape; `TrainingConfig` describes a run. Both
have a `validate()` that raises early naming the offending field, so a bad
config fails at the edge instead of as a shape mismatch deep inside a matmul.
"""

from std.math import isfinite


@fieldwise_init
struct GPTConfig(Copyable, Movable, Writable):
    """The model's shape: vocabulary, context, width, depth, and heads."""

    var vocab_size: Int  # V: number of tokens the model can emit
    var context_length: Int  # T: maximum sequence length
    var d_model: Int  # C: model/channel width
    var n_layers: Int  # number of Transformer blocks
    var n_heads: Int  # number of attention heads
    var dropout: Float64  # dropout probability (0.0 disables)

    @staticmethod
    def gpt2_124m() -> GPTConfig:
        """Return the GPT-2 small (124M) preset: vocab 50257, context 1024,
        width 768, 12 layers, 12 heads, dropout 0.1.

        dropout 0.1 is GPT-2's training-time probability; evaluation disables it
        at the layer level. Non-raising by construction, so it can be evaluated
        in a comptime context.
        """
        return GPTConfig(50257, 1024, 768, 12, 12, 0.1)

    def d_head(self) -> Int:
        """Per-head dimension C / H. Assumes d_model is divisible by n_heads."""
        return self.d_model // self.n_heads

    def validate(self) raises:
        """Raise on the first invalid field, naming it.

        Every dimension must be positive and the head split must divide evenly.

        Raises:
            Error: On the first field that violates its constraint.
        """
        if self.vocab_size <= 0:
            raise Error("vocab_size must be positive")
        if self.context_length <= 0:
            raise Error("context_length must be positive")
        if self.d_model <= 0:
            raise Error("d_model must be positive")
        if self.n_layers <= 0:
            raise Error("n_layers must be positive")
        if self.n_heads <= 0:
            raise Error("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise Error("d_model must be divisible by n_heads")
        # Negated range so a non-finite dropout raises too: every comparison with
        # NaN is false, so a bare `dropout < 0 or dropout >= 1` would let NaN (and
        # +inf, which is >= 1) slip through to corrupt the run.
        if not (self.dropout >= 0.0 and self.dropout < 1.0):
            raise Error("dropout must be in [0, 1)")

    def token_embedding_parameter_count(self) -> Int:
        """Parameters in the token embedding table alone: V * C."""
        return self.vocab_size * self.d_model

    def approx_parameter_count(self) -> Int:
        """Rough total: embeddings plus per-layer attention and MLP weights.

        Ignores biases, norms, and weight tying, so it is a back-of-envelope
        estimate. For the exact GPT-2-layout total, use parameter_count().
        """
        var embed = self.vocab_size * self.d_model
        var attn = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        var mlp = 8 * self.d_model * self.d_model  # up + down (4x hidden)
        var per_layer = attn + mlp
        return embed + self.n_layers * per_layer

    def parameter_count(self) -> Int:
        """Return the exact parameter total for the GPT-2 layout, in comptime-safe
        integer arithmetic (does not raise).

        Writing V for vocab_size, C for d_model, T for context_length, L for
        n_layers, the layout is:

            token embedding      V * C          (the tied LM head reuses this)
            positional embedding T * C          (learned, not sinusoidal)
            per block (x L):
              LayerNorm 1        2C             (weight + bias)
              attention QKV      3C^2 + 3C      (fused Q,K,V weights + biases)
              attention proj     C^2  + C       (output projection weight + bias)
              LayerNorm 2        2C             (weight + bias)
              MLP up             4C^2 + 4C      (C -> 4C weight + bias)
              MLP down           4C^2 + C       (4C -> C weight + bias)
            final LayerNorm      2C             (weight + bias)
            LM head              0              (weights tied to token embedding)

        The per-block rows sum to 12C^2 + 13C. For gpt2_124m this totals
        124,439,808 — the published GPT-2 124M figure. The count commits the
        model to biases on every linear, LayerNorm with weight and bias, learned
        positions, and a tied head.
        """
        var c = self.d_model
        var embeddings = self.vocab_size * c + self.context_length * c
        var per_block = 12 * c * c + 13 * c
        var final_norm = 2 * c
        return embeddings + self.n_layers * per_block + final_norm

    def write_to(self, mut writer: Some[Writer]):
        """Render a one-line summary for print(cfg) / String.write(cfg)."""
        writer.write(
            "GPTConfig(vocab_size=",
            self.vocab_size,
            ", context_length=",
            self.context_length,
            ", d_model=",
            self.d_model,
            ", n_layers=",
            self.n_layers,
            ", n_heads=",
            self.n_heads,
            ", dropout=",
            self.dropout,
            ")",
        )


def check_gpt2_contract():
    """Compile-time pin of the GPT-2 124M parameter contract.

    Because gpt2_124m() and parameter_count() are non-raising, the compiler runs
    the count in a comptime context; the assertion fails the build if the preset
    or the arithmetic drifts from 124,439,808. It lives in a function because a
    comptime assert is illegal at module scope; the config test calls it.
    """
    comptime GPT2 = GPTConfig.gpt2_124m()
    comptime assert (
        GPT2.parameter_count() == 124_439_808
    ), "GPT-2 124M parameter contract drifted"


@fieldwise_init
struct TrainingConfig(Copyable, Movable):
    """A run's knobs: batch size, learning rate, step budget, and seed."""

    var batch_size: Int  # B: sequences per optimizer step
    var learning_rate: Float64  # SGD step size
    var max_steps: Int  # total optimizer steps in a run
    var seed: UInt64  # seed for every reproducible draw in the run

    def validate(self) raises:
        """Raise on the first invalid field, naming it.

        Raises:
            Error: On the first field that violates its constraint.
        """
        if self.batch_size <= 0:
            raise Error("batch_size must be positive")
        # isfinite rejects NaN and +inf; the `> 0` bound alone passes both (NaN
        # fails every compare, +inf is > 0) and would poison every optimizer step.
        if not (isfinite(self.learning_rate) and self.learning_rate > 0.0):
            raise Error("learning_rate must be positive")
        if self.max_steps <= 0:
            raise Error("max_steps must be positive")
