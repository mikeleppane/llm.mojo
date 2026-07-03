# Model and training configuration.
#
# Two small value structs that carry the knobs the rest of the project reads.
# `GPTConfig` describes the model's shape (vocabulary, context, width, depth,
# heads); `TrainingConfig` describes a run (batch size, learning rate, step
# budget, seed). Both follow the same house rule: a `validate()` that raises
# early with a message naming the offending field, so a bad config fails at the
# edge instead of surfacing as a shape mismatch deep inside a matmul.
#
# `GPTConfig` implements `Writable` so `print(cfg)` renders a readable one-line
# summary — every later debugging session benefits. Reproducibility is a config
# concern from day one: `TrainingConfig` carries the `seed` that drives every
# "random" behavior downstream.


@fieldwise_init
struct GPTConfig(Copyable, Movable, Writable):
    var vocab_size: Int  # V: number of tokens the model can emit
    var context_length: Int  # T: maximum sequence length
    var d_model: Int  # C: model/channel width
    var n_layers: Int  # number of Transformer blocks
    var n_heads: Int  # number of attention heads
    var dropout: Float64  # dropout probability (0.0 disables)

    def d_head(self) -> Int:
        # Per-head dimension C / H. Assumes d_model is divisible by n_heads —
        # validate() enforces that invariant.
        return self.d_model // self.n_heads

    def validate(self) raises:
        # Raise on the first invalid field, naming it. Every dimension must be
        # positive and the head split must divide evenly.
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
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise Error("dropout must be in [0, 1)")

    def token_embedding_parameter_count(self) -> Int:
        # Parameters in the token embedding table alone: V * C.
        return self.vocab_size * self.d_model

    def approx_parameter_count(self) -> Int:
        # Rough total: embeddings + per-layer attention and MLP weights. This
        # ignores biases, norms, and weight tying (the real count arrives with
        # the full GPT), so it is an estimate for a back-of-envelope size check.
        var embed = self.vocab_size * self.d_model
        var attn = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        var mlp = 8 * self.d_model * self.d_model  # up + down (4x hidden)
        var per_layer = attn + mlp
        return embed + self.n_layers * per_layer

    def write_to(self, mut writer: Some[Writer]):
        # Render a one-line summary for print(cfg) / String.write(cfg).
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


@fieldwise_init
struct TrainingConfig(Copyable, Movable):
    var batch_size: Int  # B: sequences per optimizer step
    var learning_rate: Float64  # SGD step size
    var max_steps: Int  # total optimizer steps in a run
    var seed: UInt64  # seed for every reproducible draw in the run

    def validate(self) raises:
        # Raise on the first invalid field, naming it.
        if self.batch_size <= 0:
            raise Error("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise Error("learning_rate must be positive")
        if self.max_steps <= 0:
            raise Error("max_steps must be positive")
