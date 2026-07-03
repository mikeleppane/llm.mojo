# LayerNorm — normalize each row to zero mean, unit variance, then scale/shift.
#
# For each row over its C features:
#
#     y = (x - mean) / sqrt(var + eps) * weight + bias
#
# The variance is the *biased* estimator — sum of squared deviations divided by
# C, not C-1 — which is what GPT-2 (and PyTorch's nn.LayerNorm) computes. The
# distinction is not cosmetic: an unbiased (÷C-1) variance shifts every activation
# and would drift the model off GPT-2 parity, so the oracle test freezes biased
# goldens and rejects the unbiased ones.
#
# eps sits *inside* the square root (sqrt(var + eps), not sqrt(var) + eps) — the
# standard placement — so a constant row (variance 0) normalizes to zero instead
# of dividing by zero. weight initializes to ones and bias to zeros, so a freshly
# built LayerNorm is the identity up to the normalization itself.

from std.math import sqrt

from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, ones_2d, zeros_2d

# GPT-2's LayerNorm epsilon, named at module scope. Guards the divide when a
# row's variance is (near) zero; also the value weight loading must match for
# parity.
comptime LAYERNORM_EPS = 1e-5


@fieldwise_init
struct LayerNorm(Copyable, Movable):
    var weight: Parameter  # [1, C], init ones — per-column scale (gamma)
    var bias: Parameter  # [1, C], init zeros — per-column shift (beta)

    @staticmethod
    def init_default(d_model: Int) raises -> LayerNorm:
        # A LayerNorm with weight=ones, bias=zeros over C = d_model channels.
        # Allocates both tensors; raises on non-positive d_model (a degenerate
        # shape). No rng — the default init is deterministic and parameter-free.
        if d_model <= 0:
            raise Error(
                "LayerNorm.init_default: d_model must be positive, got "
                + String(d_model)
            )
        var w = ones_2d(1, d_model)  # [1, C]
        var b = zeros_2d(1, d_model)  # [1, C]
        return LayerNorm(Parameter(w^), Parameter(b^))

    def forward(self, x: Tensor2D) raises -> Tensor2D:
        # [N, C] -> [N, C], normalizing each row independently. Reads self only;
        # allocates the result; raises on a feature-count mismatch. Uses biased
        # variance (÷C) and eps inside the sqrt.
        var c = x.cols
        if c != self.weight.value.cols:
            raise Error(
                "LayerNorm.forward: shape mismatch, expected "
                + String(self.weight.value.cols)
                + " features, got "
                + String(c)
            )
        var out = zeros_2d(x.rows, c)
        var inv_c = 1.0 / Float64(c)
        for r in range(x.rows):
            var mean = 0.0
            for j in range(c):
                mean += x[r, j]
            mean *= inv_c
            var variance = 0.0
            for j in range(c):
                var d = x[r, j] - mean
                variance += d * d
            variance *= inv_c  # biased: divide by C, not C-1
            var inv_std = 1.0 / sqrt(variance + LAYERNORM_EPS)
            for j in range(c):
                var normed = (x[r, j] - mean) * inv_std
                out[r, j] = (
                    normed * self.weight.value[0, j] + self.bias.value[0, j]
                )
        return out^
