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
struct LayerNormCache(Copyable, Movable):
    # What LayerNorm.backward needs: the input x plus the per-row mean and rstd
    # (1/sqrt(var+eps)) the forward already computed. The normalized x̂ is
    # recomputed from these in backward — storing two scalars per row is cheaper
    # than the full [N, C] x̂, and it keeps the eps convention in one place. Valid
    # only for the forward call that produced it.
    var x: Tensor2D  # [N, C]
    var mean: List[Float64]  # [N] — per-row mean μ
    var rstd: List[Float64]  # [N] — per-row 1/sqrt(var + eps)


@fieldwise_init
struct LayerNormForward(Copyable, Movable):
    # forward_cached's output plus the cache its backward consumes.
    var output: Tensor2D  # [N, C]
    var cache: LayerNormCache

    def split(deinit self, mut cache_slot: LayerNormCache) -> Tensor2D:
        # Consume this forward: the cache moves into the caller's slot and the
        # output is returned. Lets an assembly site take both pieces by move
        # instead of copying each out of a live struct (a field cannot be
        # transferred with `^`).
        cache_slot = self.cache^
        return self.output^


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
        # weight and bias must both be [1, C]; validate the bias here rather than
        # read out of bounds at the per-column scale/shift.
        if self.bias.value.rows != 1 or self.bias.value.cols != c:
            raise Error(
                "LayerNorm.forward: bias must be [1, "
                + String(c)
                + "], got ["
                + String(self.bias.value.rows)
                + ", "
                + String(self.bias.value.cols)
                + "]"
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

    def forward_cached(self, var x: Tensor2D) raises -> LayerNormForward:
        # [N, C] -> LayerNormForward: the same normalization as forward, plus the
        # cache backward needs (x and the per-row mean and rstd). Takes x by value
        # and MOVES it into the cache after reading it (no copy) — the caller
        # hands over `x^` when x is dead or `x.copy()` when it still needs it.
        # Reads self; allocates the output and two per-row vectors; raises on a
        # feature-count or bias-shape mismatch (same guards as forward). Uses
        # biased variance (÷C) and eps inside the sqrt. The cache is valid only
        # for this call.
        var c = x.cols
        if c != self.weight.value.cols:
            raise Error(
                "LayerNorm.forward_cached: shape mismatch, expected "
                + String(self.weight.value.cols)
                + " features, got "
                + String(c)
            )
        if self.bias.value.rows != 1 or self.bias.value.cols != c:
            raise Error(
                "LayerNorm.forward_cached: bias must be [1, "
                + String(c)
                + "], got ["
                + String(self.bias.value.rows)
                + ", "
                + String(self.bias.value.cols)
                + "]"
            )
        var out = zeros_2d(x.rows, c)
        var means = List[Float64]()
        var rstds = List[Float64]()
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
            means.append(mean)
            rstds.append(inv_std)
            for j in range(c):
                var normed = (x[r, j] - mean) * inv_std
                out[r, j] = (
                    normed * self.weight.value[0, j] + self.bias.value[0, j]
                )
        return LayerNormForward(out^, LayerNormCache(x^, means^, rstds^))

    def backward(
        mut self, cache: LayerNormCache, d_out: Tensor2D
    ) raises -> Tensor2D:
        # Full per-row derivation of dx (C features). Forward:
        #   μ = mean(x),  v = mean((x-μ)^2),  r = 1/sqrt(v + eps),
        #   x̂_j = (x_j - μ) r,  y_j = γ_j x̂_j + β_j.
        # Upstream d_out = dL/dy. Write a_j = dL/dx̂_j = d_out_j γ_j. Both μ and r
        # depend on every x_k, so x̂ must be differentiated through them:
        #   ∂μ/∂x_k = 1/C
        #   ∂v/∂x_k = (2/C)(x_k - μ)                [Σ_j (x_j - μ) = 0 kills the μ term]
        #   ∂r/∂x_k = -1/2 (v+eps)^{-3/2} ∂v/∂x_k = -(r^2/C) x̂_k
        #   ∂x̂_i/∂x_k = (δ_ik - 1/C) r + (x_i - μ) ∂r/∂x_k
        #             = r[ δ_ik - 1/C - (1/C) x̂_i x̂_k ].
        # Contracting a against this Jacobian gives the three-term result:
        #   dx_k = Σ_i a_i ∂x̂_i/∂x_k = r ( a_k - mean(a) - x̂_k mean(a ⊙ x̂) ).
        # The two subtracted terms are projections: mean(a) removes the component
        # along the ones vector (μ carries no gradient), and x̂_k mean(a⊙x̂) removes
        # the component along x̂ (the scale r carries no gradient). Dropping either
        # is the canonical LayerNorm backward bug; the orthogonality test catches
        # it analytically. Parameter grads accumulate over rows:
        #   dγ_j += Σ_r d_out_{r,j} x̂_{r,j},   dβ_j += Σ_r d_out_{r,j}.
        # Mutates self.weight.grad and self.bias.grad; allocates and returns dx
        # [N, C]; raises on a shape mismatch against the cache.
        var c = self.weight.value.cols
        var n = cache.x.rows
        if cache.x.cols != c:
            raise Error(
                "LayerNorm.backward: cached input width "
                + String(cache.x.cols)
                + " must equal feature count "
                + String(c)
            )
        if d_out.rows != n or d_out.cols != c:
            raise Error(
                "LayerNorm.backward: d_out must be ["
                + String(n)
                + ", "
                + String(c)
                + "], got ["
                + String(d_out.rows)
                + ", "
                + String(d_out.cols)
                + "]"
            )
        if len(cache.mean) != n or len(cache.rstd) != n:
            raise Error(
                "LayerNorm.backward: cache mean/rstd length must equal rows "
                + String(n)
            )
        var inv_c = 1.0 / Float64(c)
        var d_x = zeros_2d(n, c)
        # This call's parameter gradients accumulate into locals first, then add
        # to the Parameters once per column at the end. Adding one fully-formed
        # delta per call (rather than += inside the row loop) is what makes two
        # backward passes double the grads *exactly*: grad = delta, then
        # grad = delta + delta = 2·delta, with no reordered partial sums.
        var d_gamma = List[Float64]()
        var d_beta = List[Float64]()
        for _ in range(c):
            d_gamma.append(0.0)
            d_beta.append(0.0)
        for r in range(n):
            var mean = cache.mean[r]
            var rstd = cache.rstd[r]
            # First pass: the two row reductions mean(a) and mean(a ⊙ x̂), and the
            # per-column dγ, dβ contributions of this row.
            var sum_a = 0.0
            var sum_a_xhat = 0.0
            for j in range(c):
                var xhat = (cache.x[r, j] - mean) * rstd
                var a = d_out[r, j] * self.weight.value[0, j]
                sum_a += a
                sum_a_xhat += a * xhat
                d_gamma[j] = d_gamma[j] + d_out[r, j] * xhat
                d_beta[j] = d_beta[j] + d_out[r, j]
            var mean_a = sum_a * inv_c
            var mean_a_xhat = sum_a_xhat * inv_c
            # Second pass: assemble the three-term dx now that both means exist.
            for j in range(c):
                var xhat = (cache.x[r, j] - mean) * rstd
                var a = d_out[r, j] * self.weight.value[0, j]
                d_x[r, j] = rstd * (a - mean_a - xhat * mean_a_xhat)
        for j in range(c):
            self.weight.grad[0, j] = self.weight.grad[0, j] + d_gamma[j]
            self.bias.grad[0, j] = self.bias.grad[0, j] + d_beta[j]
        return d_x^
