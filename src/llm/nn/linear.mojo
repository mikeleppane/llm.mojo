"""Linear — the affine transform y = x @ W^T + b.

Weight convention is [out, in]: output channel `o` is the contiguous row
`W[o, :]`, so a row-major dot product reads one weight row against the input.
The forward computes x @ W^T directly with `matmul_transpose_b` (no per-call
[in, out] transpose copy of the weight), then broadcasts the bias row across
every position.
"""

from llm.nn.parameter import Parameter
from llm.tensor.ops import matmul_transpose_a, matmul_transpose_b
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.utils.random import Rng

# GPT-2's weight-initialization standard deviation. The released model draws
# every linear/embedding weight from normal(0, 0.02), so it is the factory default.
comptime GPT2_INIT_STD = 0.02


@fieldwise_init
struct LinearCache(Copyable, Movable):
    """Everything Linear.backward needs from the forward: the input x.

    dW and dx are both built from x and d_out — the output is never needed. Valid
    only for the forward call that produced it.
    """

    var x: Tensor2D  # [N, in]


@fieldwise_init
struct LinearForward(Copyable, Movable):
    """forward_cached's output plus the cache its backward consumes.

    Bundling them keeps the forward/backward pairing explicit — the cache travels
    with the output it belongs to.
    """

    var output: Tensor2D  # [N, out]
    var cache: LinearCache

    def split(deinit self, mut cache_slot: LinearCache) -> Tensor2D:
        """Consume this forward: move the cache into `cache_slot`, return the output.

        Lets an assembly site take both pieces by move instead of deep-copying the
        [N, out] output and [N, in] cache out of a live struct (a field cannot be
        transferred with `^`).

        Args:
            cache_slot: Receives the cache by move.

        Returns:
            The output tensor, shape [N, out], by move.
        """
        cache_slot = self.cache^
        return self.output^


@fieldwise_init
struct Linear(Copyable, Movable):
    """The affine transform y = x @ W^T + b, with weight [out, in]."""

    var weight: Parameter  # [out, in]
    var bias: Parameter  # [1, out], broadcast across rows

    @staticmethod
    def init_random(
        mut rng: Rng, in_features: Int, out_features: Int
    ) raises -> Linear:
        """Build a Linear with weights from normal(0, 0.02) (GPT-2's scheme), zero bias.

        Args:
            rng: Random generator, advanced as the weights are drawn.
            in_features: Input feature count.
            out_features: Output feature count.

        Returns:
            A Linear with weight [out, in] and zero bias [1, out]. Allocates both;
            deterministic given rng's state.

        Raises:
            Error: If either feature count is non-positive.
        """
        if in_features <= 0 or out_features <= 0:
            raise Error(
                "Linear.init_random: in_features and out_features must be"
                " positive, got "
                + String(in_features)
                + " and "
                + String(out_features)
            )
        var w = zeros_2d(out_features, in_features)  # [out, in]
        for r in range(out_features):
            for c in range(in_features):
                w[r, c] = rng.normal(0.0, GPT2_INIT_STD)
        var b = zeros_2d(1, out_features)  # [1, out], zeros
        return Linear(Parameter(w^), Parameter(b^))

    def forward(self, x: Tensor2D) raises -> Tensor2D:
        """Compute x @ W^T + b, broadcasting the bias across all N positions.

        Args:
            x: Input activations, shape [N, in].

        Returns:
            Output activations, shape [N, out]. Allocates; reads self only.

        Raises:
            Error: On a feature-count or bias-shape mismatch.
        """
        if x.cols != self.weight.value.cols:
            raise Error(
                "Linear.forward: shape mismatch, expected "
                + String(self.weight.value.cols)
                + " input features, got "
                + String(x.cols)
            )
        # The bias must be [1, out] so it broadcasts across rows; validate it here
        # rather than read out of bounds at the per-column add (out = weight rows).
        var out_features = self.weight.value.rows
        if self.bias.value.rows != 1 or self.bias.value.cols != out_features:
            raise Error(
                "Linear.forward: bias must be [1, "
                + String(out_features)
                + "], got ["
                + String(self.bias.value.rows)
                + ", "
                + String(self.bias.value.cols)
                + "]"
            )
        # x @ W^T directly: W is [out, in], so out[n, o] = sum_i x[n, i] * W[o, i].
        # No [in, out] transpose copy of the weight per forward. matmul_transpose_b
        # is a reassociating SIMD dot, so this differs from a scalar dot by ~k*eps.
        var out = matmul_transpose_b(x, self.weight.value)  # [N, out]
        for r in range(out.rows):
            for c in range(out.cols):
                out[r, c] = out[r, c] + self.bias.value[0, c]
        return out^

    def forward_cached(self, var x: Tensor2D) raises -> LinearForward:
        """Compute forward(x), additionally caching x for backward.

        The cache is just x — dW = d_out^T @ x and dx = d_out @ W, neither of
        which touches the output. Takes x by value and moves it into the cache (no
        copy) — the caller hands over `x^` or `x.copy()`.

        Args:
            x: Input activations, shape [N, in]; moved into the cache.

        Returns:
            A LinearForward with output [N, out] and the x cache. Allocates the
            output; cache valid only for this call.

        Raises:
            Error: On the same feature-count mismatch forward raises.
        """
        var output = self.forward(x)
        return LinearForward(output^, LinearCache(x^))

    def backward(
        mut self, cache: LinearCache, d_out: Tensor2D
    ) raises -> Tensor2D:
        """Backprop through y = x @ W^T + b to dx and the parameter grads.

        Given d_out = dL/dy [N, out]:

            dL/dW = d_out^T @ x    -> [out, in]
            dL/db = colsum(d_out)  -> [1, out]
            dL/dx = d_out @ W      -> [N, in]   (returned)

        The parameter gradients accumulate (+=), so two backward passes without a
        zero_grad() between them sum their contributions — the property a tied
        weight (one Parameter fed by two paths) depends on.

        Args:
            cache: The input x from the paired forward.
            d_out: Upstream gradient dL/dy, shape [N, out].

        Returns:
            Gradient dL/dx, shape [N, in]. Allocates. Mutates self.weight.grad
            and self.bias.grad.

        Raises:
            Error: On a shape mismatch against the cached input.
        """
        var out_features = self.weight.value.rows
        var in_features = self.weight.value.cols
        if d_out.cols != out_features:
            raise Error(
                "Linear.backward: d_out width "
                + String(d_out.cols)
                + " must equal out_features "
                + String(out_features)
            )
        if d_out.rows != cache.x.rows:
            raise Error(
                "Linear.backward: d_out rows "
                + String(d_out.rows)
                + " must equal cached input rows "
                + String(cache.x.rows)
            )
        if cache.x.cols != in_features:
            raise Error(
                "Linear.backward: cached input width "
                + String(cache.x.cols)
                + " must equal in_features "
                + String(in_features)
            )
        var d_weight = matmul_transpose_a(
            d_out, cache.x
        )  # d_out^T @ x [out, in]
        for o in range(out_features):
            for i in range(in_features):
                self.weight.grad[o, i] = self.weight.grad[o, i] + d_weight[o, i]
        for o in range(out_features):
            var col_sum = 0.0
            for n in range(d_out.rows):
                col_sum += d_out[n, o]
            self.bias.grad[0, o] = self.bias.grad[0, o] + col_sum
        return d_out @ self.weight.value  # [N, out] @ [out, in] -> [N, in]
