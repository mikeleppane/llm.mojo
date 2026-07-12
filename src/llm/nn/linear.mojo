# Linear — the affine transform y = x @ W^T + b.
#
# Weight convention is [out, in]: output channel `o` is the contiguous row
# `W[o, :]`, so a row-major dot product reads one weight row against the input.
# This matches xavier_2d's [fan_out, fan_in] layout. (GPT-2's TensorFlow
# checkpoint stores its Conv1D kernels transposed as [in, out]; loading them is a
# transpose-at-load concern, handled where the weights are read, never by bending
# this convention.)
#
# The forward computes x @ W^T directly with `matmul_transpose_b` (no per-call
# [in, out] transpose copy of the weight), then broadcasts the bias row across
# every position.

from llm.nn.parameter import Parameter
from llm.tensor.ops import matmul_transpose_a, matmul_transpose_b
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.utils.random import Rng

# GPT-2's weight-initialization standard deviation. The released model draws
# every linear/embedding weight from normal(0, 0.02); training (a later part)
# runs under this scheme, so it is the factory default. `xavier_2d` stays the
# earlier teaching artifact — a different scheme, not what GPT-2 uses.
comptime GPT2_INIT_STD = 0.02


@fieldwise_init
struct LinearCache(Copyable, Movable):
    # Everything Linear.backward needs from the forward pass: the input x. dW and
    # dx are both built from x and d_out — the output is never needed — so this is
    # the whole cache. Valid only for the forward call that produced it.
    var x: Tensor2D  # [N, in]


@fieldwise_init
struct LinearForward(Copyable, Movable):
    # What forward_cached returns: the layer output plus the cache its backward
    # will consume. Bundling them keeps the forward/backward pairing explicit —
    # the cache travels with the output it belongs to.
    var output: Tensor2D  # [N, out]
    var cache: LinearCache

    def split(deinit self, mut cache_slot: LinearCache) -> Tensor2D:
        # Consume this forward: the cache moves into the caller's slot and the
        # output is returned. An assembly site that needs both pieces (the output
        # to feed the next stage, the cache to stash for backward) takes them by
        # move rather than copying each field out of a live struct — a struct's
        # field cannot be transferred with `^`, so without this the caller would
        # deep-copy the whole [N, out] output and the cached [N, in] input.
        cache_slot = self.cache^
        return self.output^


@fieldwise_init
struct Linear(Copyable, Movable):
    var weight: Parameter  # [out, in]
    var bias: Parameter  # [1, out], broadcast across rows

    @staticmethod
    def init_random(
        mut rng: Rng, in_features: Int, out_features: Int
    ) raises -> Linear:
        # A Linear with weights drawn from normal(0, 0.02) (GPT-2's scheme) and a
        # zero bias. Mutates rng (advances its state); allocates both tensors;
        # deterministic given the generator's state. Raises on non-positive
        # feature counts, which would produce a degenerate shape.
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
        # [N, in] -> [N, out] as x @ W^T + b. Reads self only; allocates the
        # result; raises on a feature-count mismatch. The bias row is added to
        # every one of the N positions.
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
        # x @ W^T directly: W is [out, in], so out[n, o] = sum_i x[n, i] *
        # W[o, i] — a[i,k]*b[j,k] with a=x, b=W. No [in, out] transpose copy of
        # the weight per forward, and the same k-ascending accumulation as the
        # transpose-then-matmul spelling it replaces.
        var out = matmul_transpose_b(x, self.weight.value)  # [N, out]
        for r in range(out.rows):
            for c in range(out.cols):
                out[r, c] = out[r, c] + self.bias.value[0, c]
        return out^

    def forward_cached(self, var x: Tensor2D) raises -> LinearForward:
        # [N, in] -> LinearForward: the same output as forward(x), plus the cache
        # backward needs. That cache is just x — dW = d_out^T @ x and
        # dx = d_out @ W, neither of which touches the output — so nothing else is
        # stored. Takes x by value and MOVES it into the cache (no copy): the
        # caller decides at its call site whether to hand over a dead value (`x^`)
        # or keep its own (`x.copy()`). Reads self; allocates the output; raises
        # on the same feature-count mismatch forward does. The cache is valid only
        # for this forward call: pairing it with a different call's d_out computes
        # a gradient for the wrong input.
        var output = self.forward(x)
        return LinearForward(output^, LinearCache(x^))

    def backward(
        mut self, cache: LinearCache, d_out: Tensor2D
    ) raises -> Tensor2D:
        # Reverse of y = x @ W^T + b. Given d_out = dL/dy [N, out]:
        #   dL/dW = d_out^T @ x    [out, N] @ [N, in] -> [out, in]
        #   dL/db = colsum(d_out)                     -> [1, out]
        #   dL/dx = d_out @ W      [N, out] @ [out, in] -> [N, in]   (returned)
        # Derivation from y_no = sum_i x_ni W_oi + b_o:
        #   dy_no/dx_ni = W_oi  -> dL/dx_ni = sum_o d_out_no W_oi = (d_out @ W)_ni
        #   dy_no/dW_oi = x_ni  -> dL/dW_oi = sum_n d_out_no x_ni = (d_out^T @ x)_oi
        #   dy_no/db_o  = 1     -> dL/db_o  = sum_n d_out_no       = colsum(d_out)_o
        # The parameter gradients ACCUMULATE (+=), so two backward passes without
        # a zero_grad() between them sum their contributions — the property a tied
        # weight (one Parameter fed by two paths) depends on. Mutates
        # self.weight.grad and self.bias.grad; allocates and returns dL/dx; raises
        # on a shape mismatch against the cached input.
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
