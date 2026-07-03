# Linear — the affine transform y = x @ W^T + b.
#
# Weight convention is [out, in]: output channel `o` is the contiguous row
# `W[o, :]`, so a row-major dot product reads one weight row against the input.
# This matches xavier_2d's [fan_out, fan_in] layout. (GPT-2's TensorFlow
# checkpoint stores its Conv1D kernels transposed as [in, out]; loading them is a
# transpose-at-load concern, handled where the weights are read, never by bending
# this convention.)
#
# The forward transposes the weight to [in, out] and reuses the tested `matmul`
# rather than introducing a second matmul kernel, then broadcasts the bias row
# across every position.

from llm.nn.parameter import Parameter
from llm.tensor.ops import matmul, transpose
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.utils.random import Rng

# GPT-2's weight-initialization standard deviation. The released model draws
# every linear/embedding weight from normal(0, 0.02); training (a later part)
# runs under this scheme, so it is the factory default. `xavier_2d` stays the
# earlier teaching artifact — a different scheme, not what GPT-2 uses.
comptime GPT2_INIT_STD = 0.02


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
        # result (via transpose and matmul); raises on a feature-count mismatch.
        # The bias row is added to every one of the N positions.
        if x.cols != self.weight.value.cols:
            raise Error(
                "Linear.forward: shape mismatch, expected "
                + String(self.weight.value.cols)
                + " input features, got "
                + String(x.cols)
            )
        var wt = transpose(self.weight.value)  # [out, in] -> [in, out]
        var out = matmul(x, wt)  # [N, in] @ [in, out] -> [N, out]
        for r in range(out.rows):
            for c in range(out.cols):
                out[r, c] = out[r, c] + self.bias.value[0, c]
        return out^
