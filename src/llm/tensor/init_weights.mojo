# Deterministic weight initialization.
#
# Xavier/Glorot initialization scales the standard deviation by layer width so
# activations and gradients neither explode nor vanish across layers:
#
#     std = sqrt(2 / (fan_in + fan_out))
#
# This lives in the tensor layer, not utils, on purpose: it needs Tensor2D, and
# the dependency graph runs utils -> tensor. A utils module importing Tensor2D
# would point up the graph (a cycle risk); the RNG it draws from stays in utils,
# which tensor is free to import downward.

from std.math import sqrt

from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.utils.random import Rng


def xavier_2d(mut rng: Rng, fan_in: Int, fan_out: Int) -> Tensor2D:
    # A [fan_out, fan_in] weight tensor (the [out, in] convention) filled with
    # Xavier-scaled normal draws from `rng`. Mutates rng (advances its state);
    # allocates the result. Deterministic given the generator's state.
    var std = sqrt(2.0 / Float64(fan_in + fan_out))
    var t = zeros_2d(fan_out, fan_in)
    for r in range(t.rows):
        for c in range(t.cols):
            t[r, c] = rng.normal(0.0, std)
    return t^
