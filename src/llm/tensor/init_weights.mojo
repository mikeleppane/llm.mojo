"""Deterministic weight initialization.

Xavier/Glorot initialization scales the standard deviation by layer width so
activations and gradients neither explode nor vanish: std = sqrt(2 / (fan_in +
fan_out)). It lives in the tensor layer, not utils, because it needs Tensor2D.
"""

from std.math import sqrt

from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.utils.random import Rng


def xavier_2d(mut rng: Rng, fan_in: Int, fan_out: Int) raises -> Tensor2D:
    """Draw a Xavier-initialized weight tensor from `rng`.

    Args:
        rng: Generator, advanced as draws are taken.
        fan_in: Input width.
        fan_out: Output width.

    Returns:
        A [fan_out, fan_in] tensor (the [out, in] convention) of Xavier-scaled
        normal draws. Allocates; deterministic given the generator's state.

    Raises:
        Error: If fan_in or fan_out is not positive.
    """
    if fan_in <= 0 or fan_out <= 0:
        raise Error(
            "xavier_2d: fan_in and fan_out must be positive, got "
            + String(fan_in)
            + " and "
            + String(fan_out)
        )
    var std = sqrt(2.0 / Float64(fan_in + fan_out))
    var t = zeros_2d(fan_out, fan_in)
    for r in range(t.rows):
        for c in range(t.cols):
            t[r, c] = rng.normal(0.0, std)
    return t^
