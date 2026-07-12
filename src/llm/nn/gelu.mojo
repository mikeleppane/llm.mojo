"""GELU activation — the tanh approximation.

    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Deliberately the tanh approximation, not the erf-exact GELU: GPT-2's released
weights were trained against this form, and the erf variant differs in the 4th
decimal. Stateless free functions, not a struct.
"""

from std.math import sqrt, tanh, pi

from llm.tensor.tensor2d import Tensor2D, zeros_2d

# sqrt(2/pi), the coefficient on the cubic argument. Bound at compile time: on
# the pinned toolchain std.math.sqrt evaluates in a comptime context, so the
# derivation *is* the documentation and the constant is bit-identical to the
# literal 0.7978845608028654 it would otherwise be spelled as.
comptime SQRT_2_OVER_PI = sqrt(2.0 / pi)

# The cubic-term coefficient. Unlike SQRT_2_OVER_PI this is not derivable — it is
# a fitted constant from the GELU paper (Hendrycks & Gimpel, 2016), named and
# cited rather than left as an inline magic number.
comptime GELU_CUBIC = 0.044715


def gelu(x: Float64) -> Float64:
    """One scalar through the tanh-approximation GELU.

    Args:
        x: Input scalar.

    Returns:
        The value gelu(x). Pure; allocates nothing.
    """
    var inner = SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)
    return 0.5 * x * (1.0 + tanh(inner))


def gelu_rows(x: Tensor2D) -> Tensor2D:
    """Apply GELU elementwise over a tensor.

    Args:
        x: Input, shape [N, C].

    Returns:
        GELU applied to every element, shape [N, C]. Allocates; reads x only.
    """
    var out = zeros_2d(x.rows, x.cols)
    for r in range(x.rows):
        for c in range(x.cols):
            out[r, c] = gelu(x[r, c])
    return out^


def gelu_derivative(x: Float64) -> Float64:
    """Derivative d/dx of the tanh-approximation GELU.

    With u = k(x + c*x^3), k = SQRT_2_OVER_PI, c = GELU_CUBIC, the product rule
    and sech^2(u) = 1 - tanh^2(u), u' = k(1 + 3c*x^2) give

        gelu'(x) = 0.5(1 + tanh(u)) + 0.5*x*(1 - tanh^2(u))*k*(1 + 3c*x^2).

    Args:
        x: Input scalar.

    Returns:
        The value gelu'(x). Pure; allocates nothing. Smooth everywhere (no
        ReLU-style kink).
    """
    var inner = SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)
    var t = tanh(inner)
    var d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_CUBIC * x * x)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner


def gelu_rows_backward(x: Tensor2D, d_out: Tensor2D) raises -> Tensor2D:
    """VJP of gelu_rows: elementwise `d_out * gelu_derivative(x)`.

    GELU is elementwise, so its Jacobian is diagonal. The cache is x itself; the
    cheap tanh terms are recomputed here rather than stored.

    Args:
        x: Forward input, shape [N, C].
        d_out: Upstream gradient, shape [N, C].

    Returns:
        Gradient dL/dx, shape [N, C]. Allocates; reads its args.

    Raises:
        Error: If x and d_out shapes do not match.
    """
    if x.rows != d_out.rows or x.cols != d_out.cols:
        raise Error("gelu_rows_backward shape mismatch")
    var out = zeros_2d(x.rows, x.cols)
    for r in range(x.rows):
        for c in range(x.cols):
            out[r, c] = d_out[r, c] * gelu_derivative(x[r, c])
    return out^
