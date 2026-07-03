# GELU activation — the tanh approximation.
#
#     gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#
# This is deliberately the *tanh approximation*, not the erf-exact GELU. GPT-2's
# released weights were trained against this exact form, so reproducing its logits
# demands it — the erf variant differs in the 4th decimal and would drift parity.
#
# GELU is stateless and parameter-free, so it lives as free functions, not a
# struct: `gelu` for one scalar, `gelu_rows` mapping it over a Tensor2D. Neither
# raises — the tanh form is finite for every finite input.

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
    # One scalar through the tanh-approximation GELU. Pure; allocates nothing;
    # cannot raise (finite for every finite x).
    var inner = SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)
    return 0.5 * x * (1.0 + tanh(inner))


def gelu_rows(x: Tensor2D) -> Tensor2D:
    # Elementwise GELU over an [N, C] tensor -> [N, C]. Reads x; allocates the
    # result; cannot raise. Same math as `gelu` applied to every element.
    var out = zeros_2d(x.rows, x.cols)
    for r in range(x.rows):
        for c in range(x.cols):
            out[r, c] = gelu(x[r, c])
    return out^
