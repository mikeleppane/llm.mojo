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


def gelu_derivative(x: Float64) -> Float64:
    # d/dx of the tanh-approximation GELU. Write the forward as
    #     gelu(x) = 0.5 x (1 + tanh(u)),   u = k (x + c x^3),
    # with k = SQRT_2_OVER_PI and c = GELU_CUBIC. The product rule gives
    #     gelu'(x) = 0.5 (1 + tanh(u)) + 0.5 x * sech^2(u) * u'
    # and, using sech^2(u) = 1 - tanh^2(u) and u' = k (1 + 3 c x^2),
    #     gelu'(x) = 0.5 (1 + tanh(u)) + 0.5 x (1 - tanh^2(u)) k (1 + 3 c x^2).
    # Pure; allocates nothing; cannot raise (tanh is finite for every finite x).
    # The tanh form is smooth everywhere — no ReLU-style kink — so its derivative
    # has no discontinuity for a finite difference to trip over. Asymptotes:
    # gelu'(x) -> 1 as x -> +inf and -> 0 as x -> -inf.
    var inner = SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)
    var t = tanh(inner)
    var d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_CUBIC * x * x)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner


def gelu_rows_backward(x: Tensor2D, d_out: Tensor2D) raises -> Tensor2D:
    # VJP of gelu_rows. GELU is elementwise, so its Jacobian is diagonal and the
    # backward is a plain elementwise product: dL/dx[i, j] = d_out[i, j] *
    # gelu_derivative(x[i, j]). The cache is x itself — the input — because the
    # cheap tanh terms are recomputed here rather than stored. Shapes [N, C] and
    # [N, C] -> [N, C]. Reads its args; allocates the result; raises on a shape
    # mismatch (x and d_out must line up entry for entry).
    if x.rows != d_out.rows or x.cols != d_out.cols:
        raise Error("gelu_rows_backward shape mismatch")
    var out = zeros_2d(x.rows, x.cols)
    for r in range(x.rows):
        for c in range(x.cols):
            out[r, c] = d_out[r, c] * gelu_derivative(x[r, c])
    return out^
