"""Reference values for the nn-layer Mojo tests — GELU, LayerNorm, Linear, MLP.

This script is *provenance*, not a test-time dependency. It is run **once** by
hand; its printed numbers are frozen as literals into the Mojo test files with a
comment pointing back here. The Mojo tests then stay fully offline — nothing
under ``src/`` or the test suite imports this file (same arrangement as
``gpt2_reference_encoder.py``).

Everything is computed in float64 with NumPy so the frozen goldens are an
independent oracle: the reference math lives here, the implementation lives in
``src/llm/nn/``, and the two are compared only through the frozen literals.

The GELU here is the **tanh approximation** GPT-2 was trained with —

    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

deliberately *not* the erf-exact form. LayerNorm uses **biased** variance
(divide by C, not C-1) and eps 1e-5, matching GPT-2 / PyTorch ``nn.LayerNorm``.

Run:  pixi run python tests/oracles/nn_reference.py
"""

from __future__ import annotations

import math

import numpy as np

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
GELU_CUBIC = 0.044715
LN_EPS = 1e-5


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU, tanh approximation (elementwise)."""
    inner = SQRT_2_OVER_PI * (x + GELU_CUBIC * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Per-row LayerNorm with biased variance (÷C) and eps 1e-5."""
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)  # numpy var is biased (ddof=0) by default
    norm = (x - mean) / np.sqrt(var + LN_EPS)
    return norm * weight + bias


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Linear with the [out, in] weight convention: x @ W^T + b."""
    return x @ weight.T + bias


def dump(label: str, arr: np.ndarray) -> None:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    print(f"{label}:")
    for v in flat:
        print(f"    {v!r}")


def main() -> None:
    np.set_printoptions(precision=17)

    # --- GELU scalar goldens ---
    xs = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    print("# GELU (tanh approx) at x in {-3,-1,-0.5,0,0.5,1,3}")
    for x, y in zip(xs, gelu(xs)):
        print(f"    gelu({x!r}) = {y!r}")
    print(f"    gelu(10.0)  = {float(gelu(np.array([10.0]))[0])!r}")
    print(f"    gelu(-10.0) = {float(gelu(np.array([-10.0]))[0])!r}")
    print()

    # --- LayerNorm 3x4 golden (distinguishes biased vs unbiased variance) ---
    ln_in = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [-1.0, 0.0, 2.0, 5.0],
        ]
    )
    ones = np.ones(4)
    zeros = np.zeros(4)
    print("# LayerNorm 3x4, weight=ones bias=zeros, biased var, eps 1e-5")
    dump("layernorm_ones_zeros", layernorm(ln_in, ones, zeros))
    # For contrast: the SAME input with UNBIASED variance (÷C-1). If the Mojo
    # code accidentally divides by C-1, its first row matches THESE instead.
    mean = ln_in.mean(axis=1, keepdims=True)
    var_unbiased = ln_in.var(axis=1, keepdims=True, ddof=1)
    unbiased_row0 = ((ln_in - mean) / np.sqrt(var_unbiased + LN_EPS))[0]
    dump("layernorm_UNBIASED_row0_do_not_match", unbiased_row0)
    print()

    ln_w = np.array([0.5, 1.0, 1.5, 2.0])
    ln_b = np.array([0.1, -0.1, 0.2, -0.2])
    print("# LayerNorm 3x4, weight/bias per-column, biased var, eps 1e-5")
    dump("layernorm_weight_bias", layernorm(ln_in, ln_w, ln_b))
    print()

    # --- Linear small golden, [out, in] convention ---
    lin_w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # [out=2, in=3]
    lin_b = np.array([0.5, -0.5])  # [out=2]
    lin_x = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])  # [N=2, in=3]
    print("# Linear [out,in]=[2,3], forward x @ W^T + b")
    dump("linear_out", linear(lin_x, lin_w, lin_b))
    print()

    # --- Tiny MLP composition: up -> gelu -> down ---
    up_w = np.array([[0.5, -0.5], [1.0, 0.0], [0.0, 1.0]])  # [hidden=3, C=2]
    up_b = np.array([0.1, 0.2, -0.1])  # [hidden=3]
    down_w = np.array([[1.0, 0.5, -1.0], [0.0, 1.0, 2.0]])  # [C=2, hidden=3]
    down_b = np.array([0.0, 0.5])  # [C=2]
    mlp_x = np.array([[1.0, 2.0], [-1.0, 0.5]])  # [N=2, C=2]
    h = linear(mlp_x, up_w, up_b)
    a = gelu(h)
    y = linear(a, down_w, down_b)
    print("# MLP d_model=2 d_hidden=3, forward up -> gelu -> down")
    dump("mlp_out", y)


if __name__ == "__main__":
    main()
