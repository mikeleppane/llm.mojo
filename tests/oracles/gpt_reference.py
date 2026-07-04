"""Reference values for the GPT-2 block and tiny full-model Mojo tests.

Provenance, not a test-time dependency: run **once** by hand, its printed numbers
frozen as literals into ``tests/test_block.mojo`` and ``tests/test_gpt.mojo``
with a comment pointing back here (same arrangement as ``nn_reference.py``,
``attention_reference.py``, ``encdec_reference.py``). The Mojo tests then stay
fully offline — nothing under ``src/`` or the suite imports this file.

Everything is float64 NumPy so the goldens are an independent oracle: the
reference math lives here, the implementation lives in
``src/llm/transformer/{block,gpt}.mojo``, and the two meet only through the
frozen literals.

Weights are a fixed, deterministic, ASYMMETRIC pattern (``fill`` below) authored
for these tests — NOT drawn from the model's rng and NOT derived from the Mojo
code. The Mojo tests build their layers from the identical pattern, so a golden
mismatch indicts the forward wiring (or a fill transcription error), never the
oracle. Asymmetry is deliberate: a position/transpose/wiring bug cannot cancel
against a symmetric weight.

The GPT-2 block, pre-LN, self-attention only:

    a   = x + attn(ln1(x), causal_mask)      # fused-QKV multi-head self-attention
    out = a + mlp(ln2(a))                     # up -> gelu(tanh) -> down

The tied head: logits = h @ wte.T (h = ln_f(x)), no bias.

Run:  pixi run python tests/oracles/gpt_reference.py
"""

from __future__ import annotations

import math

import numpy as np

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
GELU_CUBIC = 0.044715
LN_EPS = 1e-5
MASKED_SCORE = -1e9


def fill(rows: int, cols: int, base: int) -> np.ndarray:
    """Deterministic asymmetric pattern in [-0.5, 0.5), row-major flat index k:

        v = (((k + base) * 37 + 11) mod 101) / 100 - 0.5

    Integer modular arithmetic then /100 is exact in float64, so Mojo and NumPy
    agree bit-for-bit. `base` offsets each tensor so no two share a pattern. This
    MUST match `fill` in the Mojo tests exactly.
    """
    out = np.empty(rows * cols, dtype=np.float64)
    for k in range(rows * cols):
        out[k] = (((k + base) * 37 + 11) % 101) / 100.0 - 0.5
    return out.reshape(rows, cols)


def gelu(x: np.ndarray) -> np.ndarray:
    inner = SQRT_2_OVER_PI * (x + GELU_CUBIC * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)  # biased (ddof=0)
    norm = (x - mean) / np.sqrt(var + LN_EPS)
    return norm * weight + bias


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """[out, in] weight convention: x @ W^T + b."""
    return x @ weight.T + bias


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    m = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - m)
    return e / e.sum(axis=1, keepdims=True)


def causal_mask(t: int) -> np.ndarray:
    m = np.zeros((t, t))
    for i in range(t):
        for j in range(i + 1, t):
            m[i, j] = MASKED_SCORE
    return m


def mha(
    x: np.ndarray,
    qkv_w: np.ndarray,
    qkv_b: np.ndarray,
    proj_w: np.ndarray,
    proj_b: np.ndarray,
    n_heads: int,
    mask: np.ndarray,
) -> np.ndarray:
    """Fused-QKV multi-head self-attention, contiguous head split (D = C/H)."""
    t, c = x.shape
    d = c // n_heads
    qkv = linear(x, qkv_w, qkv_b)  # [T, 3C]
    q_all, k_all, v_all = qkv[:, :c], qkv[:, c : 2 * c], qkv[:, 2 * c :]
    heads = []
    for h in range(n_heads):
        lo, hi = h * d, (h + 1) * d
        q, k, v = q_all[:, lo:hi], k_all[:, lo:hi], v_all[:, lo:hi]
        scores = q @ k.T / math.sqrt(float(d))
        scores = scores + mask
        w = softmax_rows(scores)
        heads.append(w @ v)
    concat = np.concatenate(heads, axis=1)  # [T, C]
    return linear(concat, proj_w, proj_b)


def mlp(
    x: np.ndarray,
    up_w: np.ndarray,
    up_b: np.ndarray,
    down_w: np.ndarray,
    down_b: np.ndarray,
) -> np.ndarray:
    return linear(gelu(linear(x, up_w, up_b)), down_w, down_b)


class BlockWeights:
    """A block's weights, all from the shared `fill` pattern with distinct bases.

    Bases are spaced so no two tensors collide. Must match the Mojo test's
    block builder exactly (same shapes, same bases).
    """

    def __init__(self, c: int, n_heads: int, d_hidden: int):
        self.n_heads = n_heads
        self.ln1_w = fill(1, c, 10).ravel()
        self.ln1_b = fill(1, c, 20).ravel()
        self.qkv_w = fill(3 * c, c, 100)
        self.qkv_b = fill(1, 3 * c, 200).ravel()
        self.proj_w = fill(c, c, 300)
        self.proj_b = fill(1, c, 400).ravel()
        self.ln2_w = fill(1, c, 30).ravel()
        self.ln2_b = fill(1, c, 40).ravel()
        self.up_w = fill(d_hidden, c, 500)
        self.up_b = fill(1, d_hidden, 600).ravel()
        self.down_w = fill(c, d_hidden, 700)
        self.down_b = fill(1, c, 800).ravel()

    def forward(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        a = x + mha(
            layernorm(x, self.ln1_w, self.ln1_b),
            self.qkv_w,
            self.qkv_b,
            self.proj_w,
            self.proj_b,
            self.n_heads,
            mask,
        )
        out = a + mlp(
            layernorm(a, self.ln2_w, self.ln2_b),
            self.up_w,
            self.up_b,
            self.down_w,
            self.down_b,
        )
        return out


def dump(label: str, arr: np.ndarray) -> None:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    print(f"{label}:  # shape {tuple(arr.shape)}")
    for v in flat:
        print(f"    {v!r}")


def main() -> None:
    np.set_printoptions(precision=17)

    # --- Block forward golden: C=4, H=2, d_hidden=6, T=3, causal mask ---
    c, n_heads, d_hidden, t = 4, 2, 6, 3
    blk = BlockWeights(c, n_heads, d_hidden)
    x = fill(t, c, 0)  # block input, its own base
    mask = causal_mask(t)
    print("# Block forward: pre-LN, C=4 H=2 d_hidden=6 T=3, causal mask")
    print(f"#   input x = fill({t}, {c}, 0)")
    dump("block_out", blk.forward(x, mask))
    print()

    # --- Tiny full-model forward golden: V=5, C=4, H=2, L=2, T=3 ---
    v, c, n_heads, d_hidden, n_layers, t = 5, 4, 2, 6, 2, 3
    wte = fill(v, c, 1000)  # [V, C]
    wpe = fill(8, c, 2000)  # [context_length=8, C]  (only rows 0..T-1 used)
    ln_f_w = fill(1, c, 3000).ravel()
    ln_f_b = fill(1, c, 4000).ravel()
    # Two blocks, bases offset by layer so they differ.
    blocks = []
    for layer in range(n_layers):
        b = BlockWeights(c, n_heads, d_hidden)
        off = 10000 * (layer + 1)
        b.ln1_w = fill(1, c, 10 + off).ravel()
        b.ln1_b = fill(1, c, 20 + off).ravel()
        b.qkv_w = fill(3 * c, c, 100 + off)
        b.qkv_b = fill(1, 3 * c, 200 + off).ravel()
        b.proj_w = fill(c, c, 300 + off)
        b.proj_b = fill(1, c, 400 + off).ravel()
        b.ln2_w = fill(1, c, 30 + off).ravel()
        b.ln2_b = fill(1, c, 40 + off).ravel()
        b.up_w = fill(d_hidden, c, 500 + off)
        b.up_b = fill(1, d_hidden, 600 + off).ravel()
        b.down_w = fill(c, d_hidden, 700 + off)
        b.down_b = fill(1, c, 800 + off).ravel()
        blocks.append(b)

    ids = [1, 3, 4]  # token ids, T=3
    x = wte[ids] + wpe[[0, 1, 2]]  # token + positional embeddings
    mask = causal_mask(t)
    for b in blocks:
        x = b.forward(x, mask)
    h = layernorm(x, ln_f_w, ln_f_b)
    logits = h @ wte.T  # tied head, no bias
    print("# Tiny GPT forward: V=5 C=4 H=2 L=2 T=3, ids=[1,3,4], tied head")
    print("#   wte = fill(5,4,1000); wpe = fill(8,4,2000); ln_f from 3000/4000")
    print("#   block l uses base offset 10000*(l+1)")
    dump("gpt_logits", logits)


if __name__ == "__main__":
    main()
