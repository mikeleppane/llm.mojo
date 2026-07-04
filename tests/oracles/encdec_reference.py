"""Reference values for the encoder-decoder lab Mojo tests.

Provenance, not a test-time dependency: run **once** by hand, its printed
numbers frozen as literals into the lab tests with a comment pointing back here
(same arrangement as ``attention_reference.py``, ``nn_reference.py``, and
``gpt2_reference_encoder.py``). The Mojo tests then stay fully offline — nothing
under ``src/`` or the suite imports this file.

Everything is float64 NumPy so the goldens are an independent oracle: the
reference math lives here, the implementation lives in ``src/llm/lab/``, and the
two meet only through the frozen literals printed below.

Covered forward passes, each with hand-built weights the Mojo test rebuilds via
``from_rows``:

* Cross-multi-head attention (``T_q != T_k``, ``H=2``): the cross-attention core
  wrapped in a separate ``q`` projection and a FUSED ``kv`` projection.
* Pre-LN encoder block: ``a = x + attn(ln1(x)); out = a + mlp(ln2(a))`` — a
  post-LN or LN-on-the-sum wiring produces different numbers and must fail.
* Pre-LN decoder block: ``a = x + self_attn(ln1(x)); b = a + cross(ln2(a), mem);
  out = b + mlp(ln3(b))`` with a causal self-mask and a no-mask cross-mask.

The weights are drawn once from a seeded NumPy generator (offline) and printed
as flat literals; the Mojo side rebuilds the exact tensors from those frozen
numbers, so nothing is derived from the Mojo code.

Run:  pixi run python tests/oracles/encdec_reference.py
"""

from __future__ import annotations

import numpy as np

MASKED_SCORE = -1e9
LAYERNORM_EPS = 1e-5
SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
GELU_CUBIC = 0.044715


# --- primitives, matching the Mojo implementations exactly ---


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    m = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - m)
    return e / e.sum(axis=1, keepdims=True)


def sdpa(q, k, v, mask):
    d = q.shape[1]
    scores = q @ k.T / np.sqrt(float(d))
    scores = scores + mask
    weights = softmax_rows(scores)
    return weights, weights @ v


def linear(x, w, b):
    # w is [out, in]; forward is x @ w.T + b, matching llm.nn.linear.
    return x @ w.T + b


def layernorm(x, w, b):
    # Biased variance (÷C), eps inside the sqrt — matching llm.nn.layernorm.
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    xhat = (x - mean) / np.sqrt(var + LAYERNORM_EPS)
    return xhat * w + b


def gelu(x):
    inner = SQRT_2_OVER_PI * (x + GELU_CUBIC * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def mlp(x, up_w, up_b, down_w, down_b):
    return linear(gelu(linear(x, up_w, up_b)), down_w, down_b)


def mha_self(x, qkv_w, qkv_b, proj_w, proj_b, n_heads, mask):
    # GPT-2 fused self-attention: one Linear(C->3C), contiguous head split.
    c = x.shape[1]
    d = c // n_heads
    qkv = linear(x, qkv_w, qkv_b)  # [T, 3C]
    q_all, k_all, v_all = qkv[:, 0:c], qkv[:, c : 2 * c], qkv[:, 2 * c : 3 * c]
    heads = []
    for h in range(n_heads):
        lo = h * d
        _, oh = sdpa(
            q_all[:, lo : lo + d],
            k_all[:, lo : lo + d],
            v_all[:, lo : lo + d],
            mask,
        )
        heads.append(oh)
    return linear(np.concatenate(heads, axis=1), proj_w, proj_b)


def cross_mha(x, mem, q_w, q_b, kv_w, kv_b, proj_w, proj_b, n_heads, mask):
    # Separate q Linear(C->C) on the decoder stream; FUSED kv Linear(C->2C) on
    # memory; contiguous head split; core sdpa; proj Linear(C->C).
    c = x.shape[1]
    d = c // n_heads
    q_all = linear(x, q_w, q_b)  # [T_q, C]
    kv = linear(mem, kv_w, kv_b)  # [T_k, 2C]
    k_all, v_all = kv[:, 0:c], kv[:, c : 2 * c]
    heads = []
    for h in range(n_heads):
        lo = h * d
        _, oh = sdpa(
            q_all[:, lo : lo + d],
            k_all[:, lo : lo + d],
            v_all[:, lo : lo + d],
            mask,
        )
        heads.append(oh)
    return linear(np.concatenate(heads, axis=1), proj_w, proj_b)


def encoder_block(x, p, n_heads, mask):
    a = x + mha_self(
        layernorm(x, p["ln1_w"], p["ln1_b"]),
        p["qkv_w"],
        p["qkv_b"],
        p["proj_w"],
        p["proj_b"],
        n_heads,
        mask,
    )
    out = a + mlp(
        layernorm(a, p["ln2_w"], p["ln2_b"]),
        p["up_w"],
        p["up_b"],
        p["down_w"],
        p["down_b"],
    )
    return out


def decoder_block(x, mem, p, n_heads, self_mask, cross_mask):
    a = x + mha_self(
        layernorm(x, p["ln1_w"], p["ln1_b"]),
        p["sqkv_w"],
        p["sqkv_b"],
        p["sproj_w"],
        p["sproj_b"],
        n_heads,
        self_mask,
    )
    b = a + cross_mha(
        layernorm(a, p["ln2_w"], p["ln2_b"]),
        mem,
        p["cq_w"],
        p["cq_b"],
        p["ckv_w"],
        p["ckv_b"],
        p["cproj_w"],
        p["cproj_b"],
        n_heads,
        cross_mask,
    )
    out = b + mlp(
        layernorm(b, p["ln3_w"], p["ln3_b"]),
        p["up_w"],
        p["up_b"],
        p["down_w"],
        p["down_b"],
    )
    return out


# --- emit helpers ---


def emit(label: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float64)
    flat = arr.ravel()
    print(f"# {label}: shape {arr.shape}")
    parts = ", ".join(repr(float(v)) for v in flat)
    print(f"# [{parts}]")


def causal_mask(t: int) -> np.ndarray:
    m = np.zeros((t, t))
    for i in range(t):
        for j in range(i + 1, t):
            m[i, j] = MASKED_SCORE
    return m


def main() -> None:
    np.set_printoptions(precision=17)
    rng = np.random.default_rng(20260704)

    def draw(*shape):
        # Small spread so LayerNorm inputs are well-conditioned and GELU stays
        # in its smooth regime; the exact values are frozen either way.
        return np.round(rng.normal(0.0, 0.3, size=shape), 4)

    c, n_heads, hidden = 4, 2, 8

    # ===== Case CA: cross-MHA forward, T_q=3, T_k=4, H=2 =====
    print("===== Case CA: cross-MHA forward (T_q=3, T_k=4, H=2, C=4) =====")
    x_ca = draw(3, c)  # [T_q, C]
    mem_ca = draw(4, c)  # [T_k, C]
    q_w = draw(c, c)
    q_b = draw(c)
    kv_w = draw(2 * c, c)
    kv_b = draw(2 * c)
    proj_w = draw(c, c)
    proj_b = draw(c)
    no_mask_ca = np.zeros((3, 4))
    out_ca = cross_mha(
        x_ca, mem_ca, q_w, q_b, kv_w, kv_b, proj_w, proj_b, n_heads, no_mask_ca
    )
    emit("CA_x", x_ca)
    emit("CA_mem", mem_ca)
    emit("CA_q_w", q_w)
    emit("CA_q_b", q_b)
    emit("CA_kv_w", kv_w)
    emit("CA_kv_b", kv_b)
    emit("CA_proj_w", proj_w)
    emit("CA_proj_b", proj_b)
    emit("CA_output", out_ca)  # [3, 4]
    print()

    # ===== Case ENC: pre-LN encoder block, T=3, C=4, H=2, hidden=8 =====
    print("===== Case ENC: pre-LN encoder block (T=3, C=4, H=2, hidden=8) =====")
    x_enc = draw(3, c)
    penc = {
        "ln1_w": draw(c),
        "ln1_b": draw(c),
        "qkv_w": draw(3 * c, c),
        "qkv_b": draw(3 * c),
        "proj_w": draw(c, c),
        "proj_b": draw(c),
        "ln2_w": draw(c),
        "ln2_b": draw(c),
        "up_w": draw(hidden, c),
        "up_b": draw(hidden),
        "down_w": draw(c, hidden),
        "down_b": draw(c),
    }
    no_mask_enc = np.zeros((3, 3))
    out_enc = encoder_block(x_enc, penc, n_heads, no_mask_enc)
    emit("ENC_x", x_enc)
    for key in [
        "ln1_w",
        "ln1_b",
        "qkv_w",
        "qkv_b",
        "proj_w",
        "proj_b",
        "ln2_w",
        "ln2_b",
        "up_w",
        "up_b",
        "down_w",
        "down_b",
    ]:
        emit("ENC_" + key, penc[key])
    emit("ENC_output", out_enc)  # [3, 4]
    print()

    # ===== Case DEC: pre-LN decoder block, T_tgt=3, T_src=4 =====
    print("===== Case DEC: pre-LN decoder block (T_tgt=3, T_src=4, C=4, H=2) =====")
    x_dec = draw(3, c)  # [T_tgt, C]
    mem_dec = draw(4, c)  # [T_src, C]
    pdec = {
        "ln1_w": draw(c),
        "ln1_b": draw(c),
        "sqkv_w": draw(3 * c, c),
        "sqkv_b": draw(3 * c),
        "sproj_w": draw(c, c),
        "sproj_b": draw(c),
        "ln2_w": draw(c),
        "ln2_b": draw(c),
        "cq_w": draw(c, c),
        "cq_b": draw(c),
        "ckv_w": draw(2 * c, c),
        "ckv_b": draw(2 * c),
        "cproj_w": draw(c, c),
        "cproj_b": draw(c),
        "ln3_w": draw(c),
        "ln3_b": draw(c),
        "up_w": draw(hidden, c),
        "up_b": draw(hidden),
        "down_w": draw(c, hidden),
        "down_b": draw(c),
    }
    self_mask = causal_mask(3)
    cross_mask = np.zeros((3, 4))
    out_dec = decoder_block(
        x_dec, mem_dec, pdec, n_heads, self_mask, cross_mask
    )
    emit("DEC_x", x_dec)
    emit("DEC_mem", mem_dec)
    for key in [
        "ln1_w",
        "ln1_b",
        "sqkv_w",
        "sqkv_b",
        "sproj_w",
        "sproj_b",
        "ln2_w",
        "ln2_b",
        "cq_w",
        "cq_b",
        "ckv_w",
        "ckv_b",
        "cproj_w",
        "cproj_b",
        "ln3_w",
        "ln3_b",
        "up_w",
        "up_b",
        "down_w",
        "down_b",
    ]:
        emit("DEC_" + key, pdec[key])
    emit("DEC_output", out_dec)  # [3, 4]


if __name__ == "__main__":
    main()
