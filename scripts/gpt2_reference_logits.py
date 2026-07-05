#!/usr/bin/env python3
"""Offline float64 reference logits for the 124M GPT2W v1 parity gate.

Reads OUR converted `checkpoints/gpt2-124m.bin` — the SAME bytes the Mojo loader
consumes — reconstructs the model in float64, runs the full forward for a fixed
prompt's token ids, and prints the last row's logits (argmax + a handful of
values) to be frozen into examples/gpt2_parity_check.mojo. Because both sides read
identical bytes and both compute in float64, the frozen golden is a tight
f64-vs-f64 check (assert at 1e-6 in the example), not a cross-precision one.

This is the single-prompt version of the parity claim; the multi-prompt gauntlet
is a later part. NumPy + stdlib only — no torch, no transformers.

The token ids must match what the Mojo example's GPT2Tokenizer produces for the
same prompt string; pass them with --ids (comma-separated) so both sides consume
the identical sequence.

    python scripts/gpt2_reference_logits.py checkpoints/gpt2-124m.bin \
        --ids 15496,11,314,1101,257,3303,2746,11

Run once by hand during development; its output is frozen, not called at runtime.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

LN_EPS = 1e-5
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
GELU_CUBIC = 0.044715
MASKED_SCORE = -1e9


def read_gpt2w(path: Path) -> tuple[dict[str, int], list[np.ndarray]]:
    """Parse a GPT2W v1 file into (dims, tensors) — tensors float64 in walk order.

    Mirrors the Mojo loader: header line up to '\\n', then a raw little-endian
    float32 payload read in walk order with shapes derived from the dims, widened
    to float64.
    """
    raw = path.read_bytes()
    nl = raw.index(b"\n")
    tokens = raw[:nl].decode("ascii").split(" ")
    if tokens[0] != "GPT2W" or tokens[1] != "v1":
        raise ValueError(f"not a GPT2W v1 file: header {raw[:nl]!r}")
    v, t, c, n_layers, n_heads, count = (int(x) for x in tokens[2:8])
    dims = {"V": v, "T": t, "C": c, "L": n_layers, "H": n_heads, "count": count}

    payload = np.frombuffer(raw[nl + 1 :], dtype="<f4").astype(np.float64)
    if payload.size != count:
        raise ValueError(f"payload has {payload.size} floats, header says {count}")

    cursor = 0

    def take(rows: int, cols: int) -> np.ndarray:
        nonlocal cursor
        n = rows * cols
        block = payload[cursor : cursor + n].reshape(rows, cols)
        cursor += n
        return block

    tensors: list[np.ndarray] = []
    tensors.append(take(v, c))  # wte
    tensors.append(take(t, c))  # wpe
    for _ in range(n_layers):
        tensors.append(take(1, c))  # ln1.w
        tensors.append(take(1, c))  # ln1.b
        tensors.append(take(3 * c, c))  # qkv.w
        tensors.append(take(1, 3 * c))  # qkv.b
        tensors.append(take(c, c))  # proj.w
        tensors.append(take(1, c))  # proj.b
        tensors.append(take(1, c))  # ln2.w
        tensors.append(take(1, c))  # ln2.b
        tensors.append(take(4 * c, c))  # up.w
        tensors.append(take(1, 4 * c))  # up.b
        tensors.append(take(c, 4 * c))  # down.w
        tensors.append(take(1, c))  # down.b
    tensors.append(take(1, c))  # ln_f.w
    tensors.append(take(1, c))  # ln_f.b
    return dims, tensors


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(SQRT_2_OVER_PI * (x + GELU_CUBIC * x**3)))


def layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)  # biased
    return (x - mean) / np.sqrt(var + LN_EPS) * w + b


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w.T + b  # [out, in] convention


def softmax_rows(s: np.ndarray) -> np.ndarray:
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(dims: dict[str, int], tensors: list[np.ndarray], ids: list[int]) -> np.ndarray:
    c, n_layers, n_heads = dims["C"], dims["L"], dims["H"]
    wte, wpe = tensors[0], tensors[1]
    t = len(ids)
    x = wte[ids] + wpe[list(range(t))]
    mask = np.zeros((t, t))
    for i in range(t):
        for j in range(i + 1, t):
            mask[i, j] = MASKED_SCORE
    d = c // n_heads

    idx = 2
    for _ in range(n_layers):
        ln1_w, ln1_b = tensors[idx][0], tensors[idx + 1][0]
        qkv_w, qkv_b = tensors[idx + 2], tensors[idx + 3][0]
        proj_w, proj_b = tensors[idx + 4], tensors[idx + 5][0]
        ln2_w, ln2_b = tensors[idx + 6][0], tensors[idx + 7][0]
        up_w, up_b = tensors[idx + 8], tensors[idx + 9][0]
        down_w, down_b = tensors[idx + 10], tensors[idx + 11][0]
        idx += 12

        qkv = linear(layernorm(x, ln1_w, ln1_b), qkv_w, qkv_b)  # [T, 3C]
        q_all, k_all, v_all = qkv[:, :c], qkv[:, c : 2 * c], qkv[:, 2 * c :]
        heads = []
        for h in range(n_heads):
            lo, hi = h * d, (h + 1) * d
            q, k, vv = q_all[:, lo:hi], k_all[:, lo:hi], v_all[:, lo:hi]
            scores = q @ k.T / math.sqrt(float(d)) + mask
            heads.append(softmax_rows(scores) @ vv)
        attn = linear(np.concatenate(heads, axis=1), proj_w, proj_b)
        a = x + attn
        mlp = linear(gelu(linear(layernorm(a, ln2_w, ln2_b), up_w, up_b)), down_w, down_b)
        x = a + mlp

    lnf_w, lnf_b = tensors[idx][0], tensors[idx + 1][0]
    h_final = layernorm(x, lnf_w, lnf_b)
    return h_final @ wte.T  # tied head -> [T, V]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bin", type=Path, help="path to checkpoints/gpt2-124m.bin")
    parser.add_argument(
        "--ids", required=True,
        help="comma-separated token ids (must match the Mojo example's tokenizer)",
    )
    args = parser.parse_args(argv)

    ids = [int(x) for x in args.ids.split(",")]
    dims, tensors = read_gpt2w(args.bin)
    logits = forward(dims, tensors, ids)
    last = logits[-1]  # [V]

    argmax = int(np.argmax(last))
    print(f"# GPT2W: V={dims['V']} T={dims['T']} C={dims['C']} L={dims['L']} H={dims['H']}")
    print(f"# ids = {ids}")
    print(f"# last-row argmax id = {argmax}  (logit {last[argmax]!r})")
    # A handful of fixed indices to freeze, plus the argmax value. Indices past
    # the vocab are dropped so the script also runs on the doll-house fixture.
    v = dims["V"]
    indices = [j for j in [0, 1, 50, 100, 1000, 10000, 40000] if j < v]
    if argmax not in indices:
        indices.append(argmax)
    for j in indices:
        print(f"logit[{j}] = {last[j]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
