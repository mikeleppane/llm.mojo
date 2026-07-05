"""Doll-house fixture writer and reference logits for the GPT2W v1 loader tests.

Provenance AND a test-time fixture, two roles in one module:

  * ``write_fixture(path)`` writes a syntactically valid ``GPT2W v1`` weight file
    for a tiny config (V 11, T 8, C 8, L 1, H 2) into ``path``. The Mojo test
    imports this module at test time (same live-Python arrangement as the
    tokenizer's ``gpt2_reference_encoder``), calls ``write_fixture`` to drop the
    file into a tempdir, then loads it with ``load_gpt2`` and checks the result.
  * ``main()`` prints the frozen goldens (individual sentinel slot values and the
    reference logits) that the Mojo test hard-codes inline, so a silently broken
    fixture writer is caught by the frozen numbers even though the same writer
    produced the file under test.

Nothing under ``src/`` or the suite imports this at build time; it is pure
NumPy + stdlib and runs only when the test explicitly imports it.

The GPT2W v1 format (the ONE spec the converter, this fixture, and the Mojo
loader all agree on):

    line 0 (ASCII, newline-terminated):
        GPT2W v1 <V> <T> <C> <L> <H> <param_count>
    then the raw little-endian float32 payload, every parameter tensor back to
    back in THE MODEL'S WALK ORDER, row-major within each tensor:

        wte [V,C], wpe [T,C],
        per block (x L): ln1.w [1,C], ln1.b [1,C],
                         qkv.w [3C,C], qkv.b [1,3C],
                         proj.w [C,C], proj.b [1,C],
                         ln2.w [1,C], ln2.b [1,C],
                         up.w [4C,C], up.b [1,4C],
                         down.w [C,4C], down.b [1,C],
        ln_f.w [1,C], ln_f.b [1,C].

    No per-tensor shape records: the shapes are DERIVED from (V,T,C,L,H) by both
    writer and reader, so the walk is the single source of truth. The header's
    param_count and the payload's byte length cross-check each other.

Weights are a deterministic ASYMMETRIC sentinel (``sentinel`` below): a per-slot
base plus DIFFERENT row and column coefficients, so a transposed square kernel
(proj [C,C]) lands wrong numbers, and a per-tensor base unique to each walk slot,
so a swapped slot (ln1<->ln2, up<->down) or an off-by-one walk lands wrong
numbers. Values are stored as float32 and the reference forward reads them back
widened to float64 — exactly what the loader does — so the goldens compare
float64-against-float64 on identical bytes.

Run:  pixi run python tests/oracles/gpt2_weights_reference.py
"""

from __future__ import annotations

import math
import struct

import numpy as np

# Doll-house config: small enough to freeze goldens, large enough that H=2 and
# the 4C MLP exercise the head split and the two distinct MLP widths.
V, T, C, L, H = 11, 8, 8, 1, 2
D_HIDDEN = 4 * C  # GPT-2's 4x ratio -> 32

MAGIC = "GPT2W v1"
LN_EPS = 1e-5
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
GELU_CUBIC = 0.044715
MASKED_SCORE = -1e9

# Per-slot bases, unique and in WALK ORDER, so a swapped slot or an off-by-one
# walk lands a wrong base somewhere the goldens can see it.
BASE_WTE = 0
BASE_WPE = 1
BASE_LN1_W = 2
BASE_LN1_B = 3
BASE_QKV_W = 4
BASE_QKV_B = 5
BASE_PROJ_W = 6
BASE_PROJ_B = 7
BASE_LN2_W = 8
BASE_LN2_B = 9
BASE_UP_W = 10
BASE_UP_B = 11
BASE_DOWN_W = 12
BASE_DOWN_B = 13
BASE_LNF_W = 14
BASE_LNF_B = 15


def sentinel(rows: int, cols: int, base: int) -> np.ndarray:
    """A deterministic asymmetric [rows, cols] float32 sentinel.

        v(r, c) = 0.01 * base + 0.02 * r - 0.013 * c

    The row and column coefficients differ (0.02 vs -0.013), so v(r, c) !=
    v(c, r): a transposed square kernel is caught. `base` shifts each tensor so
    no two walk slots share a value. Rounded to float32 (the on-disk precision)
    and returned as float32; callers widen to float64 for the reference forward,
    matching what the Mojo loader reads.
    """
    out = np.empty((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = 0.01 * base + 0.02 * r - 0.013 * c
    return out.astype(np.float32)


def build_tensors() -> list[np.ndarray]:
    """Every doll-house parameter tensor, in OUR convention, in walk order.

    Weights are [out, in] (Linear), embeddings are [rows, C], biases and
    LayerNorm params are [1, out] rows — exactly the shapes the loader builds and
    the file stores. Returned as float32 (the on-disk precision).
    """
    tensors: list[np.ndarray] = []
    tensors.append(sentinel(V, C, BASE_WTE))  # wte   [V, C]
    tensors.append(sentinel(T, C, BASE_WPE))  # wpe   [T, C]
    # Block 0.
    tensors.append(sentinel(1, C, BASE_LN1_W))  # ln1.w  [1, C]
    tensors.append(sentinel(1, C, BASE_LN1_B))  # ln1.b  [1, C]
    tensors.append(sentinel(3 * C, C, BASE_QKV_W))  # qkv.w  [3C, C]
    tensors.append(sentinel(1, 3 * C, BASE_QKV_B))  # qkv.b  [1, 3C]
    tensors.append(sentinel(C, C, BASE_PROJ_W))  # proj.w [C, C]  SQUARE
    tensors.append(sentinel(1, C, BASE_PROJ_B))  # proj.b [1, C]
    tensors.append(sentinel(1, C, BASE_LN2_W))  # ln2.w  [1, C]
    tensors.append(sentinel(1, C, BASE_LN2_B))  # ln2.b  [1, C]
    tensors.append(sentinel(D_HIDDEN, C, BASE_UP_W))  # up.w   [4C, C]
    tensors.append(sentinel(1, D_HIDDEN, BASE_UP_B))  # up.b   [1, 4C]
    tensors.append(sentinel(C, D_HIDDEN, BASE_DOWN_W))  # down.w [C, 4C]
    tensors.append(sentinel(1, C, BASE_DOWN_B))  # down.b [1, C]
    tensors.append(sentinel(1, C, BASE_LNF_W))  # ln_f.w [1, C]
    tensors.append(sentinel(1, C, BASE_LNF_B))  # ln_f.b [1, C]
    return tensors


def param_count() -> int:
    """The exact GPT-2-layout parameter total for the doll-house dims.

    Mirrors GPTConfig.parameter_count(): embeddings + L*(12C^2 + 13C) + 2C.
    """
    embeddings = V * C + T * C
    per_block = 12 * C * C + 13 * C
    final_norm = 2 * C
    return embeddings + L * per_block + final_norm


def _payload_bytes(tensors: list[np.ndarray]) -> bytes:
    """The full little-endian float32 payload for `tensors`, walk order, row-major."""
    chunks: list[bytes] = []
    for tensor in tensors:
        for value in tensor.astype(np.float32).ravel():
            chunks.append(struct.pack("<f", float(value)))
    return b"".join(chunks)


def _write(path: str, header: str, payload: bytes) -> None:
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(payload)


def valid_header() -> str:
    return f"{MAGIC} {V} {T} {C} {L} {H} {param_count()}\n"


def write_fixture(path: str) -> None:
    """Write the doll-house GPT2W v1 file to `path` (header line + f32 payload)."""
    _write(path, valid_header(), _payload_bytes(build_tensors()))


# --- corrupted variants for the loader's named-error tests ---------------------
# All binary writing stays here (struct.pack) so the Mojo test just picks a writer
# and asserts that load_gpt2 raises.


def write_bad_magic(path: str) -> None:
    """A file whose family token is wrong -> load_gpt2 raises 'bad magic'."""
    header = f"XXXXX v1 {V} {T} {C} {L} {H} {param_count()}\n"
    _write(path, header, _payload_bytes(build_tensors()))


def write_wrong_version(path: str) -> None:
    """A file tagged v9 -> load_gpt2 raises 'unsupported version'."""
    header = f"{MAGIC[:6]}v9 {V} {T} {C} {L} {H} {param_count()}\n"
    _write(path, header, _payload_bytes(build_tensors()))


def write_wrong_dims(path: str) -> None:
    """A header with d_model not divisible by n_heads (C=8, H=3) -> the loader's
    cfg.validate() raises a named dim error. The payload is irrelevant (validation
    fails before it is read), so a minimal one is written."""
    header = f"{MAGIC} {V} {T} {C} {L} 3 0\n"
    _write(path, header, b"")


def write_truncated(path: str) -> None:
    """A valid header but the payload is 4 bytes short -> 'truncated payload'."""
    payload = _payload_bytes(build_tensors())
    _write(path, valid_header(), payload[:-4])


def write_trailing(path: str) -> None:
    """A valid header and payload plus 4 extra bytes -> 'trailing bytes'."""
    payload = _payload_bytes(build_tensors())
    _write(path, valid_header(), payload + struct.pack("<f", 1.0))


def write_no_newline(path: str) -> None:
    """A file with no newline at all -> 'not a GPT2W file (no header line)'."""
    with open(path, "wb") as f:
        f.write(f"{MAGIC} {V} {T} {C} {L} {H} {param_count()}".encode("ascii"))


def write_wrong_token_count(path: str) -> None:
    """A header missing the count field (7 tokens, not 8) -> 'malformed header'."""
    header = f"{MAGIC} {V} {T} {C} {L} {H}\n"  # dropped the count
    _write(path, header, _payload_bytes(build_tensors()))


def write_count_mismatch(path: str) -> None:
    """A header whose declared count disagrees with the dims -> count mismatch.
    Validation fails before the payload is read, so a minimal one is written."""
    header = f"{MAGIC} {V} {T} {C} {L} {H} {param_count() + 1}\n"
    _write(path, header, b"")


# u64 bit pattern of float32(0.1) widened to float64, for the exact-widening pin.
WIDEN_PROBE_VALUE = 0.1
WIDEN_PROBE_BITS = struct.unpack(
    "<Q", struct.pack("<d", np.float64(np.float32(0.1)))
)[0]


def write_widen_probe(path: str) -> None:
    """A valid doll-house file whose wte[0,0] is exactly float32(0.1). The Mojo
    test loads it and pins that wte[0,0]'s float64 bit pattern is WIDEN_PROBE_BITS
    — proving the f32 -> f64 read is an EXACT widening, not a re-parse."""
    tensors = build_tensors()
    tensors[0] = tensors[0].copy()
    tensors[0][0, 0] = np.float32(WIDEN_PROBE_VALUE)
    _write(path, valid_header(), _payload_bytes(tensors))


# --- reference forward (float64, reads the widened f32 sentinels) --------------


def widen(tensor: np.ndarray) -> np.ndarray:
    """float32 sentinel -> float64, exactly as the Mojo loader widens on read."""
    return tensor.astype(np.float32).astype(np.float64)


def gelu(x: np.ndarray) -> np.ndarray:
    inner = SQRT_2_OVER_PI * (x + GELU_CUBIC * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)  # biased (ddof=0), GPT-2's choice
    norm = (x - mean) / np.sqrt(var + LN_EPS)
    return norm * weight + bias


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """[out, in] weight convention: y = x @ W^T + b (our Linear)."""
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


def reference_logits(ids: list[int]) -> np.ndarray:
    """Doll-house logits [T, V] for `ids` via the float64 reference forward.

    Reads the same widened-f32 sentinel weights the Mojo loader builds, runs one
    pre-LN block (fused-QKV self-attention + MLP) and the tied head, so the
    result is an independent oracle for the loaded model's forward.
    """
    tensors = [widen(t) for t in build_tensors()]
    (
        wte,
        wpe,
        ln1_w,
        ln1_b,
        qkv_w,
        qkv_b,
        proj_w,
        proj_b,
        ln2_w,
        ln2_b,
        up_w,
        up_b,
        down_w,
        down_b,
        lnf_w,
        lnf_b,
    ) = tensors
    t = len(ids)
    x = wte[ids] + wpe[list(range(t))]  # [T, C] token + positional
    mask = causal_mask(t)

    # Pre-LN self-attention sublayer.
    ln1 = layernorm(x, ln1_w[0], ln1_b[0])
    qkv = linear(ln1, qkv_w, qkv_b[0])  # [T, 3C]
    q_all, k_all, v_all = qkv[:, :C], qkv[:, C : 2 * C], qkv[:, 2 * C :]
    d = C // H
    heads = []
    for h in range(H):
        lo, hi = h * d, (h + 1) * d
        q, k, vv = q_all[:, lo:hi], k_all[:, lo:hi], v_all[:, lo:hi]
        scores = q @ k.T / math.sqrt(float(d)) + mask
        heads.append(softmax_rows(scores) @ vv)
    attn = linear(np.concatenate(heads, axis=1), proj_w, proj_b[0])
    a = x + attn

    # Pre-LN MLP sublayer.
    ln2 = layernorm(a, ln2_w[0], ln2_b[0])
    mlp = linear(gelu(linear(ln2, up_w, up_b[0])), down_w, down_b[0])
    out = a + mlp

    h_final = layernorm(out, lnf_w[0], lnf_b[0])
    return h_final @ wte.T  # tied head, no bias -> [T, V]


# Fixed id sequence the Mojo end-to-end parity test uses (T=5, ids in [0, V)).
FIXTURE_IDS = [1, 3, 4, 0, 2]


def dump(label: str, arr: np.ndarray) -> None:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    print(f"{label}:  # shape {tuple(np.asarray(arr).shape)}")
    for v in flat:
        print(f"    {v!r}")


def main() -> None:
    np.set_printoptions(precision=17)
    tensors = build_tensors()
    wide = [widen(t) for t in tensors]
    print(f"# GPT2W v1 doll-house: V={V} T={T} C={C} L={L} H={H}")
    print(f"# param_count = {param_count()}")
    print()

    # Transpose / walk pins: individual widened-f32 slot values the Mojo test
    # freezes. proj is the SQUARE kernel — proj.w[0,1] != proj.w[1,0] proves the
    # loader did not transpose it.
    proj_w = wide[6]
    qkv_w = wide[4]
    print("# proj.w is [C,C] SQUARE; asymmetric so a transpose is visible:")
    print(f"#   proj_w[0,1] = {proj_w[0, 1]!r}")
    print(f"#   proj_w[1,0] = {proj_w[1, 0]!r}")
    print(f"#   qkv_w[0,0]  = {qkv_w[0, 0]!r}")
    print(f"#   qkv_w[23,7] = {qkv_w[23, 7]!r}")
    print(f"#   wte[10,7]   = {wide[0][10, 7]!r}")
    print(f"#   ln_f.w[0,0] = {wide[14][0, 0]!r}  ln_f.b[0,0] = {wide[15][0, 0]!r}")
    print()

    # A single widening pin: 0.1f32 -> its exact float64 image.
    x = np.float64(np.float32(0.1))
    print(f"# widen(0.1f32) = {x!r}  (u64 bits {struct.unpack('<Q', struct.pack('<d', x))[0]})")
    print()

    print(f"# reference logits for ids={FIXTURE_IDS}")
    dump("gpt2w_logits", reference_logits(FIXTURE_IDS))


if __name__ == "__main__":
    main()
