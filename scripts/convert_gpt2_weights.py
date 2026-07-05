#!/usr/bin/env python3
"""Convert HuggingFace GPT-2 `model.safetensors` to our `GPT2W v1` weight file.

Offline, manual, run BY HAND once — NumPy + Python stdlib only. There is NO torch
and NO transformers import anywhere in this file (or allowed in it, ever): the
safetensors container is a JSON header plus a raw little-endian tensor buffer,
which we parse ourselves in a few dozen commented lines. This is the ONE place
that knows GPT-2's tensor names and its Conv1D layout; the Mojo loader
(src/llm/transformer/gpt2_weights.mojo) is deliberately dumb and knows only the
walk order and the shapes.

Where the input comes from
--------------------------
Download the HuggingFace `openai-community/gpt2` repo's `model.safetensors`
(~475 MB) once; record its sha256 in notes/part-16-notes.md. It is never
committed (checkpoints/ and *.safetensors are gitignored). Example:

    curl -L -o gpt2-model.safetensors \
      https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
    python scripts/convert_gpt2_weights.py gpt2-model.safetensors

Output defaults to `checkpoints/gpt2-124m.bin` (also gitignored).

The GPT2W v1 format (the single spec this converter, the fixture oracle
tests/oracles/gpt2_weights_reference.py, and the Mojo loader all agree on):

    line 0 (ASCII, newline-terminated):
        GPT2W v1 <V> <T> <C> <L> <H> <param_count>
    then the raw little-endian float32 payload, every parameter tensor back to
    back in THE MODEL'S WALK ORDER, row-major within each tensor:
        wte [V,C], wpe [T,C],
        per block (x L): ln1.w [1,C], ln1.b [1,C], qkv.w [3C,C], qkv.b [1,3C],
                         proj.w [C,C], proj.b [1,C], ln2.w [1,C], ln2.b [1,C],
                         up.w [4C,C], up.b [1,4C], down.w [C,4C], down.b [1,C],
        ln_f.w [1,C], ln_f.b [1,C].

The layout fixes, exhaustively (every silent-garbage bug lives in this list)
----------------------------------------------------------------------------
Our Linear is y = x @ W^T + b with W stored [out, in]; HF's Conv1D computes
x @ W with W stored [in, out]. Same math iff we TRANSPOSE every Conv1D kernel:

    c_attn.weight   [C, 3C]  -> qkv.weight   [3C, C]
    attn.c_proj.w   [C, C]   -> proj.weight  [C, C]     (SQUARE — a wrong
                                                          transpose still
                                                          shape-checks; the
                                                          sentinel fixture and
                                                          the 124M parity gate
                                                          are what catch it)
    mlp.c_fc.weight [C, 4C]  -> up.weight    [4C, C]
    mlp.c_proj.w    [4C, C]  -> down.weight  [C, 4C]

Do NOT transpose the embeddings (wte [V,C], wpe [T,C] are row-gather tables,
same as ours) or any LayerNorm weight/bias. Reshape every 1-D vector (all biases,
all LayerNorm params) to a [1, out] row. SKIP the buffers attn.bias (the causal
mask) and attn.masked_bias — they are not parameters; we build our own mask. The
head is TIED, so wte appears exactly once and there is no lm_head weight: if the
file carries lm_head.weight, we assert it equals wte and drop it. HF has shipped
GPT-2 both with and without a `transformer.` name prefix, so we strip that prefix
once up front. Column order needs NO fix: HF's c_attn packs Q|K|V in thirds and
splits heads contiguously — exactly our attention convention; the 124M parity
gate is the final arbiter.

Structural defense against ingesting the wrong tensor: this converter PULLS each
tensor BY OUR parameter inventory (raising on a missing name), it never iterates
the HF file pushing tensors into slots. A buffer can therefore never land in a
weight slot, and a renamed/missing tensor fails loudly.

Only GPT-2 124M is converted, tested, and claimed here. The one generalization
point is `--n-heads` (12 for 124M); the walk and the format already carry the
other dims, so a larger GPT-2 would differ only in the dims read off the file and
this one argument.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = "GPT2W v1"
DEFAULT_OUTPUT = "checkpoints/gpt2-124m.bin"


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Parse a .safetensors file into {name: float32 ndarray}, ourselves.

    Layout: 8-byte little-endian uint64 header length N, then N bytes of JSON
    mapping each tensor name to {dtype, shape, data_offsets: [begin, end]}, then
    the raw tensor buffer. Each tensor's bytes are buffer[begin:end], row-major,
    little-endian. We only support F32 (GPT-2's dtype); anything else raises.
    """
    raw = path.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    buffer = raw[8 + header_len :]
    tensors: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] != "F32":
            raise ValueError(
                f"{name}: unsupported dtype {meta['dtype']!r} (only F32)"
            )
        begin, end = meta["data_offsets"]
        flat = np.frombuffer(buffer[begin:end], dtype="<f4")
        tensors[name] = flat.reshape(meta["shape"])
    return tensors


def strip_prefix(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop a leading `transformer.` from every key (HF has shipped both)."""
    out: dict[str, np.ndarray] = {}
    for name, arr in tensors.items():
        out[name.removeprefix("transformer.")] = arr
    return out


def as_row(vec: np.ndarray) -> np.ndarray:
    """A 1-D bias / LayerNorm vector -> a [1, out] row (our Tensor2D convention)."""
    if vec.ndim != 1:
        raise ValueError(f"expected a 1-D vector to reshape to a row, got {vec.shape}")
    return vec.reshape(1, vec.shape[0])


def transpose_kernel(w: np.ndarray) -> np.ndarray:
    """An HF Conv1D kernel [in, out] -> our Linear weight [out, in]."""
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D Conv1D kernel to transpose, got {w.shape}")
    return np.ascontiguousarray(w.T)


class Puller:
    """Pull tensors out of the HF dict BY NAME, raising on a missing key.

    Pulling (never pushing) is the structural guarantee that no buffer or stray
    tensor can slip into a parameter slot: every slot names exactly the HF tensor
    it wants, and a rename/miss fails loudly instead of silently.
    """

    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = tensors
        self.pulled: set[str] = set()

    def get(self, name: str) -> np.ndarray:
        if name not in self.tensors:
            raise KeyError(
                f"required GPT-2 tensor {name!r} not found in the safetensors "
                f"(have {len(self.tensors)} tensors; is this a GPT-2 checkpoint?)"
            )
        self.pulled.add(name)
        return self.tensors[name]


def build_walk(
    pull: Puller, n_layers: int
) -> tuple[list[np.ndarray], int, int, int]:
    """Assemble every parameter tensor in walk order, applying the layout fixes.

    Returns (tensors, V, T, C). Each tensor is float32 in our convention:
    weights [out, in], embeddings [rows, C], biases/LN as [1, out] rows.
    """
    walk: list[np.ndarray] = []

    wte = pull.get("wte.weight")  # [V, C], no transpose (row-gather table)
    wpe = pull.get("wpe.weight")  # [T, C], no transpose
    vocab_size, c = wte.shape
    context_length = wpe.shape[0]
    walk.append(wte)
    walk.append(wpe)

    for i in range(n_layers):
        p = f"h.{i}."
        # LayerNorm 1 (NOT transposed; reshaped to rows).
        walk.append(as_row(pull.get(p + "ln_1.weight")))
        walk.append(as_row(pull.get(p + "ln_1.bias")))
        # Fused QKV: Conv1D kernel [C, 3C] -> [3C, C]; bias [3C] -> [1, 3C].
        walk.append(transpose_kernel(pull.get(p + "attn.c_attn.weight")))
        walk.append(as_row(pull.get(p + "attn.c_attn.bias")))
        # Attention output proj: SQUARE Conv1D [C, C] -> [C, C]; bias [C] -> row.
        walk.append(transpose_kernel(pull.get(p + "attn.c_proj.weight")))
        walk.append(as_row(pull.get(p + "attn.c_proj.bias")))
        # LayerNorm 2.
        walk.append(as_row(pull.get(p + "ln_2.weight")))
        walk.append(as_row(pull.get(p + "ln_2.bias")))
        # MLP up: Conv1D [C, 4C] -> [4C, C]; bias [4C] -> [1, 4C].
        walk.append(transpose_kernel(pull.get(p + "mlp.c_fc.weight")))
        walk.append(as_row(pull.get(p + "mlp.c_fc.bias")))
        # MLP down: Conv1D [4C, C] -> [C, 4C]; bias [C] -> [1, C].
        walk.append(transpose_kernel(pull.get(p + "mlp.c_proj.weight")))
        walk.append(as_row(pull.get(p + "mlp.c_proj.bias")))

    # Final LayerNorm.
    walk.append(as_row(pull.get("ln_f.weight")))
    walk.append(as_row(pull.get("ln_f.bias")))

    return walk, vocab_size, context_length, c


def parameter_count(v: int, t: int, c: int, n_layers: int) -> int:
    """The exact GPT-2-layout total, mirroring GPTConfig.parameter_count()."""
    embeddings = v * c + t * c
    per_block = 12 * c * c + 13 * c
    final_norm = 2 * c
    return embeddings + n_layers * per_block + final_norm


def count_layers(tensors: dict[str, np.ndarray]) -> int:
    """How many blocks are present (highest h.{i}. index + 1)."""
    layers = set()
    for name in tensors:
        if name.startswith("h."):
            layers.add(int(name.split(".")[1]))
    if not layers:
        raise ValueError("no h.{i}. blocks found — not a GPT-2 checkpoint?")
    if sorted(layers) != list(range(len(layers))):
        raise ValueError(f"block indices are not contiguous: {sorted(layers)}")
    return len(layers)


def write_gpt2w(
    path: Path,
    walk: list[np.ndarray],
    v: int,
    t: int,
    c: int,
    n_layers: int,
    n_heads: int,
) -> None:
    """Write the header line + float32 payload (walk order, row-major)."""
    count = parameter_count(v, t, c, n_layers)
    total = sum(int(arr.size) for arr in walk)
    if total != count:
        raise ValueError(
            f"assembled {total} floats but the dims imply {count} — walk drifted"
        )
    header = f"{MAGIC} {v} {t} {c} {n_layers} {n_heads} {count}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for arr in walk:
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("safetensors", type=Path, help="path to GPT-2 model.safetensors")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path(DEFAULT_OUTPUT),
        help=f"output GPT2W v1 path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n-heads", type=int, default=12,
        help="attention heads — 12 for GPT-2 124M (the generalization point)",
    )
    args = parser.parse_args(argv)

    tensors = strip_prefix(read_safetensors(args.safetensors))

    # Tied head paranoia: if lm_head.weight is present it must equal wte, and we
    # drop it (our model has no head Parameter). Paranoia is free at convert time.
    if "lm_head.weight" in tensors:
        if not np.array_equal(tensors["lm_head.weight"], tensors["wte.weight"]):
            raise ValueError(
                "lm_head.weight is present but does NOT equal wte.weight — the "
                "head is not tied as this architecture assumes"
            )
        del tensors["lm_head.weight"]

    n_layers = count_layers(tensors)
    pull = Puller(tensors)
    walk, v, t, c = build_walk(pull, n_layers)

    # Report anything in the file we did NOT pull, so an unexpected extra tensor
    # is visible rather than silently ignored. attn.bias / attn.masked_bias are
    # the expected buffers we deliberately skip.
    unpulled = sorted(set(tensors) - pull.pulled)
    expected_skips = {
        name
        for name in unpulled
        if name.endswith("attn.bias") or name.endswith("attn.masked_bias")
    }
    surprises = [n for n in unpulled if n not in expected_skips]
    if surprises:
        print(
            f"warning: {len(surprises)} tensor(s) in the file were not used: "
            f"{surprises[:8]}{' ...' if len(surprises) > 8 else ''}",
            file=sys.stderr,
        )

    write_gpt2w(args.output, walk, v, t, c, n_layers, args.n_heads)
    count = parameter_count(v, t, c, n_layers)
    print(
        f"wrote {args.output}: V={v} T={t} C={c} L={n_layers} H={args.n_heads} "
        f"params={count} (skipped {len(expected_skips)} mask buffers)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
