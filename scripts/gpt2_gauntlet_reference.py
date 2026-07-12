#!/usr/bin/env python3
"""Offline golden generator for the 124M gauntlet — NumPy + stdlib only.

Reads the committed prompt set (data/gauntlet/prompts.txt), encodes each prompt
with the SAME tokenizer oracle the parity tests trust
(tests/oracles/gpt2_reference_encoder.py — OpenAI's original BPE), runs the SAME
float64 forward the single-prompt parity gate uses
(scripts/gpt2_reference_logits.py, reading OUR checkpoints/gpt2-124m.bin so both
sides consume identical bytes), and writes the frozen reference goldens the Mojo
harness (examples/gpt2_gauntlet.mojo) checks against.

Nothing here reimplements BPE or the forward pass — both are imported, so a
divergence indicts the model, not a second copy of the math. No torch, no
transformers (standing repo rule).

Per prompt, the goldens record:
  * tokens: the encoded ids (the Mojo GPT2Tokenizer must reproduce these EXACTLY);
  * argmax + top5: the final row's argmax id and top-5 ids (discrete — no
    tolerance on the Mojo side);
  * probe: the final-row logit at a fixed spread of vocab indices plus the argmax
    (checked at 1e-6, the float64-vs-float64 bar);
  * nll: the mean next-token cross-entropy of the model on the prompt's own tokens
    — the whole-model-in-one-scalar drift detector (checked at 1e-6). Undefined
    for a single-token prompt, recorded as "none".

The output header pins the .bin's sha256 so every goldens.txt is tied to the exact
weights that produced it. GOLDEN LIFECYCLE: a red gauntlet after a code change
indicts THE CHANGE. Regenerating this file is legitimate only when the oracle side
changed (a new .bin — visible in the header hash — or a converter fix) or a
near-tie logit delta at ~1e-13 scale is documented; "the number looks close
enough" is never evidence, and these goldens are regenerated ONLY by this script,
never by hand.

Usage (run once by hand; the output is committed and frozen):
    pixi run python scripts/gpt2_gauntlet_reference.py
    pixi run python scripts/gpt2_gauntlet_reference.py checkpoints/gpt2-124m.bin \\
        --prompts data/gauntlet/prompts.txt --out data/gauntlet/goldens.txt
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
# Reuse the proven oracles rather than reimplementing them.
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "tests" / "oracles"))
from gpt2_reference_encoder import get_encoder  # noqa: E402
from gpt2_reference_logits import forward, read_gpt2w  # noqa: E402

# Fixed vocab indices probed on every prompt's final row (all < V = 50257): the
# two lowest ids, a spread across the vocab, and the end-of-text id. The per-prompt
# argmax is appended so its exact value is always frozen too.
PROBE_INDICES = [0, 1, 50, 100, 1000, 10000, 40000, 50256]

SEPARATOR_PREFIX = "=== id: "


def parse_prompts(text: str) -> list[tuple[str, str]]:
    """Parse prompts.txt into (id, prompt_text) pairs.

    The canonical parser, mirrored byte-for-byte in the Mojo harness: split on
    "\\n" ONLY (never a unicode-aware splitlines), drop the single file-terminating
    empty line, then walk records. A line starting with "=== id: " opens a record
    ("=== id: <name> ===", rationale after the closing "===" ignored); every line
    up to the next separator is the prompt, joined by "\\n".
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]  # drop the mandatory file-terminating newline

    records: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.startswith(SEPARATOR_PREFIX):
            rest = line[len(SEPARATOR_PREFIX) :]
            end = rest.find(" ===")
            if end < 0:
                raise ValueError(f"malformed separator (no closing ' ==='): {line!r}")
            name = rest[:end]
            records.append((name, []))
        else:
            if not records:
                continue  # preamble / header comments before the first record
            records[-1][1].append(line)
    return [(name, "\n".join(body)) for name, body in records]


def logsumexp(row: np.ndarray) -> float:
    m = float(row.max())
    return m + math.log(float(np.exp(row - m).sum()))


def mean_nll(logits: np.ndarray, ids: list[int]) -> float | None:
    """Mean next-token cross-entropy of the prompt's own tokens.

    Teacher-forced: position i predicts ids[i+1]. By causality logits[i] equals the
    forward over ids[:i+1], so this equals GPT.loss(ids[:-1], ids[1:]) — what the
    Mojo harness computes. None when there is no next-token pair (T < 2).
    """
    t = len(ids)
    if t < 2:
        return None
    total = 0.0
    for i in range(t - 1):
        total += logsumexp(logits[i]) - float(logits[i][ids[i + 1]])
    return total / (t - 1)


def fmt(x: float) -> str:
    """Round-trippable float64 text. Guards against exponent notation, which the
    deliberately-dumb Mojo parser does not accept (all logits/NLL are O(1)-O(1e2),
    so plain decimal always suffices)."""
    s = repr(x)
    if "e" in s or "E" in s:
        raise ValueError(f"value {s} needs exponent notation the Mojo parser rejects")
    return s


def build_goldens(bin_path: Path, prompts_path: Path) -> str:
    enc = get_encoder()
    dims, tensors = read_gpt2w(bin_path)
    vocab = dims["V"]
    sha = hashlib.sha256(bin_path.read_bytes()).hexdigest()

    out: list[str] = [
        f"# generated by scripts/gpt2_gauntlet_reference.py from"
        f" {bin_path.name} sha256={sha} — do not hand-edit",
        "# One block per prompt id (order matches data/gauntlet/prompts.txt).",
        "#   tokens: encoded ids   argmax/top5: final-row ids (exact match)",
        "#   probe:  <idx>:<final-row logit> pairs (1e-6)   nll: mean next-token"
        " cross-entropy (1e-6), or 'none' for a single-token prompt",
    ]

    for name, text in parse_prompts(prompts_path.read_text(encoding="utf-8")):
        ids = enc.encode(text)
        if not ids:
            raise ValueError(f"prompt {name!r} encoded to zero tokens")
        for tid in ids:
            if not (0 <= tid < vocab):
                raise ValueError(f"prompt {name!r}: token id {tid} out of vocab")
        logits = forward(dims, tensors, ids)  # [T, V], float64
        last = logits[-1]

        argmax = int(np.argmax(last))
        # Highest logit first, LOWEST id first on a tie — a stable sort of -last
        # keeps ascending-index order among equal values, matching the Mojo
        # harness's lowest-index-first _top_k_ids. (np.argmax already returns the
        # first max, so argmax uses the same tie rule.) Real logits do not tie
        # exactly, so this only fixes the contract, not any current value.
        top5 = [int(j) for j in np.argsort(-last, kind="stable")[:5]]
        probes = list(PROBE_INDICES)
        if argmax not in probes:
            probes.append(argmax)
        probe_str = " ".join(f"{j}:{fmt(float(last[j]))}" for j in probes)
        nll = mean_nll(logits, ids)

        out.append(f"=== id: {name} ===")
        out.append("tokens: " + " ".join(str(t) for t in ids))
        out.append(f"argmax: {argmax}")
        out.append("top5: " + " ".join(str(t) for t in top5))
        out.append("probe: " + probe_str)
        out.append("nll: " + ("none" if nll is None else fmt(nll)))

    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bin",
        type=Path,
        nargs="?",
        default=_ROOT / "checkpoints" / "gpt2-124m.bin",
        help="path to the converted GPT2W v1 weights",
    )
    parser.add_argument("--prompts", type=Path, default=_ROOT / "data" / "gauntlet" / "prompts.txt")
    parser.add_argument("--out", type=Path, default=_ROOT / "data" / "gauntlet" / "goldens.txt")
    parser.add_argument(
        "--skip-self-check",
        action="store_true",
        help="skip the determinism guard (a second pass compared byte-for-byte)",
    )
    args = parser.parse_args(argv)

    if not args.bin.exists():
        print(
            f"missing '{args.bin}' — download the HF gpt2 model.safetensors and run"
            " scripts/convert_gpt2_weights.py to produce it. No random-weight"
            " fallback: the goldens need the real weights.",
            file=sys.stderr,
        )
        return 1

    content = build_goldens(args.bin, args.prompts)

    if not args.skip_self_check:
        again = build_goldens(args.bin, args.prompts)
        if again != content:
            print("determinism self-check FAILED: two passes disagree", file=sys.stderr)
            return 1
        print("determinism self-check OK (two passes byte-identical)", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(f"wrote {args.out} ({len(content)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
