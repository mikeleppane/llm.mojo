#!/usr/bin/env python3
"""Fetch and verify the GPT-2 reference tokenizer files.

These two files are committed to the repository under ``data/gpt2/`` so that
tests and CI run offline with no network fetch. This script exists for
*provenance*: it records exactly where the files came from and pins their
SHA-256 checksums, so anyone can reproduce ``data/gpt2/`` from scratch and prove
the committed bytes were not tampered with.

The source is OpenAI's original GPT-2 release (124M model) on Azure blob
storage. OpenAI names the files ``encoder.json`` (token string -> id) and
``vocab.bpe`` (ordered merge rules). The wider ecosystem (Hugging Face) calls
the same two files ``vocab.json`` and ``merges.txt``; we save them under the
latter, more common names. The bytes are identical to OpenAI's originals.

Usage::

    pixi run python scripts/download_gpt2_files.py           # download + verify
    pixi run python scripts/download_gpt2_files.py --check   # verify existing only

No third-party dependencies: standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Canonical source: OpenAI's original GPT-2 (124M) release on Azure blob storage.
BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M"

# Each entry: local filename -> (source filename at BASE_URL, expected SHA-256).
# Checksums were computed from the canonical OpenAI files; if a download does not
# match, the script refuses to write it rather than committing unknown bytes.
FILES = {
    "vocab.json": (
        "encoder.json",
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    ),
    "merges.txt": (
        "vocab.bpe",
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    ),
}

DEST_DIR = Path(__file__).resolve().parent.parent / "data" / "gpt2"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_existing() -> int:
    """Check committed files against their pinned checksums. Returns exit code."""
    failed = False
    for local_name, (_source, expected) in FILES.items():
        path = DEST_DIR / local_name
        if not path.exists():
            print(f"MISSING: {path}", file=sys.stderr)
            failed = True
            continue
        got = sha256(path.read_bytes())
        if got != expected:
            print(f"CHECKSUM MISMATCH: {path}", file=sys.stderr)
            print(f"  expected {expected}", file=sys.stderr)
            print(f"  got      {got}", file=sys.stderr)
            failed = True
        else:
            print(f"OK: {local_name} ({got})")
    return 1 if failed else 0


def download() -> int:
    """Download each file, verify its checksum, then write it. Returns exit code."""
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for local_name, (source_name, expected) in FILES.items():
        url = f"{BASE_URL}/{source_name}"
        print(f"downloading {url}")
        with urllib.request.urlopen(url) as response:  # noqa: S310 (trusted host)
            data = response.read()
        got = sha256(data)
        if got != expected:
            print(f"CHECKSUM MISMATCH for {url}", file=sys.stderr)
            print(f"  expected {expected}", file=sys.stderr)
            print(f"  got      {got}", file=sys.stderr)
            print("refusing to write unverified bytes", file=sys.stderr)
            return 1
        dest = DEST_DIR / local_name
        dest.write_bytes(data)
        print(f"wrote {dest} ({len(data)} bytes, sha256 {got})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing committed files instead of downloading",
    )
    args = parser.parse_args()
    return verify_existing() if args.check else download()


if __name__ == "__main__":
    raise SystemExit(main())
