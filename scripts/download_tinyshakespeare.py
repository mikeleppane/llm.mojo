#!/usr/bin/env python3
"""Fetch and verify the tiny Shakespeare corpus.

The corpus is committed to the repository under ``data/tinyshakespeare/`` so
that tests and CI run offline with no network fetch. This script exists for
*provenance*: it records exactly where the file came from and pins its SHA-256
checksum, so anyone can reproduce ``data/tinyshakespeare/input.txt`` from
scratch and prove the committed bytes were not tampered with.

The source is Andrej Karpathy's ``char-rnn`` repository — a ~1.1 MB
concatenation of Shakespeare, the same file nanoGPT and essentially every
reference implementation trains on. It is a single plain-text document, which is
why the dataset layer treats it as one contiguous corpus (train = prefix, val =
suffix) rather than a shuffled collection of documents.

Usage::

    pixi run python scripts/download_tinyshakespeare.py           # download + verify
    pixi run python scripts/download_tinyshakespeare.py --check   # verify existing only

No third-party dependencies: standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Canonical source: Karpathy's char-rnn raw file on GitHub.
URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

# Expected SHA-256 of the canonical file. If a download does not match, the
# script refuses to write it rather than committing unknown bytes.
EXPECTED_SHA256 = (
    "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
)

DEST = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "tinyshakespeare"
    / "input.txt"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_existing() -> int:
    """Check the committed file against its pinned checksum. Returns exit code."""
    if not DEST.exists():
        print(f"MISSING: {DEST}", file=sys.stderr)
        return 1
    got = sha256(DEST.read_bytes())
    if got != EXPECTED_SHA256:
        print(f"CHECKSUM MISMATCH: {DEST}", file=sys.stderr)
        print(f"  expected {EXPECTED_SHA256}", file=sys.stderr)
        print(f"  got      {got}", file=sys.stderr)
        return 1
    print(f"OK: {DEST.name} ({got})")
    return 0


def download() -> int:
    """Download the file, verify its checksum, then write it. Returns exit code."""
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL) as response:  # noqa: S310 (trusted host)
        data = response.read()
    got = sha256(data)
    if got != EXPECTED_SHA256:
        print(f"CHECKSUM MISMATCH for {URL}", file=sys.stderr)
        print(f"  expected {EXPECTED_SHA256}", file=sys.stderr)
        print(f"  got      {got}", file=sys.stderr)
        print("refusing to write unverified bytes", file=sys.stderr)
        return 1
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_bytes(data)
    print(f"wrote {DEST} ({len(data)} bytes, sha256 {got})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing committed file instead of downloading",
    )
    args = parser.parse_args()
    return verify_existing() if args.check else download()


if __name__ == "__main__":
    raise SystemExit(main())
