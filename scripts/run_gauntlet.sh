#!/usr/bin/env bash
# Run the Tier 2 gauntlet with an up-front provenance check.
#
# The goldens are frozen against ONE checkpoint, pinned by the sha256 in their
# header. Enforcing that pin mechanically — before the ~30 s run — turns "the
# committed .bin is not the one these goldens were generated from" into a clear
# provenance error instead of a confusing probe mismatch 30 seconds in. The
# numeric checks would eventually catch a wrong .bin too, but not by name.
#
# If the .bin is absent we skip the hash check and let the harness raise its
# canonical converter-pointing error (no duplicated message here).
#
# Usage:  pixi run gauntlet
#         bash scripts/run_gauntlet.sh
set -euo pipefail

cd "$(dirname "$0")/.."

BIN="checkpoints/gpt2-124m.bin"
GOLDENS="data/gauntlet/goldens.txt"

if [[ -f "$BIN" ]]; then
    want="$(grep -oE 'sha256=[0-9a-f]{64}' "$GOLDENS" | head -1 | cut -d= -f2)"
    if [[ -z "$want" ]]; then
        echo "run_gauntlet: no sha256 header in $GOLDENS — regenerate it via" \
            "scripts/gpt2_gauntlet_reference.py" >&2
        exit 1
    fi
    got="$(sha256sum "$BIN" | cut -d' ' -f1)"
    if [[ "$want" != "$got" ]]; then
        echo "run_gauntlet: $BIN does not match the goldens' pinned weights." >&2
        echo "  goldens sha256: $want" >&2
        echo "  $BIN sha256:    $got" >&2
        echo "  Either the .bin was regenerated (re-run" \
            "scripts/gpt2_gauntlet_reference.py and document the new hash in" \
            "notes) or the wrong checkpoint is present." >&2
        exit 1
    fi
    echo "==> provenance OK: $BIN matches goldens sha256 ${want:0:12}..."
fi

# Precompile the package (like scripts/test_all.sh), then run the harness against
# the binary package at -I build.
echo "==> precompiling src/llm -> build/llm.mojopkg"
pixi run mojo precompile src/llm -o build/llm.mojopkg
pixi run mojo run -I build examples/gpt2_gauntlet.mojo
