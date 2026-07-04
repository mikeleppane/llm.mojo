#!/usr/bin/env bash
# Run every Mojo test in tests/ against a PRECOMPILED `llm` package.
#
# Why precompile: `mojo run -I src tests/X.mojo` recompiles AND re-optimizes the
# entire `llm` source tree, inlined into each test binary, at full `-O` every
# time. For test files that pull the whole model into one function (the training
# loop, the end-to-end finite-difference checks) LLVM grinds on those giant
# monomorphized functions for minutes — per file. Precompiling `llm` once into a
# binary package (~1s) lets each test build against that binary instead, so a
# test compiles only its own small file and the library's optimizer passes run
# once, not per test. This turns a tens-of-minutes suite into a couple of
# minutes with identical results. (The mojo docs note a precompiled package is
# "faster to build with".)
#
# Ordering: the smoke test runs first so a broken toolchain (or a broken package
# build) fails fast. The remaining tests run in a glob loop so new tests are
# picked up automatically — no hand-maintained list to drift.
#
# Usage:  pixi run test      (preferred, see pixi.toml)
#         bash scripts/test_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# Precompile src/llm -> build/llm.mojopkg. The build/ dir is gitignored; the
# package must be named llm.mojopkg so `-I build` resolves `from llm... import`.
PKG_DIR="build"
mkdir -p "$PKG_DIR"
echo "==> precompiling src/llm -> $PKG_DIR/llm.mojopkg"
pixi run mojo precompile src/llm -o "$PKG_DIR/llm.mojopkg"

# Tests build against the prebuilt package, not the source tree, and at -O0.
# Correctness tests do not need optimized codegen, and the heavy math already ran
# through the optimizer once when the package was built — so the thin per-test
# glue compiles at -O0 (seconds) instead of stalling the LLVM -O pass for minutes
# on files heavy in list literals and the TestSuite discovery metaprogramming.
INCLUDE=(--no-optimization -I "$PKG_DIR")

# Fast-fail smoke test first.
if [[ -f tests/test_smoke.mojo ]]; then
    echo "==> tests/test_smoke.mojo"
    pixi run mojo run "${INCLUDE[@]}" tests/test_smoke.mojo
fi

# Every other test file, sorted for a stable run order.
shopt -s nullglob
failed=0
for test_file in $(printf '%s\n' tests/test_*.mojo | sort); do
    [[ "$test_file" == "tests/test_smoke.mojo" ]] && continue
    echo "==> $test_file"
    if ! pixi run mojo run "${INCLUDE[@]}" "$test_file"; then
        echo "FAILED: $test_file" >&2
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "Some tests failed." >&2
    exit 1
fi

echo "All tests passed."
