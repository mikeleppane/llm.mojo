#!/usr/bin/env bash
# Compile-check every file in examples/ and benchmarks/ WITHOUT running it.
#
# Examples are the guide's artifacts and benchmarks are the perf harnesses; both
# import the `llm` package but are not covered by the test suite, so they bit-rot
# silently between the parts that touch them — a renamed function or a changed
# signature only surfaces the next time someone runs one by hand. This builds each
# one against the PRECOMPILED package at -O0 (seconds per file, no weights, no
# network), so a broken example fails CI instead of being discovered later.
#
# Why build, not run: running needs the 475 MB weights (Tier 2) and real time;
# compiling only needs the types to line up, which is exactly the bit-rot we want
# to catch cheaply. Same precompile-once trick as scripts/test_all.sh — build the
# library once, then each file compiles only its own small tree against a binary.
#
# Why --emit object (compile, don't link): the full parse/typecheck/codegen is
# what catches bit-rot; the final AOT link is not. This toolchain's `mojo build`
# link step omits libm, so linking any file that pulls in exp/tanh (every model
# example) fails with an `expm1` undefined-reference that says nothing about the
# source — `mojo run` JIT-links fine, which is why the examples still RUN. Emitting
# an object file runs the whole compiler and stops before that irrelevant link.
#
# Usage:  pixi run build-examples
#         bash scripts/build_examples.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# Precompile src/llm -> build/llm.mojopkg (named llm.mojopkg so `-I build`
# resolves `from llm... import`); build/ is gitignored.
PKG_DIR="build"
mkdir -p "$PKG_DIR"
echo "==> precompiling src/llm -> $PKG_DIR/llm.mojopkg"
pixi run mojo precompile src/llm -o "$PKG_DIR/llm.mojopkg"

# Throwaway output dir for the emitted object files — we only care that they
# compile, not that they run.
OUT_DIR="$PKG_DIR/_build_check"
mkdir -p "$OUT_DIR"

# -O0 against the prebuilt package: correctness is not the point (the tests own
# that), only that every example/benchmark still compiles.
INCLUDE=(--emit object --no-optimization -I "$PKG_DIR")

shopt -s nullglob
failed=0
for src_file in $(printf '%s\n' examples/*.mojo benchmarks/*.mojo | sort); do
    out_obj="$OUT_DIR/$(basename "${src_file%.mojo}").o"
    echo "==> $src_file"
    if ! pixi run mojo build "${INCLUDE[@]}" "$src_file" -o "$out_obj"; then
        echo "FAILED to build: $src_file" >&2
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "Some examples/benchmarks failed to build." >&2
    exit 1
fi

echo "All examples and benchmarks build."
