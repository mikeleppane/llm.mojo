#!/usr/bin/env bash
# Run every Mojo test in tests/ with the src package on the include path.
#
# Ordering: the smoke test runs first so a broken toolchain fails fast before we
# spend time compiling everything else. The remaining tests run in a glob loop so
# new tests are picked up automatically — no hand-maintained list to drift.
#
# Usage:  pixi run test      (preferred, see pixi.toml)
#         bash scripts/test_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# Fast-fail smoke test first.
if [[ -f tests/test_smoke.mojo ]]; then
    echo "==> tests/test_smoke.mojo"
    pixi run mojo run -I src tests/test_smoke.mojo
fi

# Every other test file, sorted for a stable run order.
shopt -s nullglob
failed=0
for test_file in $(printf '%s\n' tests/test_*.mojo | sort); do
    [[ "$test_file" == "tests/test_smoke.mojo" ]] && continue
    echo "==> $test_file"
    if ! pixi run mojo run -I src "$test_file"; then
        echo "FAILED: $test_file" >&2
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "Some tests failed." >&2
    exit 1
fi

echo "All tests passed."
