# Smoke test: proves the toolchain works and TestSuite discovery runs.
#
# This is the first test in the pyramid (see AGENTS.md, "Testing").
# It deliberately imports nothing from `src/llm` so it stays green on a fresh
# checkout and answers exactly one question: does `mojo run` build and execute a
# TestSuite in this environment? Real correctness tests live in their own files.

from std.testing import (
    assert_equal,
    assert_true,
    assert_almost_equal,
    TestSuite,
)


def test_toolchain_runs() raises:
    # If this file compiled and reached here, the Mojo toolchain works.
    assert_equal(2 + 2, 4)


def test_bool_assert() raises:
    assert_true(True)


def test_float_tolerance() raises:
    # Never compare floats with ==; use a tolerance. See the testing skill.
    assert_almost_equal(0.1 + 0.2, 0.3, atol=1e-9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
