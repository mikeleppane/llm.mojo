"""Smoke test: proves the toolchain works and TestSuite discovery runs.

The first test in the pyramid (see AGENTS.md, "Testing"). It imports nothing from
`src/llm`, so it stays green on a fresh checkout and answers one question: does
`mojo run` build and execute a TestSuite here?
"""

from std.testing import (
    assert_equal,
    assert_true,
    assert_almost_equal,
    TestSuite,
)


def test_toolchain_runs() raises:
    """Reaching here proves the Mojo toolchain compiled and ran the file."""
    assert_equal(2 + 2, 4)


def test_bool_assert() raises:
    """`assert_true` works."""
    assert_true(True)


def test_float_tolerance() raises:
    """Floats are compared with a tolerance, never ==."""
    assert_almost_equal(0.1 + 0.2, 0.3, atol=1e-9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
