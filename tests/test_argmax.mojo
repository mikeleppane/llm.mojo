# Tests for argmax, including the deliberate first-wins tie rule.
#
# The tie behavior is a design decision (greedy decoding must be deterministic),
# so it gets a test that pins it: with two equal maxima, the first index wins.

from std.testing import assert_equal, TestSuite

from llm.tensor.ops import argmax


def test_argmax_basic() raises:
    assert_equal(argmax([0.1, 0.9, 0.3]), 1)


def test_argmax_tie_prefers_first() raises:
    assert_equal(argmax([0.5, 2.0, 2.0, 1.0]), 1)


def test_argmax_single_element() raises:
    assert_equal(argmax([42.0]), 0)


def test_argmax_negative_values() raises:
    assert_equal(argmax([-3.0, -1.0, -2.0]), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
