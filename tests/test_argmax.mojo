# Tests for argmax, including the deliberate first-wins tie rule.
#
# The tie behavior is a design decision (greedy decoding must be deterministic),
# so it gets a test that pins it: with two equal maxima, the first index wins.

from std.testing import assert_equal, TestSuite

from llm.tensor.ops import argmax


def test_argmax_basic() raises:
    var logits = [0.1, 0.9, 0.3]
    assert_equal(argmax(logits), 1)


def test_argmax_tie_prefers_first() raises:
    var logits = [0.5, 2.0, 2.0, 1.0]
    assert_equal(argmax(logits), 1)


def test_argmax_single_element() raises:
    var logits = [42.0]
    assert_equal(argmax(logits), 0)


def test_argmax_negative_values() raises:
    var logits = [-3.0, -1.0, -2.0]
    assert_equal(argmax(logits), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
