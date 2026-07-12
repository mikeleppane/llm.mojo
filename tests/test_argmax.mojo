"""Tests for argmax, including the first-wins tie rule.

Greedy decoding must be deterministic, so the tie behavior is pinned: with two
equal maxima, the first index wins.
"""

from std.testing import assert_equal, TestSuite

from llm.tensor.ops import argmax


def test_argmax_basic() raises:
    """Returns the index of the single maximum."""
    var logits = [0.1, 0.9, 0.3]
    assert_equal(argmax(logits), 1)


def test_argmax_tie_prefers_first() raises:
    """With two equal maxima, the first index wins."""
    var logits = [0.5, 2.0, 2.0, 1.0]
    assert_equal(argmax(logits), 1)


def test_argmax_single_element() raises:
    """A one-element input returns index 0."""
    var logits = [42.0]
    assert_equal(argmax(logits), 0)


def test_argmax_negative_values() raises:
    """Finds the maximum among all-negative values."""
    var logits = [-3.0, -1.0, -2.0]
    assert_equal(argmax(logits), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
