# Tests for cross-entropy and its gradient.
#
# Uniform logits give loss = log(V) (hand-computable). The gradient p - y sums
# to zero for a valid distribution and one target. logsumexp is checked for the
# stability that motivates it.

from std.testing import assert_almost_equal, assert_raises, TestSuite
from std.math import log

from llm.tensor.ops import cross_entropy_one, cross_entropy_grad, logsumexp


def test_uniform_logits_loss_is_log_v() raises:
    var logits = [0.0, 0.0, 0.0, 0.0]
    var loss = cross_entropy_one(logits, 2)
    assert_almost_equal(loss, log(4.0), atol=1e-12)


def test_grad_sums_to_zero() raises:
    var logits = [1.0, 2.0, 3.0]
    var g = cross_entropy_grad(logits, 1)
    var s = 0.0
    for i in range(len(g)):
        s += g[i]
    assert_almost_equal(s, 0.0, atol=1e-12)


def test_target_out_of_range_raises() raises:
    var logits = [0.0, 1.0]
    with assert_raises(contains="target out of range"):
        _ = cross_entropy_one(logits, 5)


def test_grad_target_out_of_range_raises() raises:
    # The gradient rejects a bad target symmetrically with the loss, rather than
    # writing out of bounds.
    var logits = [0.0, 1.0]
    with assert_raises(contains="target out of range"):
        _ = cross_entropy_grad(logits, 5)
    with assert_raises(contains="target out of range"):
        _ = cross_entropy_grad(logits, -1)


def test_logsumexp_stable_under_large_values() raises:
    # logsumexp of three equal 1000s is 1000 + log(3); a naive sum of exp(1000)
    # would overflow.
    var logits = [1000.0, 1000.0, 1000.0]
    var v = logsumexp(logits)
    assert_almost_equal(v, 1000.0 + log(3.0), atol=1e-9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
