"""Finite-difference gradient check for cross_entropy_grad, verifying the analytic gradient against a central difference df/dx ~ (f(x+h) - f(x-h)) / 2h with h = 1e-6."""

from std.testing import assert_almost_equal, TestSuite

from llm.tensor.ops import cross_entropy_one, cross_entropy_grad


def perturbed_loss(
    logits: List[Float64], idx: Int, delta: Float64, target: Int
) raises -> Float64:
    var x = logits.copy()
    x[idx] = x[idx] + delta
    return cross_entropy_one(x, target)


def test_cross_entropy_gradient_matches_finite_difference() raises:
    """The analytic cross_entropy_grad matches a central finite difference at every logit.
    """
    var logits = [0.5, -1.0, 2.0, 0.25]
    var target = 2
    var analytic = cross_entropy_grad(logits, target)

    var h = 1e-6
    for i in range(len(logits)):
        var plus = perturbed_loss(logits, i, h, target)
        var minus = perturbed_loss(logits, i, -h, target)
        var numeric = (plus - minus) / (2.0 * h)
        assert_almost_equal(analytic[i], numeric, atol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
