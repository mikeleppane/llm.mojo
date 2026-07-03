# The finite-difference step-size study.
#
# For f(x) = x^3 the central difference has an exactly computable truncation
# error: ((x+h)^3 - (x-h)^3) / 2h = 3x^2 + h^2. So the central difference
# overshoots the true derivative 3x^2 by exactly h^2 — which lets us watch the
# truncation/roundoff tradeoff in a test instead of taking it on faith. A
# well-chosen h is accurate; a large h shows the h^2 error precisely.

from std.testing import assert_almost_equal, TestSuite


def f(x: Float64) -> Float64:
    return x * x * x


def central_difference(x: Float64, h: Float64) -> Float64:
    return (f(x + h) - f(x - h)) / (2.0 * h)


def test_well_chosen_h_is_accurate() raises:
    # True derivative at x = 2 is 3 * 4 = 12.
    assert_almost_equal(central_difference(2.0, 1e-5), 12.0, atol=1e-8)


def test_large_h_shows_truncation_error() raises:
    # For f = x^3 the error is exactly h^2 = 0.01 at h = 0.1.
    assert_almost_equal(central_difference(2.0, 0.1), 12.01, atol=1e-9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
