"""Tests for GELU — the tanh approximation GPT-2 was trained with.

The frozen goldens pin the tanh form; the tight tolerance rejects the erf-exact
variant, which would drift GPT-2's logit parity. gelu_rows must agree with scalar
gelu elementwise.
"""

from std.testing import assert_almost_equal, assert_true, TestSuite

from llm.nn.gelu import gelu, gelu_rows
from llm.tensor.tensor2d import from_rows


def test_gelu_scalar_goldens() raises:
    """Scalar gelu matches the tanh-approx goldens from tests/oracles/nn_reference.py.
    """
    assert_almost_equal(gelu(-3.0), -0.0036373920817729943, atol=1e-12)
    assert_almost_equal(gelu(-1.0), -0.1588080093917233, atol=1e-12)
    assert_almost_equal(gelu(-0.5), -0.15428599017485606, atol=1e-12)
    assert_almost_equal(gelu(0.0), 0.0, atol=1e-12)
    assert_almost_equal(gelu(0.5), 0.34571400982514394, atol=1e-12)
    assert_almost_equal(gelu(1.0), 0.8411919906082768, atol=1e-12)
    assert_almost_equal(gelu(3.0), 2.996362607918227, atol=1e-12)


def test_gelu_rejects_erf_variant() raises:
    """The tanh gelu(1.0) is measurably far from the erf-exact GELU, not confused with it.
    """
    # The erf-exact GELU at x = 1 is 0.8413447460685429; our tanh value must differ.
    assert_true(abs(gelu(1.0) - 0.8413447460685429) > 1e-4)


def test_gelu_zero() raises:
    """GELU of zero is exactly 0."""
    assert_almost_equal(gelu(0.0), 0.0, atol=1e-15)


def test_gelu_asymptotes() raises:
    """Large positive x -> ~x (gate saturates to 1); large negative x -> ~0."""
    assert_almost_equal(gelu(10.0), 10.0, atol=1e-6)
    assert_almost_equal(gelu(-10.0), 0.0, atol=1e-6)


def test_gelu_rows_matches_scalar_elementwise() raises:
    """The gelu_rows kernel agrees with scalar gelu at every element."""
    var x = from_rows([[-3.0, -0.5, 0.0], [0.5, 1.0, 3.0]])
    var y = gelu_rows(x)
    for r in range(x.rows):
        for c in range(x.cols):
            assert_almost_equal(y[r, c], gelu(x[r, c]), atol=1e-15)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
