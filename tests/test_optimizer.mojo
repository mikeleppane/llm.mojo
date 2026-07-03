# Tests for the SGD optimizer step.
#
# sgd_step is param -= learning_rate * grad in place. Pin the arithmetic on a
# hand-computed case and the shape-mismatch guard (its own, separate from the
# model's).

from std.testing import assert_almost_equal, assert_raises, TestSuite

from llm.tensor.tensor2d import from_rows, zeros_2d
from llm.training.optimizer import sgd_step


def test_sgd_step_updates_in_place() raises:
    var param = from_rows([[1.0, 2.0], [3.0, 4.0]])
    var grad = from_rows([[0.5, 1.0], [-2.0, 0.0]])
    sgd_step(param, grad, 0.1)
    assert_almost_equal(param[0, 0], 0.95, atol=1e-12)  # 1.0 - 0.1*0.5
    assert_almost_equal(param[0, 1], 1.9, atol=1e-12)  # 2.0 - 0.1*1.0
    assert_almost_equal(param[1, 0], 3.2, atol=1e-12)  # 3.0 - 0.1*(-2.0)
    assert_almost_equal(param[1, 1], 4.0, atol=1e-12)  # 4.0 - 0.1*0.0


def test_sgd_step_shape_mismatch_raises() raises:
    var param = zeros_2d(2, 3)
    var grad = zeros_2d(3, 2)
    with assert_raises(contains="shapes must match"):
        sgd_step(param, grad, 0.1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
