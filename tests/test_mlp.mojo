"""Tests for MLP — the two-Linear feed-forward block, up -> gelu -> down.

Two things are pinned: the composition equals manually running up, gelu_rows,
then down (right wiring order), and it matches an independent oracle golden (the
pieces are right). The hidden width follows the constructor argument, never a
hardcoded 4x.
"""

from std.testing import assert_almost_equal, assert_equal, TestSuite

from llm.nn.gelu import gelu_rows
from llm.nn.linear import Linear
from llm.nn.mlp import MLP
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import from_rows
from llm.utils.random import Rng


def make_mlp() raises -> MLP:
    """Build an MLP with d_model=2, d_hidden=3 matching the nn_reference golden.
    """
    var up_w = from_rows(
        [[0.5, -0.5], [1.0, 0.0], [0.0, 1.0]]
    )  # [hidden=3, C=2]
    var up_b = from_rows([[0.1, 0.2, -0.1]])  # [1, hidden=3]
    var down_w = from_rows(
        [[1.0, 0.5, -1.0], [0.0, 1.0, 2.0]]
    )  # [C=2, hidden=3]
    var down_b = from_rows([[0.0, 0.5]])  # [1, C=2]
    var up = Linear(Parameter(up_w^), Parameter(up_b^))
    var down = Linear(Parameter(down_w^), Parameter(down_b^))
    return MLP(up^, down^)


def test_forward_matches_oracle_golden() raises:
    """MLP.forward matches the nn_reference.py oracle golden."""
    var mlp = make_mlp()
    var x = from_rows([[1.0, 2.0], [-1.0, 0.5]])  # [N=2, C=2]
    var y = mlp.forward(x)
    assert_equal(y.rows, 2)
    assert_equal(y.cols, 2)
    assert_almost_equal(y[0, 0], -1.4524386804590441, atol=1e-12)
    assert_almost_equal(y[0, 1], 5.25260523804853, atol=1e-12)
    assert_almost_equal(y[1, 0], -0.5145885774902862, atol=1e-12)
    assert_almost_equal(y[1, 1], 0.8547540302911605, atol=1e-12)


def test_forward_equals_manual_composition() raises:
    """MLP.forward exactly equals manual up -> gelu_rows -> down."""
    # Running the steps independently pins that the block can't reorder or drop
    # one unnoticed.
    var mlp = make_mlp()
    var x = from_rows([[1.0, 2.0], [-1.0, 0.5]])
    var y = mlp.forward(x)
    var manual = mlp.down.forward(gelu_rows(mlp.up.forward(x)))
    for r in range(y.rows):
        for c in range(y.cols):
            assert_almost_equal(y[r, c], manual[r, c], atol=1e-15)


def test_hidden_width_reflects_constructor() raises:
    """Weight shapes follow the constructor's d_model and d_hidden arguments."""
    var rng = Rng(3)
    var mlp = MLP.init_random(rng, 4, 16)  # d_model=4, d_hidden=16
    # up: C -> hidden, weight [hidden, C]
    assert_equal(mlp.up.weight.value.rows, 16)
    assert_equal(mlp.up.weight.value.cols, 4)
    # down: hidden -> C, weight [C, hidden]
    assert_equal(mlp.down.weight.value.rows, 4)
    assert_equal(mlp.down.weight.value.cols, 16)


def test_forward_shape_contract() raises:
    """MLP.forward maps [N, C] back to [N, d_model]."""
    var rng = Rng(1)
    var mlp = MLP.init_random(rng, 4, 16)
    var x = from_rows(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    )  # [N=2, C=4]
    var y = mlp.forward(x)
    assert_equal(y.rows, 2)
    assert_equal(y.cols, 4)  # back to d_model


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
