# Tests for MLP — the two-Linear feed-forward block, up -> gelu -> down.
#
# Two things are pinned: the composition equals manually running up, gelu_rows,
# then down (so the block wires its pieces in the right order), and it matches an
# independent oracle golden (so the pieces themselves are right). The hidden width
# is whatever the constructor was given — GPT-2 passes 4C later, but the block
# never hardcodes 4x, so the weight shapes reflect the argument.

from std.testing import assert_almost_equal, assert_equal, TestSuite

from llm.nn.gelu import gelu_rows
from llm.nn.linear import Linear
from llm.nn.mlp import MLP
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import from_rows
from llm.utils.random import Rng


def make_mlp() raises -> MLP:
    # d_model = 2, d_hidden = 3 — matches the nn_reference.py MLP golden.
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
    # Golden from tests/oracles/nn_reference.py ("MLP d_model=2 d_hidden=3").
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
    # Independently run up -> gelu_rows -> down and require an exact match, so the
    # block can't reorder or drop a step unnoticed.
    var mlp = make_mlp()
    var x = from_rows([[1.0, 2.0], [-1.0, 0.5]])
    var y = mlp.forward(x)
    var manual = mlp.down.forward(gelu_rows(mlp.up.forward(x)))
    for r in range(y.rows):
        for c in range(y.cols):
            assert_almost_equal(y[r, c], manual[r, c], atol=1e-15)


def test_hidden_width_reflects_constructor() raises:
    var rng = Rng(3)
    var mlp = MLP.init_random(rng, 4, 16)  # d_model=4, d_hidden=16
    # up: C -> hidden, weight [hidden, C]
    assert_equal(mlp.up.weight.value.rows, 16)
    assert_equal(mlp.up.weight.value.cols, 4)
    # down: hidden -> C, weight [C, hidden]
    assert_equal(mlp.down.weight.value.rows, 4)
    assert_equal(mlp.down.weight.value.cols, 16)


def test_forward_shape_contract() raises:
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
