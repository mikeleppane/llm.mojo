# Finite-difference tests for MLP.backward — chain-rule wiring end to end.
#
# The MLP is down(gelu(up(x))): composition is where chain-rule bugs live (a
# transposed factor, a stage backward fed the wrong cache, gelu applied to the
# post- instead of pre-activation). So every gradient the composition produces is
# checked against a central finite difference of the whole MLP forward: dx and
# all four parameter grads (up weight/bias, down weight/bias).
#
# Finite-difference convention (D5, shared across this part's backward tests):
#   L = sum(cotangent ⊙ y); central diff h = 1e-5; tolerance
#   |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|.

from std.testing import assert_true, TestSuite

from llm.nn.linear import Linear
from llm.nn.mlp import MLP
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, from_rows
from llm.utils.random import Rng


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    # D5 mixed tolerance |a - n| <= 1e-7 + 1e-5 * |n|.
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


def build_mlp(
    uw: Tensor2D, ub: Tensor2D, dw: Tensor2D, db: Tensor2D
) raises -> MLP:
    # Assemble an MLP from four weight tensors (copied in) so a finite-difference
    # loop can perturb any one entry and rebuild.
    var up = Linear(Parameter(uw.copy()), Parameter(ub.copy()))
    var down = Linear(Parameter(dw.copy()), Parameter(db.copy()))
    return MLP(up^, down^)


def sample_input() raises -> Tensor2D:
    # [N=3, C=4].
    return from_rows(
        [[1.0, 0.5, -1.0, 0.3], [0.2, -0.4, 0.9, -1.1], [-0.7, 1.2, 0.1, 0.6]]
    )


def cotangent() raises -> Tensor2D:
    # Fixed asymmetric d_out [N=3, C=4].
    return from_rows(
        [[0.7, -0.2, 1.3, -0.5], [0.1, 0.9, -1.1, 0.4], [-0.6, 0.3, 0.2, -0.8]]
    )


def projected(mlp: MLP, x: Tensor2D, cot: Tensor2D) raises -> Float64:
    var y = mlp.forward(x)
    var total = 0.0
    for i in range(y.rows):
        for j in range(y.cols):
            total += cot[i, j] * y[i, j]
    return total


def finite_diff_param(
    uw: Tensor2D,
    ub: Tensor2D,
    dw: Tensor2D,
    db: Tensor2D,
    which: Int,
    r: Int,
    c: Int,
    x: Tensor2D,
    cot: Tensor2D,
) raises -> Float64:
    # Central difference of the projected loss wrt one parameter entry. `which`
    # selects the tensor: 0=up weight, 1=up bias, 2=down weight, 3=down bias.
    var h = 1e-5
    var uw_p = uw.copy()
    var ub_p = ub.copy()
    var dw_p = dw.copy()
    var db_p = db.copy()
    var uw_m = uw.copy()
    var ub_m = ub.copy()
    var dw_m = dw.copy()
    var db_m = db.copy()
    if which == 0:
        uw_p[r, c] = uw_p[r, c] + h
        uw_m[r, c] = uw_m[r, c] - h
    elif which == 1:
        ub_p[r, c] = ub_p[r, c] + h
        ub_m[r, c] = ub_m[r, c] - h
    elif which == 2:
        dw_p[r, c] = dw_p[r, c] + h
        dw_m[r, c] = dw_m[r, c] - h
    else:
        db_p[r, c] = db_p[r, c] + h
        db_m[r, c] = db_m[r, c] - h
    var plus = projected(build_mlp(uw_p, ub_p, dw_p, db_p), x, cot)
    var minus = projected(build_mlp(uw_m, ub_m, dw_m, db_m), x, cot)
    return (plus - minus) / (2.0 * h)


def base_weights() raises -> List[Tensor2D]:
    # Deterministic seeded weights for C=4, hidden=6; returned as [uw, ub, dw, db].
    var rng = Rng(11)
    var base = MLP.init_random(rng, 4, 6)
    var out = List[Tensor2D]()
    out.append(base.up.weight.value.copy())  # [6, 4]
    out.append(base.up.bias.value.copy())  # [1, 6]
    out.append(base.down.weight.value.copy())  # [4, 6]
    out.append(base.down.bias.value.copy())  # [1, 4]
    return out^


def test_d_x_matches_finite_difference() raises:
    var w = base_weights()
    var mlp = build_mlp(w[0], w[1], w[2], w[3])
    var x = sample_input()
    var cot = cotangent()
    var fwd = mlp.forward_cached(x)
    var d_x = mlp.backward(fwd.cache, cot)

    var h = 1e-5
    for i in range(x.rows):
        for j in range(x.cols):
            var plus = x.copy()
            plus[i, j] = plus[i, j] + h
            var minus = x.copy()
            minus[i, j] = minus[i, j] - h
            var numeric = (
                projected(mlp, plus, cot) - projected(mlp, minus, cot)
            ) / (2.0 * h)
            assert_grad_close(d_x[i, j], numeric)


def test_parameter_grads_match_finite_difference() raises:
    var w = base_weights()
    var mlp = build_mlp(w[0], w[1], w[2], w[3])
    var x = sample_input()
    var cot = cotangent()
    mlp.up.weight.zero_grad()
    mlp.up.bias.zero_grad()
    mlp.down.weight.zero_grad()
    mlp.down.bias.zero_grad()
    var fwd = mlp.forward_cached(x)
    _ = mlp.backward(fwd.cache, cot)

    # up weight [6, 4]
    for r in range(mlp.up.weight.value.rows):
        for c in range(mlp.up.weight.value.cols):
            var numeric = finite_diff_param(
                w[0], w[1], w[2], w[3], 0, r, c, x, cot
            )
            assert_grad_close(mlp.up.weight.grad[r, c], numeric)
    # up bias [1, 6]
    for c in range(mlp.up.bias.value.cols):
        var numeric = finite_diff_param(w[0], w[1], w[2], w[3], 1, 0, c, x, cot)
        assert_grad_close(mlp.up.bias.grad[0, c], numeric)
    # down weight [4, 6]
    for r in range(mlp.down.weight.value.rows):
        for c in range(mlp.down.weight.value.cols):
            var numeric = finite_diff_param(
                w[0], w[1], w[2], w[3], 2, r, c, x, cot
            )
            assert_grad_close(mlp.down.weight.grad[r, c], numeric)
    # down bias [1, 4]
    for c in range(mlp.down.bias.value.cols):
        var numeric = finite_diff_param(w[0], w[1], w[2], w[3], 3, 0, c, x, cot)
        assert_grad_close(mlp.down.bias.grad[0, c], numeric)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
