# Tests for nn.optim.adamw_update — one tensor's decoupled-decay AdamW step.
#
# Goldens are frozen from tests/oracles/adamw_reference.py (independent NumPy
# math; nothing under src/ or the suite imports it). Coverage:
#   - multi-step oracle goldens, decay ON and OFF (value, and the m/v moments);
#   - the hand-computed step 1: at t=1 the (1 - beta^1) bias correction exactly
#     cancels the (1 - beta) that formed the moment, so mhat = g and vhat = g^2;
#   - the decoupled-decay pin: with g = 0 the moments stay EXACTLY zero yet the
#     value still shrinks to value*(1 - lr*wd) — decay never flows through g/m/v;
#   - raises on t < 1 and on a mis-shaped state tensor;
#   - determinism (pure arithmetic: two identical runs agree bit-for-bit).

from std.math import sqrt

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.nn.optim import adamw_update
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, from_rows, zeros_2d


def _param(value: Tensor2D) raises -> Parameter:
    # A Parameter holding `value` with a fresh zeros grad.
    return Parameter(value.copy())


def _set_grad(mut p: Parameter, g: Tensor2D):
    # Overwrite p.grad with g (same shape).
    for i in range(g.rows):
        for j in range(g.cols):
            p.grad[i, j] = g[i, j]


# GPT-family defaults used across the oracle cases.
comptime B1 = 0.9
comptime B2 = 0.95
comptime EPS = 1e-8


def test_oracle_constant_grad_decay_on() raises:
    # Case A: value0 = [[0.5,-1.0],[2.0,0.25]], constant grad [[0.1,-0.2],
    # [0.3,0.05]], lr=0.01, wd=0.1, three steps. Goldens: adamw_reference.py.
    var p = _param(from_rows([[0.5, -1.0], [2.0, 0.25]]))
    var grad = from_rows([[0.1, -0.2], [0.3, 0.05]])
    var m = zeros_2d(2, 2)
    var v = zeros_2d(2, 2)
    for t in range(1, 4):
        _set_grad(p, grad)
        adamw_update(p, m, v, t, 0.01, B1, B2, EPS, 0.1)

    # value after step 3
    assert_almost_equal(p.value[0, 0], 0.4685314924970007, atol=1e-12)
    assert_almost_equal(p.value[0, 1], -0.9670329904985003, atol=1e-12)
    assert_almost_equal(p.value[1, 0], 1.9640359889990002, atol=1e-12)
    assert_almost_equal(p.value[1, 1], 0.2192807457440008, atol=1e-12)
    # first moment m after step 3
    assert_almost_equal(m[0, 0], 0.027099999999999996, atol=1e-12)
    assert_almost_equal(m[0, 1], -0.05419999999999999, atol=1e-12)
    assert_almost_equal(m[1, 0], 0.08129999999999998, atol=1e-12)
    assert_almost_equal(m[1, 1], 0.013549999999999998, atol=1e-12)
    # second moment v after step 3
    assert_almost_equal(v[0, 0], 0.0014262500000000015, atol=1e-12)
    assert_almost_equal(v[0, 1], 0.005705000000000006, atol=1e-12)
    assert_almost_equal(v[1, 0], 0.01283625000000001, atol=1e-12)
    assert_almost_equal(v[1, 1], 0.0003565625000000004, atol=1e-12)


def test_oracle_constant_grad_decay_off() raises:
    # Case B: identical to A but wd=0.0. Same moments (decay never touches them),
    # different values. Goldens: adamw_reference.py.
    var p = _param(from_rows([[0.5, -1.0], [2.0, 0.25]]))
    var grad = from_rows([[0.1, -0.2], [0.3, 0.05]])
    var m = zeros_2d(2, 2)
    var v = zeros_2d(2, 2)
    for t in range(1, 4):
        _set_grad(p, grad)
        adamw_update(p, m, v, t, 0.01, B1, B2, EPS, 0.0)

    assert_almost_equal(p.value[0, 0], 0.4700000029999997, atol=1e-12)
    assert_almost_equal(p.value[0, 1], -0.9700000014999998, atol=1e-12)
    assert_almost_equal(p.value[1, 0], 1.970000001, atol=1e-12)
    assert_almost_equal(p.value[1, 1], 0.2200000059999988, atol=1e-12)
    # Moments are identical to the decay-on run (decay is decoupled from m/v).
    assert_almost_equal(m[1, 0], 0.08129999999999998, atol=1e-12)
    assert_almost_equal(v[1, 0], 0.01283625000000001, atol=1e-12)


def test_oracle_varying_grad_four_steps() raises:
    # Case C: value0 as above, lr=0.05, wd=0.1, four DIFFERENT gradients. A
    # varying-gradient run exercises the bias correction at t=1..4, not just the
    # steady state. Goldens: adamw_reference.py (step-4 value).
    var p = _param(from_rows([[0.5, -1.0], [2.0, 0.25]]))
    var grads = List[Tensor2D]()
    grads.append(from_rows([[0.1, -0.2], [0.3, 0.05]]))
    grads.append(from_rows([[-0.4, 0.1], [0.0, 0.2]]))
    grads.append(from_rows([[0.2, 0.2], [-0.1, -0.3]]))
    grads.append(from_rows([[0.05, -0.05], [0.15, 0.1]]))
    var m = zeros_2d(2, 2)
    var v = zeros_2d(2, 2)
    for t in range(1, 5):
        _set_grad(p, grads[t - 1])
        adamw_update(p, m, v, t, 0.05, B1, B2, EPS, 0.1)

    assert_almost_equal(p.value[0, 0], 0.47502153304870726, atol=1e-12)
    assert_almost_equal(p.value[0, 1], -0.9374823618438017, atol=1e-12)
    assert_almost_equal(p.value[1, 0], 1.8391497794621552, atol=1e-12)
    assert_almost_equal(p.value[1, 1], 0.1571345394950073, atol=1e-12)


def test_step_one_bias_correction_cancels() raises:
    # Hand check at t=1. With m=v=0 and one gradient g:
    #   m1   = (1-b1)*g,  v1 = (1-b2)*g^2
    #   mhat = m1/(1-b1^1) = m1/(1-b1) = g          (bias correction cancels)
    #   vhat = v1/(1-b2^1) = v1/(1-b2) = g^2
    #   value <- value - lr*( g/(sqrt(g^2)+eps) + wd*value )
    # For value=2.0, g=0.3, lr=0.01, wd=0.1:
    #   = 2.0 - 0.01*( 0.3/(0.3+1e-8) + 0.02 )  ~ 1.9880000003333334
    var p = _param(from_rows([[2.0]]))
    _set_grad(p, from_rows([[0.3]]))
    var m = zeros_2d(1, 1)
    var v = zeros_2d(1, 1)
    adamw_update(p, m, v, 1, 0.01, B1, B2, EPS, 0.1)
    assert_almost_equal(p.value[0, 0], 1.9880000003333334, atol=1e-12)
    # mhat == g and vhat == g^2 exactly at t=1: verify via the stored moments.
    # m1 = (1-b1)*g -> m1/(1-b1) must be g; v1 = (1-b2)*g^2 -> v1/(1-b2)=g^2.
    assert_almost_equal(m[0, 0] / (1.0 - B1), 0.3, atol=1e-15)
    assert_almost_equal(v[0, 0] / (1.0 - B2), 0.09, atol=1e-15)


def test_decoupled_decay_g_zero_shrinks_value() raises:
    # The W in AdamW. With g = 0 the moments stay EXACTLY zero (nothing to
    # accumulate) and the adaptive term is 0/(0+eps) = 0, so the ONLY change is
    # the decoupled decay: value <- value - lr*wd*value = value*(1 - lr*wd).
    # If decay were coupled through the gradient (Adam+L2), g=0 would leave the
    # value untouched — this is the test that tells the two apart.
    var p = _param(from_rows([[3.0]]))
    _set_grad(p, from_rows([[0.0]]))
    var m = zeros_2d(1, 1)
    var v = zeros_2d(1, 1)
    var lr = 0.01
    var wd = 0.1
    var expected = 3.0
    for t in range(1, 4):
        adamw_update(p, m, v, t, lr, B1, B2, EPS, wd)
        expected = expected * (1.0 - lr * wd)
        assert_almost_equal(p.value[0, 0], expected, atol=1e-14)
        # Moments never left zero — decay did not flow through g, m, or v.
        assert_true(m[0, 0] == 0.0, "first moment moved under g=0")
        assert_true(v[0, 0] == 0.0, "second moment moved under g=0")
    # Concrete goldens too (adamw_reference.py): 2.997, 2.994003, 2.991008997.
    assert_almost_equal(p.value[0, 0], 2.991008997, atol=1e-12)


def test_raises_on_t_below_one() raises:
    var p = _param(from_rows([[1.0]]))
    var m = zeros_2d(1, 1)
    var v = zeros_2d(1, 1)
    with assert_raises(contains="t must be >= 1"):
        adamw_update(p, m, v, 0, 0.01, B1, B2, EPS, 0.1)


def test_raises_on_shape_mismatch() raises:
    var p = _param(from_rows([[1.0, 2.0]]))  # [1, 2]
    var m = zeros_2d(1, 2)
    var v = zeros_2d(2, 1)  # wrong shape
    with assert_raises(contains="v shape must match"):
        adamw_update(p, m, v, 1, 0.01, B1, B2, EPS, 0.1)


def test_deterministic() raises:
    # Pure arithmetic: two identical runs produce bit-identical results.
    var pa = _param(from_rows([[0.5, -1.0], [2.0, 0.25]]))
    var pb = _param(from_rows([[0.5, -1.0], [2.0, 0.25]]))
    var grad = from_rows([[0.1, -0.2], [0.3, 0.05]])
    var ma = zeros_2d(2, 2)
    var va = zeros_2d(2, 2)
    var mb = zeros_2d(2, 2)
    var vb = zeros_2d(2, 2)
    for t in range(1, 6):
        _set_grad(pa, grad)
        _set_grad(pb, grad)
        adamw_update(pa, ma, va, t, 0.02, B1, B2, EPS, 0.1)
        adamw_update(pb, mb, vb, t, 0.02, B1, B2, EPS, 0.1)
    for i in range(2):
        for j in range(2):
            assert_true(
                pa.value[i, j] == pb.value[i, j], "value not bit-identical"
            )
            assert_true(ma[i, j] == mb[i, j], "m not bit-identical")
            assert_true(va[i, j] == vb[i, j], "v not bit-identical")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
