"""Per-parameter optimizer updates — the in-place math one Parameter's step needs.

SGD and AdamW steps that operate directly on a Parameter (its value and grad)
plus caller-owned optimizer state. Lives in `nn/` because it is Parameter-level
math.
"""

from std.math import sqrt

from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D


def sgd_update(mut p: Parameter, lr: Float64):
    """Plain SGD, in place: `p.value -= lr * p.grad`.

    Args:
        p: Parameter whose value [R, C] is updated from its grad [R, C].
        lr: Learning rate.

    Mutates p.value in place; allocates nothing. Value and grad always share a
    shape (allocated together), so no shape check is needed.
    """
    for i in range(p.value.rows):
        for j in range(p.value.cols):
            p.value[i, j] = p.value[i, j] - lr * p.grad[i, j]


def adamw_update(
    mut p: Parameter,
    mut m: Tensor2D,
    mut v: Tensor2D,
    t: Int,
    lr: Float64,
    beta1: Float64,
    beta2: Float64,
    eps: Float64,
    weight_decay: Float64,
) raises:
    """One tensor's AdamW step, in place (Loshchilov & Hutter decoupled decay).

    Given gradient g = p.grad and running first/second moments m, v (carried
    across steps by the caller), with step counter t starting at 1:

        m <- beta1*m + (1 - beta1)*g          first moment  (mean of g)
        v <- beta2*v + (1 - beta2)*g^2        second moment (mean of g^2)
        mhat = m / (1 - beta1^t)              bias-corrected first moment
        vhat = v / (1 - beta2^t)              bias-corrected second moment
        value <- value - lr*( mhat/(sqrt(vhat) + eps) + weight_decay*value )

    Decay is decoupled: weight_decay*value enters the update directly, not folded
    into g or the moments (that would be Adam + L2). The (1 - beta^t) denominators
    correct the zero-initialized moments' early bias toward zero.

    Args:
        p: Parameter whose value [R, C] is updated from its grad [R, C].
        m: First-moment state [R, C], advanced in place.
        v: Second-moment state [R, C], advanced in place.
        t: Step counter, starting at 1.
        lr: Learning rate.
        beta1: First-moment decay.
        beta2: Second-moment decay.
        eps: Denominator floor.
        weight_decay: Decoupled decay coefficient.

    Mutates p.value, m, v in place; allocates nothing.

    Raises:
        Error: If t < 1, or if p.value, p.grad, m, v do not all share one shape.
    """
    if t < 1:
        raise Error(
            "adamw_update: step t must be >= 1 (bias correction), got "
            + String(t)
        )
    var r = p.value.rows
    var c = p.value.cols
    if p.grad.rows != r or p.grad.cols != c:
        raise Error("adamw_update: p.grad shape must match p.value")
    if m.rows != r or m.cols != c:
        raise Error("adamw_update: m shape must match p.value")
    if v.rows != r or v.cols != c:
        raise Error("adamw_update: v shape must match p.value")

    # Bias-correction denominators, computed once per tensor (shared by all
    # entries): 1 - beta^t. beta**t is Float64**Int (libm pow).
    var bias1 = 1.0 - beta1**t
    var bias2 = 1.0 - beta2**t

    for i in range(r):
        for j in range(c):
            var g = p.grad[i, j]
            var mij = beta1 * m[i, j] + (1.0 - beta1) * g
            var vij = beta2 * v[i, j] + (1.0 - beta2) * g * g
            m[i, j] = mij
            v[i, j] = vij
            var mhat = mij / bias1
            var vhat = vij / bias2
            var value = p.value[i, j]
            p.value[i, j] = value - lr * (
                mhat / (sqrt(vhat) + eps) + weight_decay * value
            )
