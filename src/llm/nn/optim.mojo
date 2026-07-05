# Per-parameter optimizer updates — the in-place math one Parameter's step needs.
#
# This is the layering-honest home for optimizer arithmetic. A step operates on a
# Parameter (its value and grad) plus optimizer state tensors the caller owns; it
# is Parameter-level math, so it belongs in `nn/` — the package that owns
# Parameter — not in `training/`. The model (`transformer/GPT`) must call this
# math from its walk methods, and `transformer/` imports `nn/` but never
# `training/` (the dependency layering runs nn -> transformer -> training). Free
# `training.optimizer.sgd_step` (over bare Tensor2D) stays where it is; the bigram
# consumes it. These two functions are what the GPT walk methods delegate to.

from std.math import sqrt

from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D


def sgd_update(mut p: Parameter, lr: Float64):
    # Plain SGD, in place: p.value -= lr * p.grad.
    #   in/out: reads and writes p.value [R, C]; reads p.grad [R, C].
    #   mutates: p.value in place.
    #   allocates: nothing.
    #   raises: never — value and grad always share a shape (allocated together
    #           in Parameter), so no shape check is needed or possible to trip.
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
    # One tensor's AdamW step, in place (Loshchilov & Hutter's decoupled-decay
    # variant — the "W" in AdamW). Given the gradient g = p.grad and the running
    # first/second moments m, v (carried across steps by the caller), with the
    # step counter t starting at 1:
    #
    #   m <- beta1*m + (1 - beta1)*g          first moment  (mean of g)
    #   v <- beta2*v + (1 - beta2)*g^2        second moment (mean of g^2)
    #   mhat = m / (1 - beta1^t)              bias-corrected first moment
    #   vhat = v / (1 - beta2^t)              bias-corrected second moment
    #   value <- value - lr*( mhat/(sqrt(vhat) + eps) + weight_decay*value )
    #
    # The moments are initialized to zero, so early on they are biased toward
    # zero; the (1 - beta^t) denominators correct that. At t = 1, 1 - beta1^1 =
    # 1 - beta1, which exactly cancels the (1 - beta1) that formed m, so
    # mhat = g and vhat = g^2 on the first step (the update's adaptive term is
    # then g/(|g| + eps), ~ sign(g)) — the check that bias correction starts at 1.
    #
    # Decay is DECOUPLED: weight_decay*value is added to the update directly, NOT
    # folded into g or the moments (that is Adam + L2, a different algorithm). The
    # observable consequence, and the pin: with g = 0 the moments stay zero and
    # the adaptive term is zero, yet value still shrinks to value*(1 - lr*wd).
    #
    #   in/out: reads and writes p.value [R, C]; reads p.grad [R, C]; reads and
    #           writes m [R, C] and v [R, C] (the caller's state, advanced here).
    #   mutates: p.value, m, v in place.
    #   allocates: nothing.
    #   raises: on t < 1 (bias correction is undefined), or if p.value, p.grad,
    #           m, v do not all share one shape (a mis-aligned state tensor).
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
