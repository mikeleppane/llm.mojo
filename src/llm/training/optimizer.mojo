"""Training-loop-level optimizer machinery.

Plain stochastic gradient descent over a bare Tensor2D (sgd_step), which the
bigram consumes. The per-Parameter AdamW update lives in nn/optim.mojo; here we
add the whole-model pieces: global-norm gradient clipping (clip_grad_norm),
applied after gradient accumulation and before the optimizer step.
"""


from std.math import isfinite

from llm.tensor.tensor2d import Tensor2D
from llm.transformer.gpt import GPT


def sgd_step(
    mut param: Tensor2D, grad: Tensor2D, learning_rate: Float64
) raises:
    """Apply an in-place gradient-descent update: param -= learning_rate * grad.

    Args:
        param: The parameter tensor to update in place.
        grad: The gradient, same shape as param.
        learning_rate: The step size.

    Raises:
        Error: If param and grad shapes differ, so a mis-shaped gradient is
            caught rather than silently truncating.
    """
    if param.rows != grad.rows or param.cols != grad.cols:
        raise Error("sgd_step: param and grad shapes must match")
    for i in range(param.rows):
        for j in range(param.cols):
            param[i, j] = param[i, j] - learning_rate * grad[i, j]


def clip_grad_norm(mut gpt: GPT, max_norm: Float64) raises -> Float64:
    """Rescale the whole-model gradient so its L2 norm is at most max_norm.

    The norm is the single vector norm over every gradient entry in the model
    (gpt.grad_norm() over wte, wpe, all blocks, ln_f), not a per-tensor norm. If
    it exceeds max_norm, every gradient is multiplied by max_norm / norm
    (gpt.scale_grads), bringing the global norm to exactly max_norm while
    preserving direction. Below the threshold it is a bit-for-bit no-op; a zero
    gradient never triggers the division. Applied after the batch's gradient
    accumulation and before the optimizer step.

    Args:
        gpt: The model whose gradients are clipped in place.
        max_norm: The clip threshold; must be positive.

    Returns:
        The global gradient norm before clipping (for logging). Mutates gpt's
        gradients only when clipping fires; allocates nothing.

    Raises:
        Error: If max_norm <= 0, which has no meaning as a clip.
    """
    if max_norm <= 0.0:
        raise Error(
            "clip_grad_norm: max_norm must be positive, got " + String(max_norm)
        )
    var norm = gpt.grad_norm()
    if norm > max_norm:
        gpt.scale_grads(max_norm / norm)
    return norm


@fieldwise_init
struct AdamWConfig(Copyable, Movable):
    """The AdamW hyperparameters a run holds, separate from TrainingConfig.

    TrainingConfig keeps batch_size, the peak learning_rate, max_steps, and seed.
    These optimizer knobs live in their own struct so config.mojo (a model-shape
    concern whose GPT-2 preset never carried optimizer knobs) stays untouched.
    """

    var beta1: Float64  # first-moment decay
    var beta2: Float64  # second-moment decay
    var eps: Float64  # denominator floor
    var weight_decay: Float64  # decoupled decay coefficient
    var grad_clip: Float64  # global gradient-norm clip threshold

    @staticmethod
    def gpt2_defaults() -> AdamWConfig:
        """Build the GPT-training preset.

        beta2 is 0.95, the GPT-family value, not Adam's 0.999 habit. eps 1e-8,
        weight decay 0.1, grad clip 1.0.

        Returns:
            An AdamWConfig with the GPT-2 defaults.
        """
        return AdamWConfig(0.9, 0.95, 1e-8, 0.1, 1.0)

    def validate(self) raises:
        """Validate the hyperparameters.

        Raises:
            Error: On the first invalid or non-finite field, naming it. Betas
                live in [0, 1); eps and grad_clip must be positive; weight_decay
                must be >= 0.
        """
        # Negated ranges reject NaN and +inf too: every comparison with NaN is
        # false, so a bare `< 0 or >= 1` would let a NaN beta through, and a
        # non-finite hyperparameter silently poisons every optimizer step. The
        # lower-bounded fields (eps, weight_decay, grad_clip) also need isfinite,
        # since +inf passes their one-sided bound.
        if not (self.beta1 >= 0.0 and self.beta1 < 1.0):
            raise Error("AdamWConfig: beta1 must be in [0, 1)")
        if not (self.beta2 >= 0.0 and self.beta2 < 1.0):
            raise Error("AdamWConfig: beta2 must be in [0, 1)")
        if not (isfinite(self.eps) and self.eps > 0.0):
            raise Error("AdamWConfig: eps must be positive")
        if not (isfinite(self.weight_decay) and self.weight_decay >= 0.0):
            raise Error("AdamWConfig: weight_decay must be >= 0")
        if not (isfinite(self.grad_clip) and self.grad_clip > 0.0):
            raise Error("AdamWConfig: grad_clip must be positive")
