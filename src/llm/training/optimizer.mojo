# Optimizer machinery for training.
#
# The simplest optimizer first: plain stochastic gradient descent over a bare
# Tensor2D (sgd_step), which the bigram consumes. The per-Parameter AdamW update
# lives in nn/optim.mojo (the layer that owns Parameter); here we add the
# training-loop-level pieces that operate on the whole model: global-norm
# gradient clipping (clip_grad_norm), applied after gradient accumulation and
# before the optimizer step.


from llm.tensor.tensor2d import Tensor2D
from llm.transformer.gpt import GPT


def sgd_step(
    mut param: Tensor2D, grad: Tensor2D, learning_rate: Float64
) raises:
    # In-place gradient-descent update: param -= learning_rate * grad. Shapes
    # must match; raises otherwise so a mis-shaped gradient is caught rather than
    # silently truncating. Mutates `param`.
    if param.rows != grad.rows or param.cols != grad.cols:
        raise Error("sgd_step: param and grad shapes must match")
    for i in range(param.rows):
        for j in range(param.cols):
            param[i, j] = param[i, j] - learning_rate * grad[i, j]


def clip_grad_norm(mut gpt: GPT, max_norm: Float64) raises -> Float64:
    # Global-norm gradient clipping: rescale the WHOLE-model gradient so its L2
    # norm is at most max_norm, and return the norm BEFORE clipping (for logging).
    #
    # The norm is the single vector norm over every gradient entry in the model
    # (gpt.grad_norm() — wte once, wpe, all blocks, ln_f), NOT a per-tensor norm.
    # If that norm exceeds max_norm, every gradient is multiplied by
    # max_norm / norm (gpt.scale_grads), which brings the global norm to exactly
    # max_norm while preserving the gradient's DIRECTION. Below the threshold it
    # is exactly a no-op — no tensor is touched, so gradients pass through
    # bit-for-bit. A zero gradient has norm 0, which never exceeds a positive
    # max_norm, so the division max_norm / norm is never reached.
    #
    # Applied AFTER the batch's gradient accumulation and BEFORE the optimizer
    # step. Mutates gpt's gradients only when clipping fires; allocates nothing;
    # raises on max_norm <= 0 (a non-positive clip has no meaning).
    if max_norm <= 0.0:
        raise Error(
            "clip_grad_norm: max_norm must be positive, got " + String(max_norm)
        )
    var norm = gpt.grad_norm()
    if norm > max_norm:
        gpt.scale_grads(max_norm / norm)
    return norm
