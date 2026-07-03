# Parameter updates for training.
#
# The simplest optimizer: plain stochastic gradient descent, one step of
# param -= learning_rate * grad. No momentum, no adaptive rates — those arrive
# with the real trainer. Kept as a free function over a Tensor2D parameter so the
# bigram table (and, later, any weight matrix) updates the same way.


from llm.tensor.tensor2d import Tensor2D


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
