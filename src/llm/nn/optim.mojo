# Per-parameter optimizer updates — the in-place math one Parameter's step needs.
#
# This is the layering-honest home for optimizer arithmetic. A step operates on a
# Parameter (its value and grad) plus optimizer state tensors the caller owns; it
# is Parameter-level math, so it belongs in `nn/` — the package that owns
# Parameter — not in `training/`. The model (`transformer/GPT`) must call this
# math from its walk methods, and `transformer/` imports `nn/` but never
# `training/` (the dependency layering runs nn -> transformer -> training). Free
# `training.optimizer.sgd_step` (over bare Tensor2D) stays where it is; the bigram
# consumes it. These functions are what the GPT walk methods delegate to.

from llm.nn.parameter import Parameter


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
