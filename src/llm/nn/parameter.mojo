# Parameter — a trainable tensor paired with its gradient.
#
# Every layer in this package owns Parameters, not bare tensors. A Parameter is
# the smallest shared building block the whole network agrees on: a `value`
# tensor the forward pass reads, and a `grad` tensor of the same shape that the
# backward pass (a later part) fills and the optimizer (a later part still)
# consumes. Bundling them now means those later parts add *methods*, not a
# rewrite of every layer's fields.
#
# Until the backward pass exists, `grad` stays all-zeros — it is allocated here
# so its shape is pinned from day one, and `zero_grad()` is the reset the
# training loop calls between steps. Preferring a plain struct over a `Module`
# trait is deliberate: layer signatures genuinely differ (ids vs floats vs
# masks), so a shared parameter type buys the uniformity a trait would, without
# a one-implementer-per-shape hierarchy.

from llm.tensor.tensor2d import Tensor2D, zeros_2d


struct Parameter(Copyable, Movable):
    var value: Tensor2D  # the trainable weights
    var grad: Tensor2D  # same shape as value; zeros until the backward pass

    def __init__(out self, var value: Tensor2D):
        # Take ownership of `value` and allocate a matching zeros grad. Reads
        # value's shape before moving it in (Tensor2D is Copyable but not
        # ImplicitlyCopyable, so the move needs `^`). Allocates the grad buffer;
        # cannot raise.
        self.grad = zeros_2d(value.rows, value.cols)
        self.value = value^

    def zero_grad(mut self):
        # Reset every gradient entry to zero, leaving `value` untouched. Mutates
        # in place, allocates nothing, cannot raise. Called between optimizer
        # steps so gradients accumulate within a step, not across steps.
        self.grad.fill(0.0)
