"""Parameter — a trainable tensor paired with its gradient.

The smallest shared building block every layer owns: a `value` tensor the
forward reads, and a `grad` tensor of the same shape the backward fills and the
optimizer consumes.
"""

from llm.tensor.tensor2d import Tensor2D, zeros_2d


struct Parameter(Copyable, Movable):
    """A trainable `value` tensor bundled with its same-shape `grad` tensor."""

    var value: Tensor2D  # the trainable weights
    var grad: Tensor2D  # same shape as value; zeros until the backward pass

    def __init__(out self, var value: Tensor2D):
        """Take ownership of `value` and allocate a matching zeros grad.

        Args:
            value: The trainable weights, shape [R, C]; moved in.

        Returns:
            A Parameter owning `value` with a zeros grad. Allocates the grad.
        """
        self.grad = zeros_2d(value.rows, value.cols)
        self.value = value^

    def zero_grad(mut self):
        """Reset every gradient entry to zero, leaving `value` untouched.

        Called between optimizer steps so gradients accumulate within a step,
        not across steps. Mutates in place; allocates nothing.
        """
        self.grad.fill(0.0)
