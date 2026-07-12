"""A minimal full-batch training loop for the bigram model.

Each step computes the loss and its gradient on one fixed batch, then takes an
SGD step. It returns the per-step loss history rather than printing, so tests can
assert on the curve instead of scraping stdout. This is the smallest correct
loop; the real trainer (multi-batch, epochs, scheduling) grows from here.
"""

from llm.models.bigram import BigramLM
from llm.tensor.tensor2d import zeros_2d
from llm.data.batch import TokenBatch
from llm.training.optimizer import sgd_step


def train_bigram(
    mut model: BigramLM,
    batch: TokenBatch,
    steps: Int,
    learning_rate: Float64,
) raises -> List[Float64]:
    """Run `steps` full-batch gradient-descent updates on the fixed `batch`.

    Args:
        model: The bigram model to train in place.
        batch: The fixed token batch to train on.
        steps: Number of gradient-descent updates.
        learning_rate: The SGD step size.

    Returns:
        The loss measured at the start of each step, so history[k] is the loss
        after k updates. history[0] is the pre-training loss; the loss after the
        final update is not recorded, so pass one extra step if you need it.
        Allocates the list; mutates model.
    """
    var history = List[Float64]()
    var grad = zeros_2d(model.vocab_size, model.vocab_size)
    for _ in range(steps):
        var loss = model.loss_and_grad(batch, grad)
        history.append(loss)
        sgd_step(model.table, grad, learning_rate)
    return history^
