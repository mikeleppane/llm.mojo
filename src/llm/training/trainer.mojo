# A minimal training loop for the bigram model.
#
# Full-batch gradient descent on one fixed batch: each step computes the loss and
# its gradient, then takes an SGD step. It returns the per-step loss history
# rather than printing, so tests can assert on the curve (monotone decrease,
# overfit to near zero, convergence to the count-model optimum) instead of
# scraping stdout. This is deliberately the smallest correct loop; the real
# trainer (multi-batch, epochs, scheduling) grows from here.

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
    # Run `steps` full-batch gradient-descent updates on the fixed `batch`,
    # returning the loss measured at the start of each step (so history[k] is the
    # loss after k updates). Mutates `model`. history[0] is the pre-training loss;
    # the loss after the final update is not recorded, so pass one extra step if
    # you need it. Allocates and returns the history list.
    var history = List[Float64]()
    var grad = zeros_2d(model.vocab_size, model.vocab_size)
    for _ in range(steps):
        var loss = model.loss_and_grad(batch, grad)
        history.append(loss)
        sgd_step(model.table, grad, learning_rate)
    return history^
