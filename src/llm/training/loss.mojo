# Loss-related helpers for training.
#
# The per-position cross-entropy itself lives in the tensor layer
# (cross_entropy_one); this module holds the training-facing summaries built on
# top of it. For now that is perplexity — the exponential of the mean loss, the
# number usually reported alongside a language model's loss.

from std.math import exp


def perplexity(mean_loss: Float64) -> Float64:
    # Perplexity = exp(mean cross-entropy). A uniform model over V tokens has
    # loss log(V) and perplexity exactly V, so perplexity reads as an "effective
    # vocabulary size": how many equally-likely choices the model behaves as if
    # it faces per step. Lower is better.
    return exp(mean_loss)
