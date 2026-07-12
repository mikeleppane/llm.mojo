"""Training-facing loss summaries built on the tensor-layer cross-entropy."""

from std.math import exp


def perplexity(mean_loss: Float64) -> Float64:
    """Compute perplexity = exp(mean cross-entropy).

    A uniform model over V tokens has loss log(V) and perplexity exactly V, so
    perplexity reads as an effective vocabulary size: how many equally-likely
    choices the model behaves as if it faces per step. Lower is better.

    Args:
        mean_loss: Mean per-position cross-entropy loss.

    Returns:
        The perplexity, exp(mean_loss).
    """
    return exp(mean_loss)
