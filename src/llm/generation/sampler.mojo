# Sampling a token from a probability distribution.
#
# The one primitive generation needs before anything fancier: draw an index from
# a categorical distribution. It walks the cumulative distribution (inverse-CDF
# sampling) using a single uniform draw, so the same seeded Rng that shuffles
# batches also makes generation reproducible. Top-k / top-p and the real
# generate loop build on this later.

from llm.utils.random import Rng


def sample_categorical(probs: List[Float64], mut rng: Rng) raises -> Int:
    # Draw an index i with probability probs[i] by inverse-CDF sampling: take a
    # threshold in [0, total) and return the first i where the running cumulative
    # sum exceeds it. Raises on an empty distribution, a negative entry, or a
    # total that is not ~1 (tolerance 1e-6), catching un-normalized input early.
    # Mutates rng (consumes one draw).
    #
    # The threshold is rng.uniform() * total, NOT rng.uniform(). Scaling by the
    # actual total matters because the sum is only required to be within 1e-6 of
    # 1: an un-scaled u could land in the gap [total, 1) and fall through the loop
    # to the last index — which might be a zero-probability entry. Scaling keeps
    # u < total strictly (uniform() < 1), so every draw resolves inside the real
    # cumulative range and a zero-probability entry (which adds nothing to the
    # sum) can never be selected. The final `return` is then only a defensive
    # backstop for floating-point rounding at the very top.
    var n = len(probs)
    if n == 0:
        raise Error("sample_categorical: empty distribution")

    var total = 0.0
    for i in range(n):
        if probs[i] < 0.0:
            raise Error(
                "sample_categorical: probabilities must be non-negative, got "
                + String(probs[i])
            )
        total += probs[i]
    var drift = total - 1.0
    if drift < 0.0:
        drift = -drift
    if drift > 1e-6:
        raise Error(
            "sample_categorical: probabilities must sum to 1, got "
            + String(total)
        )

    var threshold = rng.uniform() * total
    var cumulative = 0.0
    for i in range(n):
        cumulative += probs[i]
        if threshold < cumulative:
            return i
    return n - 1
