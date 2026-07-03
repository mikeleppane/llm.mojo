# Sampling a token from a probability distribution.
#
# The one primitive generation needs before anything fancier: draw an index from
# a categorical distribution. It walks the cumulative distribution (inverse-CDF
# sampling) using a single uniform draw, so the same seeded Rng that shuffles
# batches also makes generation reproducible. Top-k / top-p and the real
# generate loop build on this later.

from llm.utils.random import Rng


def sample_categorical(probs: List[Float64], mut rng: Rng) raises -> Int:
    # Draw an index i with probability probs[i] by inverse-CDF sampling: take
    # u ~ uniform[0, 1) and return the first i where the running cumulative sum
    # exceeds u. Raises on an empty distribution or one whose entries do not sum
    # to ~1 (tolerance 1e-6), catching an un-normalized input early. Mutates rng
    # (consumes one draw).
    #
    # The final `return` handles the u -> 1.0 boundary: floating-point rounding
    # can leave the cumulative sum a hair below u, so a draw in the top slice
    # falls through the loop and is clamped to the last index rather than
    # returning an out-of-range value. Zero-probability entries add nothing to
    # the cumulative sum, so they can never be selected.
    var n = len(probs)
    if n == 0:
        raise Error("sample_categorical: empty distribution")

    var total = 0.0
    for i in range(n):
        total += probs[i]
    var drift = total - 1.0
    if drift < 0.0:
        drift = -drift
    if drift > 1e-6:
        raise Error(
            "sample_categorical: probabilities must sum to 1, got "
            + String(total)
        )

    var u = rng.uniform()
    var cumulative = 0.0
    for i in range(n):
        cumulative += probs[i]
        if u < cumulative:
            return i
    return n - 1
