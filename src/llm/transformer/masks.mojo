"""Additive [T_q, T_k] attention masks folded into the pre-softmax scores.

A mask is data, not behavior: attention adds it to the scores before softmax.
0.0 leaves a score untouched ("attend"); MASKED_SCORE drives that key's
post-softmax weight to ~0 ("blocked"). Masks are additive, so they compose by
plain tensor addition, and causal / padding masks stay separate builders that a
caller sums as needed.
"""

from llm.tensor.tensor2d import Tensor2D, zeros_2d

# The additive "blocked" score. Finite (not -inf) on purpose: a fully-blocked
# query row would make the stable softmax compute exp(NaN) = NaN under -inf,
# whereas a large finite value keeps the row finite with weights summing to 1.
# -1e9 also dwarfs any real logit, so a partially-blocked row drives every masked
# key to ~0 weight; the exact magnitude is not load-bearing.
comptime MASKED_SCORE = -1e9


def no_mask(t_q: Int, t_k: Int) -> Tensor2D:
    """Build the identity mask: [t_q, t_k] all zeros, "attend to everything".

    The neutral element of mask composition. Allocates; cannot fail.

    Args:
        t_q: Query length.
        t_k: Key length.

    Returns:
        A [t_q, t_k] zero mask.
    """
    return zeros_2d(t_q, t_k)


def causal_mask(t: Int) -> Tensor2D:
    """Build the [t, t] autoregressive mask.

    0.0 on and below the diagonal, MASKED_SCORE strictly above, so query i
    attends only to keys j <= i (itself and the past), never the future.

    Args:
        t: Sequence length.

    Returns:
        A [t, t] causal mask. Allocates; cannot fail.
    """
    var m = zeros_2d(t, t)
    for i in range(t):
        for j in range(i + 1, t):  # strictly above the diagonal
            m[i, j] = MASKED_SCORE
    return m^


def key_padding_mask(keep: List[Bool], t_q: Int) -> Tensor2D:
    """Build a [t_q, len(keep)] mask blocking padded key positions.

    Column j is MASKED_SCORE for every query row when keep[j] is False, else
    0.0 — padding is a property of the key, so the mask is the same per query.

    Args:
        keep: Per-key-position flag; False marks padding to block.
        t_q: Query length.

    Returns:
        A [t_q, len(keep)] mask. Allocates; cannot fail.
    """
    var t_k = len(keep)
    var m = zeros_2d(t_q, t_k)
    for j in range(t_k):
        if not keep[j]:
            for i in range(t_q):
                m[i, j] = MASKED_SCORE
    return m^
