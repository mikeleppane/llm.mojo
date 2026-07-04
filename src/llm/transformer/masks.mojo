# Attention masks — additive [T_q, T_k] tensors folded into the pre-softmax
# scores.
#
# A mask is *data*, not behavior baked into attention: attention takes a mask
# argument and adds it to the scores before softmax. `0.0` leaves a score
# untouched ("attend"); MASKED_SCORE is a large negative addend that drives the
# post-softmax weight for that key to ~0 ("blocked"). Because masks are additive
# they compose by plain tensor addition: `add(causal_mask(t),
# key_padding_mask(keep, t))` blocks a cell whenever either mask blocks it. This
# is why causal and padding masks stay separate builders instead of one fused
# rule — the caller sums exactly the constraints it needs.

from llm.tensor.tensor2d import Tensor2D, zeros_2d

# The additive "blocked" score. Finite on purpose (not -inf): a fully-blocked
# query row — every key masked — would, with -inf, make the stable softmax
# compute exp(-inf - (-inf)) = exp(NaN) = NaN and poison everything downstream.
# With a large finite value the row instead stays finite and its weights still
# sum to 1 — the load-bearing guarantee. (Because softmax is shift-invariant,
# adding the same MASKED_SCORE to every key leaves the row equal to
# softmax(unmasked scores); that is uniform only when the underlying scores tie,
# and with composed masks — one cell MASKED_SCORE, another 2*MASKED_SCORE — the
# row can even lean toward the least-masked key. "Uniform" is the tied-score
# special case, not the general rule.) None of this matters in practice: a
# fully-blocked query is itself padding whose output is never read; only the
# finiteness is. -1e9 also dwarfs any real score (attention logits here are
# O(sqrt(d_head) * unit variance)), so a partially-blocked row drives every
# masked key to ~0 weight; the exact magnitude is not load-bearing.
comptime MASKED_SCORE = -1e9


def no_mask(t_q: Int, t_k: Int) -> Tensor2D:
    # The identity mask: [t_q, t_k] all zeros — "attend to everything". Adding it
    # changes no score. The neutral element of mask composition, and what
    # self-attention passes when there is nothing to hide. Allocates; cannot
    # fail.
    return zeros_2d(t_q, t_k)


def causal_mask(t: Int) -> Tensor2D:
    # [t, t]: 0.0 on and below the diagonal, MASKED_SCORE strictly above. Query
    # position i may attend to key positions j <= i (itself and the past) but not
    # to the future j > i — the constraint that makes a decoder autoregressive.
    # Allocates; cannot fail.
    var m = zeros_2d(t, t)
    for i in range(t):
        for j in range(i + 1, t):  # strictly above the diagonal
            m[i, j] = MASKED_SCORE
    return m^


def key_padding_mask(keep: List[Bool], t_q: Int) -> Tensor2D:
    # [t_q, len(keep)]: column j is MASKED_SCORE for every query row when
    # keep[j] is False, else 0.0. Blocks padded key positions so no query
    # attends to them — needed for padded batches and the encoder-decoder lab.
    # The mask is the same for every query row (padding is a property of the key,
    # not the query). Allocates; cannot fail.
    var t_k = len(keep)
    var m = zeros_2d(t_q, t_k)
    for j in range(t_k):
        if not keep[j]:
            for i in range(t_q):
                m[i, j] = MASKED_SCORE
    return m^
