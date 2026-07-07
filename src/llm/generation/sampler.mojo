# Sampling a token from a probability distribution.
#
# The one primitive generation needs before anything fancier: draw an index from
# a categorical distribution. It walks the cumulative distribution (inverse-CDF
# sampling) using a single uniform draw, so the same seeded Rng that shuffles
# batches also makes generation reproducible. Top-k / top-p and the real
# generate loop build on this later.

from llm.tensor.ops import argmax, softmax_row_temperature
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


# --- distribution filters -----------------------------------------------------
#
# Both filters live in PROBABILITY space: probs [V] -> probs [V], keeping a subset
# of tokens, zeroing the rest, and renormalizing so the result is again a valid
# distribution. This is the deliberate alternative to the "set filtered logits to
# -inf, softmax once" idiom: every intermediate stays a finite, valid distribution
# (no +/-inf ever enters the codebase), each filter is independently testable as
# distribution -> distribution against a NumPy-free Python oracle, and
# sample_categorical's own sum-to-1 guard re-validates the filters' output for
# free. The two forms are algebraically equivalent; this one keeps the arithmetic
# honest and legible.


def _order_by_prob_desc(probs: List[Float64]) -> List[Int]:
    # Return the indices 0..n-1 ordered by probability DESCENDING, ties broken by
    # ascending index (lower index first). This single ordering serves both
    # filters and encodes the one tie rule the whole part shares with argmax
    # (first-wins). Bottom-up merge sort: O(n log n) and stable, which both
    # matter at n = 50257 (the BPE vocab in the next part) where an O(n^2) scan
    # would be a real defect even though the n = 11 tests never feel it.
    # Allocates two index buffers; does not mutate probs; draws no rng.
    var n = len(probs)
    var a = List[Int]()
    for i in range(n):
        a.append(i)
    if n < 2:
        return a^

    # "i sorts before j" — strictly greater probability, or equal probability
    # (neither strictly greater) with the smaller index. Expressed without an
    # `==` on floats: if neither is strictly larger, they are equal and the index
    # decides. The order is total (indices are distinct), so no stability subtlety
    # remains — merging can take the "before" element unconditionally.
    var buf = List[Int]()
    for _ in range(n):
        buf.append(0)

    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = lo + width
            if mid > n:
                mid = n
            var hi = lo + 2 * width
            if hi > n:
                hi = n
            # Merge the two sorted runs a[lo:mid] and a[mid:hi] into buf[lo:hi].
            var i = lo
            var j = mid
            var out_pos = lo
            while i < mid and j < hi:
                var li = a[i]
                var rj = a[j]
                # li before rj iff probs[li] > probs[rj], or (tie) li < rj.
                var li_first = probs[li] > probs[rj] or (
                    not (probs[rj] > probs[li]) and li < rj
                )
                if li_first:
                    buf[out_pos] = li
                    i += 1
                else:
                    buf[out_pos] = rj
                    j += 1
                out_pos += 1
            while i < mid:
                buf[out_pos] = a[i]
                i += 1
                out_pos += 1
            while j < hi:
                buf[out_pos] = a[j]
                j += 1
                out_pos += 1
            lo += 2 * width
        # Copy the merged buffer back into a for the next pass.
        for k in range(n):
            a[k] = buf[k]
        width *= 2
    return a^


def _renormalize(probs: List[Float64]) raises -> List[Float64]:
    # Divide a non-negative vector by its sum so it sums to 1. Raises if the sum
    # is not positive (every kept mass zeroed out) — a filter that keeps at least
    # one token from a softmax distribution can never hit this, so it is a genuine
    # "the input was not a valid distribution" signal, not a normal path.
    # Allocates the result; does not mutate the input.
    var total = 0.0
    for i in range(len(probs)):
        total += probs[i]
    if total <= 0.0:
        raise Error(
            "filter: kept probability mass is not positive (got "
            + String(total)
            + "); input was not a valid distribution"
        )
    var out = List[Float64]()
    for i in range(len(probs)):
        out.append(probs[i] / total)
    return out^


def filter_top_k(probs: List[Float64], k: Int) raises -> List[Float64]:
    # Keep the k highest-probability entries, zero the rest, renormalize to sum 1.
    # Shapes: probs [V] -> [V]. Does not mutate probs; allocates the result; draws
    # NO rng. Raises on k < 0 or an empty input.
    #
    # Sentinels and edges: k == 0 is DISABLED (identity — the documented "off"
    # value used by SamplerConfig); k >= V is also the identity (nothing to drop);
    # k == 1 yields a one-hot at the argmax (greedy-by-filter). A tie at the k-th
    # boundary keeps the LOWER index (the shared first-wins rule), because
    # _order_by_prob_desc breaks equal probabilities toward the smaller index.
    var n = len(probs)
    if n == 0:
        raise Error("filter_top_k: empty distribution")
    if k < 0:
        raise Error("filter_top_k: k must be non-negative, got " + String(k))
    if k == 0 or k >= n:
        return probs.copy()  # disabled / nothing to drop -> identity

    var order = _order_by_prob_desc(probs)
    var kept = List[Float64]()
    for _ in range(n):
        kept.append(0.0)
    for rank in range(k):
        var idx = order[rank]
        kept[idx] = probs[idx]
    return _renormalize(kept)


def filter_top_p(probs: List[Float64], p: Float64) raises -> List[Float64]:
    # Nucleus filter: sort by probability descending (ties: lower index first),
    # keep the smallest prefix whose cumulative probability is >= p, zero the
    # rest, renormalize. Always keeps at least one token, so a tiny p collapses to
    # the argmax one-hot rather than an empty distribution.
    # Shapes: probs [V] -> [V]. Does not mutate probs; allocates the result; draws
    # NO rng. Raises on p <= 0, p > 1, or an empty input.
    #
    # p >= 1.0 is the DISABLED identity and is returned BEFORE any cumulative sum:
    # deciding "disabled" via `cumsum >= 1.0` would let a distribution whose mass
    # rounds to 0.999... silently drop its tail. The explicit short-circuit is the
    # honest gate.
    var n = len(probs)
    if n == 0:
        raise Error("filter_top_p: empty distribution")
    if p <= 0.0 or p > 1.0:
        raise Error("filter_top_p: p must be in (0, 1], got " + String(p))
    if p >= 1.0:
        return probs.copy()  # disabled -> identity, no cumulative-sum path

    var order = _order_by_prob_desc(probs)
    var kept = List[Float64]()
    for _ in range(n):
        kept.append(0.0)
    var cumulative = 0.0
    for rank in range(n):
        var idx = order[rank]
        kept[idx] = probs[idx]
        cumulative += probs[idx]
        if cumulative >= p:
            break  # smallest prefix reaching p; at least one token always kept
    return _renormalize(kept)


# --- the decoding policy: SamplerConfig + sample_next -------------------------


@fieldwise_init
struct SamplerConfig(Copyable, Movable):
    # One policy struct composing the four standard decoding knobs. Greedy is not
    # a separate mode but a POINT in this space: temperature 0.0 means argmax.
    var temperature: Float64  # 0.0 = greedy (argmax, no rng draw); else softmax T
    var top_k: Int  # 0 = disabled; else keep the k highest-probability tokens
    var top_p: Float64  # 1.0 = disabled; else nucleus threshold in (0, 1)

    @staticmethod
    def greedy() -> SamplerConfig:
        # Deterministic argmax decoding: (temperature 0, no filters).
        return SamplerConfig(0.0, 0, 1.0)

    @staticmethod
    def standard() -> SamplerConfig:
        # Plain temperature-1 sampling, no truncation: (1.0, disabled, disabled).
        return SamplerConfig(1.0, 0, 1.0)

    def validate(self) raises:
        # Reject an out-of-range field, naming it. temperature 0 is VALID (the
        # greedy sentinel); only a negative temperature is rejected. top_k must be
        # non-negative (0 disabled); top_p must be in (0, 1] (1.0 disabled). Reads
        # self; allocates nothing; draws no rng.
        if self.temperature < 0.0:
            raise Error(
                "SamplerConfig: temperature must be >= 0 (0 = greedy), got "
                + String(self.temperature)
            )
        if self.top_k < 0:
            raise Error(
                "SamplerConfig: top_k must be >= 0 (0 = disabled), got "
                + String(self.top_k)
            )
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise Error(
                "SamplerConfig: top_p must be in (0, 1] (1 = disabled), got "
                + String(self.top_p)
            )


def sample_next(
    logits: Span[Float64, _], cfg: SamplerConfig, mut rng: Rng
) raises -> Int:
    # Turn one logit row [V] into the next token id under the policy `cfg`. This is
    # the SINGLE decoding entry point. `logits` is a borrowed `Span` view — the
    # caller passes a Tensor2D row directly (no copy). Reads logits; allocates the
    # intermediate distributions; raises on an invalid config or (in the sampled
    # path) a degenerate distribution.
    #
    # rng-draw count — the invariant the tests pin:
    #   * greedy (temperature == 0.0): ZERO draws. It is pure argmax, so switching
    #     a run to greedy never perturbs any other seeded draw stream (the same
    #     "disabling randomness consumes no draws" invariant dropout established).
    #   * sampled (temperature > 0): EXACTLY ONE draw, consumed by
    #     sample_categorical at the end of the pipeline.
    #
    # Pipeline (sampled path), each stage renormalizing:
    #   softmax_row_temperature(logits, T)
    #     -> filter_top_k (if k > 0) -> filter_top_p (if p < 1) -> sample_categorical
    # softmax_row_temperature raises on T <= 0, so temperature 0.0 is an
    # unambiguous greedy sentinel with no valid sampled interpretation.
    cfg.validate()
    if cfg.temperature == 0.0:
        return argmax(logits)  # greedy: no rng draw

    var probs = softmax_row_temperature(logits, cfg.temperature)
    if cfg.top_k > 0:
        probs = filter_top_k(probs, cfg.top_k)
    if cfg.top_p < 1.0:
        probs = filter_top_p(probs, cfg.top_p)
    return sample_categorical(probs, rng)
