"""Reference sampling values for the Mojo generation tests.

Provenance, not a test-time dependency: run once by hand, its printed numbers
frozen as literals into tests/test_sampling_filters.mojo and
tests/test_sample_next.mojo. Independent Python math; nothing under src/ or the
suite imports this, and it derives nothing from the Mojo code.

Two kinds of golden:

1. Filter goldens -- top-k / top-p / composition distributions for fixed
   probability vectors, tie cases included. Each filter maps a distribution to a
   distribution (probability space, renormalized), matching the Mojo
   implementation's contract.
2. Exact sampled-token goldens -- replay the project's LCG (Knuth MMIX
   constants, the same recurrence Part VI's test_rng pins) plus the inverse-CDF
   walk of sample_categorical, so a fixed seed + fixed logit row predicts the
   EXACT token ids sample_next must emit. Sampling assertions are therefore
   integer-exact, never statistical.

Pure Python float is IEEE-754 double, the same as Mojo's Float64; softmax is
implemented the same stable way as tensor/ops.softmax_row_temperature (subtract
the row max, exp the difference, divide by the sum) so the oracle's buckets match
the Mojo pipeline's. Token ids are integers over wide buckets, so ulp-level
differences between the two exp implementations cannot flip a draw.

Run:  pixi run python tests/oracles/sampling_reference.py
"""

import math

# Knuth MMIX LCG constants (D. E. Knuth, TAOCP vol. 2) -- the same pair
# src/llm/utils/random.mojo uses.
A = 6364136223846793005
C = 1442695040888963407
MASK64 = (1 << 64) - 1
INV_2_POW_53 = 1.0 / float(1 << 53)


# --- the LCG, mirroring Rng exactly ---------------------------------------


def lcg_next(state):
    """state <- state*A + C (mod 2**64); returns the NEW state (== next_u64)."""
    return (state * A + C) & MASK64


def uniform(state):
    """Rng.uniform(): advance, take the top 53 bits as the mantissa. Returns
    (u in [0,1), new_state)."""
    state = lcg_next(state)
    bits = state >> 11
    return float(bits) * INV_2_POW_53, state


# --- the distribution pipeline, mirroring the Mojo contract ----------------


def softmax_temp(logits, T):
    """Stable temperature softmax: exp((x_i - max)/T) / sum. Matches
    tensor/ops.softmax_row_temperature. T must be > 0."""
    m = max(logits)
    exps = [math.exp((x - m) / T) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def filter_top_k(probs, k):
    """Keep the k highest-probability entries (ties: lower index wins), zero the
    rest, renormalize. k == 0 or k >= len is the identity."""
    n = len(probs)
    if k == 0 or k >= n:
        return list(probs)
    # Stable sort on (-prob, index): equal probs keep ascending index, so the
    # lower index wins the k-th boundary. Take the first k indices.
    order = sorted(range(n), key=lambda i: (-probs[i], i))
    keep = set(order[:k])
    out = [probs[i] if i in keep else 0.0 for i in range(n)]
    s = sum(out)
    return [x / s for x in out]


def filter_top_p(probs, p):
    """Sort by probability descending (ties: lower index first); keep the
    smallest prefix whose cumulative probability is >= p; zero the rest;
    renormalize. Always keeps at least one entry. p >= 1.0 is the identity,
    checked BEFORE any cumulative sum."""
    if p >= 1.0:
        return list(probs)
    n = len(probs)
    order = sorted(range(n), key=lambda i: (-probs[i], i))
    keep = []
    cumulative = 0.0
    for i in order:
        keep.append(i)
        cumulative += probs[i]
        if cumulative >= p:
            break  # smallest prefix reaching p; at least one entry always kept
    keepset = set(keep)
    out = [probs[i] if i in keepset else 0.0 for i in range(n)]
    s = sum(out)
    return [x / s for x in out]


def sample_categorical(probs, state):
    """Inverse-CDF walk with one uniform draw, matching
    generation/sampler.sample_categorical: threshold = u * total, return the
    first i whose running cumulative sum exceeds it. Returns (index, new_state)."""
    total = sum(probs)
    u, state = uniform(state)
    threshold = u * total
    cumulative = 0.0
    for i in range(len(probs)):
        cumulative += probs[i]
        if threshold < cumulative:
            return i, state
    return len(probs) - 1, state


def sample_next(logits, T, k, p, state):
    """The full sample_next pipeline for the sampled (T > 0) path:
    softmax_temp -> filter_top_k (if k>0) -> filter_top_p (if p<1) ->
    sample_categorical. Returns (id, new_state)."""
    probs = softmax_temp(logits, T)
    if k > 0:
        probs = filter_top_k(probs, k)
    if p < 1.0:
        probs = filter_top_p(probs, p)
    return sample_categorical(probs, state)


def fmt(xs):
    return "[" + ", ".join(repr(x) for x in xs) + "]"


def main():
    print("# ============================================================")
    print("# FILTER GOLDENS (freeze into test_sampling_filters.mojo)")
    print("# ============================================================")

    # Fixture A: an 8-entry distribution (already sums to 1). Two 0.05 entries
    # at indices 0 and 7 give a top-k boundary tie for the largest k.
    A8 = [0.05, 0.10, 0.30, 0.20, 0.15, 0.08, 0.07, 0.05]
    print(f"\n# Fixture A8 = {fmt(A8)}  (sum={sum(A8)!r})")
    for k in (1, 2, 3, 5):
        print(f"# top_k(A8, {k}) = {fmt(filter_top_k(A8, k))}")
    for p in (0.5, 0.9, 0.01):
        print(f"# top_p(A8, {p}) = {fmt(filter_top_p(A8, p))}")

    # Fixture T4: a boundary-tie distribution. 0.4 ties at idx 1,3; 0.1 ties at
    # idx 0,2. top_k=1 must pick idx 1 (lower index of the tied max); top_k=3
    # must break the 0.1 tie toward idx 0.
    T4 = [0.1, 0.4, 0.1, 0.4]
    print(f"\n# Fixture T4 = {fmt(T4)}  (sum={sum(T4)!r})")
    for k in (1, 2, 3):
        print(f"# top_k(T4, {k}) = {fmt(filter_top_k(T4, k))}")
    for p in (0.5, 0.8, 0.85):
        print(f"# top_p(T4, {p}) = {fmt(filter_top_p(T4, p))}")

    # Composition fixture: top_k THEN top_p differs from the swapped order.
    # Chosen so the pinned order keeps 2 tokens and the swapped order keeps 3.
    COMP = [0.5, 0.25, 0.15, 0.10]
    k, p = 3, 0.80
    pinned = filter_top_p(filter_top_k(COMP, k), p)  # top_k -> top_p (the order)
    swapped = filter_top_k(filter_top_p(COMP, p), k)  # top_p -> top_k
    print(f"\n# Composition fixture = {fmt(COMP)}  k={k} p={p}")
    print(f"# pinned  (top_k then top_p) = {fmt(pinned)}")
    print(f"# swapped (top_p then top_k) = {fmt(swapped)}")
    print(f"# orders differ: {pinned != swapped}")

    print("\n# ============================================================")
    print("# SAMPLED-TOKEN GOLDENS (freeze into test_sample_next.mojo)")
    print("# ============================================================")

    # Pure temperature sampling: fixed logit row, T=1.0, no filters, replay the
    # LCG from a fixed seed and predict the exact id sequence sample_next emits.
    logits = [1.0, 2.0, 0.5, -1.0, 0.0]
    T = 1.0
    probs = softmax_temp(logits, T)
    print(f"\n# logits = {fmt(logits)}  T={T}")
    print(f"# softmax = {fmt(probs)}")
    cdf = []
    acc = 0.0
    for x in probs:
        acc += x
        cdf.append(acc)
    print(f"# cdf     = {fmt(cdf)}")

    for seed in (42, 7, 0):
        state = seed
        ids = []
        # First draw's arithmetic worked out, for the test's hand-check comment.
        first_u, _ = uniform(seed)
        for _ in range(12):
            idx, state = sample_next(logits, T, 0, 1.0, state)
            ids.append(idx)
        print(f"\n# seed {seed}: first uniform u1 = {first_u!r}")
        print(f"#   threshold1 = u1*total = {first_u * sum(probs)!r}")
        print(f"#   sample_next x12 ids = {ids}")

    # top_k = 1 forces the argmax regardless of seed (filter one-hot -> the
    # categorical draw is deterministic). argmax of the logits is index 1.
    print("\n# top_k=1 forces argmax (index 1) for every seed:")
    for seed in (42, 7, 0, 123456):
        idx, _ = sample_next(logits, T, 1, 1.0, seed)
        print(f"#   seed {seed}: id = {idx}")

    # Hand inverse-CDF check (documented in the test): a clean 3-way
    # distribution and the first uniform from seed 42.
    print("\n# hand inverse-CDF check:")
    hand = [0.2, 0.5, 0.3]
    u1, _ = uniform(42)
    thr = u1 * sum(hand)
    hand_cdf = [0.2, 0.7, 1.0]
    hand_idx, _ = sample_categorical(hand, 42)
    print(f"#   probs={fmt(hand)} cdf={fmt(hand_cdf)}")
    print(f"#   seed 42 u1={u1!r} threshold={thr!r} -> id {hand_idx}")


if __name__ == "__main__":
    main()
