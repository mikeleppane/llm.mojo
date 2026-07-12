"""Toy sequence tasks for the encoder-decoder lab: copy and reverse.

Both map a source sequence of T symbols to a same-length target over the same
alphabet. Copy is the warmup (target IS the source, a near-diagonal alignment).
Reverse is the proof (target position i is source position T-1-i, a
position-crossing alignment only cross-attention can learn). Success is
exact-match.

The vocabulary is V_data data symbols (ids 0..V_data-1) plus one BOS id equal to
V_data, one past the last symbol so it never collides with a real one. The
decoder is teacher-forced on [BOS] + tgt[:-1].
"""

from llm.utils.random import Rng


@fieldwise_init
struct SeqPair(Copyable, Movable):
    """One training example: a source sequence and its task target.

    Both length T over the data alphabet [0, V_data). The teacher-forcing decoder
    input is derived from tgt on demand by decoder_input, not stored, so the two
    cannot drift.
    """

    var src: List[Int]  # [T]
    var tgt: List[Int]  # [T]


def bos_id(v_data: Int) -> Int:
    """Return the BOS token id for a data alphabet of size V_data.

    It is V_data itself, one past the last data symbol, so it never collides with
    a real symbol; the full model vocab is then V = V_data + 1.

    Args:
        v_data: Data alphabet size.

    Returns:
        The BOS token id.
    """
    return v_data


def random_source(mut rng: Rng, v_data: Int, t: Int) raises -> List[Int]:
    """Draw a fresh source sequence of t ids, each uniform on [0, V_data).

    Deterministic given the generator's state.

    Args:
        rng: Random generator; its state is advanced one draw per position.
        v_data: Data alphabet size; must be positive.
        t: Sequence length; must be positive.

    Returns:
        The source ids, length t. Allocates.

    Raises:
        Error: If v_data <= 0 or t <= 0.
    """
    if v_data <= 0:
        raise Error(
            "random_source: v_data must be positive, got " + String(v_data)
        )
    if t <= 0:
        raise Error("random_source: t must be positive, got " + String(t))
    var out = List[Int]()
    for _ in range(t):
        out.append(rng.next_below(v_data))
    return out^


def copy_target(src: List[Int]) -> List[Int]:
    """Build the copy task's target: an independent copy of the source.

    Args:
        src: Source ids.

    Returns:
        A copy of src. Allocates.
    """
    var out = List[Int]()
    for i in range(len(src)):
        out.append(src[i])
    return out^


def reverse_target(src: List[Int]) -> List[Int]:
    """Build the reverse task's target: the source read back to front.

    Target position i holds source position len(src)-1-i.

    Args:
        src: Source ids.

    Returns:
        The reversed source. Allocates.
    """
    var out = List[Int]()
    for i in range(len(src) - 1, -1, -1):
        out.append(src[i])
    return out^


def make_pair(src: List[Int], reverse: Bool) -> SeqPair:
    """Build a SeqPair from a source.

    Args:
        src: Source ids.
        reverse: True gives the reverse task, else copy.

    Returns:
        The training pair. Allocates.
    """
    if reverse:
        return SeqPair(copy_target(src), reverse_target(src))
    return SeqPair(copy_target(src), copy_target(src))


def decoder_input(tgt: List[Int], bos: Int) raises -> List[Int]:
    """Build the teacher-forcing decoder input: [BOS] + tgt[:-1].

    The target shifted right by one so position i is fed the true token i-1 (BOS
    at position 0). Same length as tgt: one BOS prepended, the last target token
    dropped (it is only ever a label, never an input).

    Args:
        tgt: True target tokens.
        bos: Beginning-of-sequence token id.

    Returns:
        The decoder input, same length as tgt. Allocates.

    Raises:
        Error: On an empty target (no room for the shift).
    """
    var t = len(tgt)
    if t == 0:
        raise Error("decoder_input: target is empty, no shift is defined")
    var out = List[Int]()
    out.append(bos)
    for i in range(t - 1):
        out.append(tgt[i])
    return out^


def sequences_equal(a: List[Int], b: List[Int]) -> Bool:
    """Return True iff two integer sequences are elementwise equal.

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        True iff same length and same entries.
    """
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def unique_sources(
    mut rng: Rng, v_data: Int, t: Int, count: Int
) raises -> List[List[Int]]:
    """Draw count distinct source sequences, de-duplicated by a linear scan.

    Distinctness lets a caller split the result into a train set and a truly
    held-out set, so a held-out source the model never saw rules out
    memorization. Deterministic given the generator's state.

    Args:
        rng: Random generator; its state is advanced.
        v_data: Data alphabet size; must be positive.
        t: Sequence length; must be positive.
        count: Number of distinct sequences; must be non-negative.

    Returns:
        The distinct source sequences, count of them. Allocates.

    Raises:
        Error: On negative count, or a degenerate alphabet/length (via
            random_source).
    """
    if count < 0:
        raise Error(
            "unique_sources: count must be non-negative, got " + String(count)
        )
    var out = List[List[Int]]()
    while len(out) < count:
        var candidate = random_source(rng, v_data, t)
        var seen = False
        for i in range(len(out)):
            if sequences_equal(out[i], candidate):
                seen = True
                break
        if not seen:
            out.append(candidate^)
    return out^
