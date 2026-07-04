# Toy sequence tasks for the encoder-decoder lab: copy and reverse.
#
# Both tasks map a source sequence of `T` symbols to a target sequence of the
# same length over the same alphabet. Copy is the warmup — the target IS the
# source, a near-diagonal alignment a decoder can almost fake. Reverse is the
# proof — target position i is source position T-1-i, a position-crossing
# alignment that only cross-attention can learn. A trained model either emits the
# exact right sequence or it does not, so success is exact-match, no perplexity
# judgment calls.
#
# The vocabulary is V_data data symbols (ids 0..V_data-1) plus one BOS id. BOS is
# V_data itself — one past the last data symbol — so it can never collide with a
# real symbol. The decoder is teacher-forced: its input is [BOS] + tgt[:-1], the
# target shifted right by one, so predicting position i sees the true tokens
# 0..i-1 and never its own answer.

from llm.utils.random import Rng


@fieldwise_init
struct SeqPair(Copyable, Movable):
    # One training example: a source sequence and its task target, both length T
    # over the data alphabet [0, V_data). The teacher-forcing decoder input is
    # derived from `tgt` on demand by `decoder_input`, not stored — it is a view
    # of the same data shifted, and storing it would let the two drift.
    var src: List[Int]  # [T]
    var tgt: List[Int]  # [T]


def bos_id(v_data: Int) -> Int:
    # The BOS token id for a data alphabet of size V_data: V_data itself, the id
    # one past the last data symbol so it never collides with a real symbol. The
    # full model vocab is then V = V_data + 1. Pure; allocates nothing; cannot
    # fail (a non-positive V_data is a caller error caught where sequences are
    # drawn, not here).
    return v_data


def random_source(mut rng: Rng, v_data: Int, t: Int) raises -> List[Int]:
    # A fresh source sequence: `t` ids each drawn uniformly from [0, V_data).
    # Mutates rng (advances its state, one draw per position); allocates the
    # result; deterministic given the generator's state. Raises if v_data <= 0 or
    # t <= 0 (a degenerate alphabet or length has no valid draw).
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
    # The copy task's target: an independent copy of the source. Reads src;
    # allocates the result; cannot fail.
    var out = List[Int]()
    for i in range(len(src)):
        out.append(src[i])
    return out^


def reverse_target(src: List[Int]) -> List[Int]:
    # The reverse task's target: the source read back to front, so target
    # position i holds source position len(src)-1-i. Reads src; allocates the
    # result; cannot fail.
    var out = List[Int]()
    for i in range(len(src) - 1, -1, -1):
        out.append(src[i])
    return out^


def make_pair(src: List[Int], reverse: Bool) -> SeqPair:
    # Build a SeqPair from a source: reverse=True gives the reverse task, else
    # copy. Reads src; allocates the pair; cannot fail.
    if reverse:
        return SeqPair(copy_target(src), reverse_target(src))
    return SeqPair(copy_target(src), copy_target(src))


def decoder_input(tgt: List[Int], bos: Int) raises -> List[Int]:
    # The teacher-forcing decoder input: [BOS] + tgt[:-1], the target shifted
    # right by one so position i is fed the true token i-1 (BOS at position 0).
    # The result has the same length as tgt: one BOS prepended, the last target
    # token dropped (it is only ever a label, never an input). Reads tgt;
    # allocates the result; raises on an empty target (no room for the shift).
    var t = len(tgt)
    if t == 0:
        raise Error("decoder_input: target is empty, no shift is defined")
    var out = List[Int]()
    out.append(bos)
    for i in range(t - 1):
        out.append(tgt[i])
    return out^


def sequences_equal(a: List[Int], b: List[Int]) -> Bool:
    # True iff two integer sequences are elementwise equal (same length, same
    # entries). The exact-match success criterion and the uniqueness dedup below
    # both lean on it. Reads its args; allocates nothing; cannot fail.
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def unique_sources(
    mut rng: Rng, v_data: Int, t: Int, count: Int
) raises -> List[List[Int]]:
    # `count` DISTINCT source sequences, drawn in order and de-duplicated by a
    # linear scan against those already kept. Distinctness is what lets a caller
    # split the result into a train set and a truly held-out set: a held-out
    # source the model never saw during training is the evidence that rules out
    # memorization. Mutates rng; allocates the result; deterministic given the
    # generator's state. Raises on non-positive count (via the loop guard) or a
    # degenerate alphabet/length (via random_source). At V_data^T >> count the
    # rejection loop almost never fires, but it makes the guarantee exact rather
    # than probabilistic.
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
