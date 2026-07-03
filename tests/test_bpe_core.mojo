# Tests for the byte-level BPE core (BPETokenizer).
#
# These lock the two things a BPE implementation can get subtly wrong: the merge
# loop (encode) must always apply the *lowest-rank* mergeable pair, and the
# trainer must learn the same merges a human would by hand. The worked example
# "aaabdaaabac" is the canonical BPE walkthrough; we assert the exact merges and
# the exact encoding it produces, so the trainer and the merge loop are pinned to
# a reference a reader can reproduce with pencil and paper.

from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from std.tempfile import gettempdir

from llm.tokenizer import BPETokenizer, pair_key


def _tmp_path(name: String) raises -> String:
    var d = gettempdir()
    if not d:
        raise Error("no temp directory available")
    return d.value() + "/" + name


def _bytes(values: List[Int]) -> List[UInt8]:
    var out: List[UInt8] = []
    for i in range(len(values)):
        out.append(UInt8(values[i]))
    return out^


def _assert_ids_equal(got: List[Int], expected: List[Int]) raises:
    assert_equal(len(got), len(expected))
    for i in range(len(expected)):
        assert_equal(got[i], expected[i])


def test_pair_key_packs_losslessly() raises:
    # Distinct (left, right) pairs must map to distinct packed keys. A collision
    # would corrupt the merge dictionaries.
    assert_true(pair_key(97, 98) != pair_key(98, 97))
    assert_true(pair_key(256, 97) != pair_key(97, 256))
    assert_equal(pair_key(0, 0), 0)


def test_no_merges_is_identity_on_bytes() raises:
    # A fresh tokenizer has 256 byte tokens and no merges: one id per byte.
    var tok = BPETokenizer()
    assert_equal(tok.vocab_size(), 256)
    var ids = tok.encode_bytes(_bytes([104, 105]))  # "hi"
    _assert_ids_equal(ids, [104, 105])
    assert_equal(tok.decode(ids), String("hi"))


def test_train_hand_computed() raises:
    # The classic worked example. 3 merges (target 259) reproduce, in order:
    #   rank 0: (a, a) -> 256      "aa"
    #   rank 1: (a, b) -> 257      "ab"
    #   rank 2: (aa, ab) -> 258    "aaab"
    # This is Wikipedia's BPE walkthrough: aa->Z, ab->Y, ZY->X, leaving the
    # string "XdXac", i.e. [aaab, d, aaab, a, c].
    var tok = BPETokenizer()
    tok.train(String("aaabdaaabac"), 259)
    assert_equal(tok.vocab_size(), 259)

    # Each learned merge, verified black-box through encode_bytes.
    _assert_ids_equal(tok.encode_bytes(_bytes([97, 97])), [256])  # aa
    _assert_ids_equal(tok.encode_bytes(_bytes([97, 98])), [257])  # ab
    _assert_ids_equal(tok.encode_bytes(_bytes([97, 97, 97, 98])), [258])  # aaab

    # The full training string "aaabdaaabac" -> "XdXac". a=97, c=99, d=100.
    _assert_ids_equal(
        tok.encode(String("aaabdaaabac")), [258, 100, 258, 97, 99]
    )


def test_merge_respects_rank() raises:
    # Two overlapping merges compete on "abc": (a,b) at rank 0 and (b,c) at
    # rank 1. The loop must take the lower-rank pair first, giving [ab, c], not
    # [a, bc]. If rank were ignored the result would differ.
    var tok = BPETokenizer()
    var ab = tok.register_merge(97, 98)  # rank 0 -> id 256
    var bc = tok.register_merge(98, 99)  # rank 1 -> id 257
    assert_equal(ab, 256)
    assert_equal(bc, 257)
    _assert_ids_equal(tok.encode_bytes(_bytes([97, 98, 99])), [256, 99])


def test_round_trip_after_training() raises:
    var corpus = String(
        "the quick brown fox jumps over the lazy dog. "
        "the dog was not amused by the quick fox."
    )
    var tok = BPETokenizer()
    tok.train(corpus, 320)
    # Byte-level BPE can encode anything: both the training text and unseen text.
    assert_equal(tok.decode(tok.encode(corpus)), corpus)
    var unseen = String("a totally different sentence — with café!")
    assert_equal(tok.decode(tok.encode(unseen)), unseen)


def test_decode_invalid_utf8_replaces() raises:
    # Ids may split a multi-byte character; decode must not trap. 0xC3 is the
    # lead byte of "é" (0xC3 0xA9); on its own it is invalid UTF-8 and must
    # decode to the replacement character U+FFFD.
    var tok = BPETokenizer()
    var replacement = String(Codepoint.from_u32(UInt32(0xFFFD)).value())
    assert_equal(tok.decode([0xC3]), replacement)


def test_train_is_deterministic() raises:
    # Same corpus, trained twice, must learn identical merges (the tie-break is
    # fixed: highest count, then lowest pair key). Compare via encodings.
    var corpus = String("abracadabra abracadabra banana bandana")
    var tok_a = BPETokenizer()
    var tok_b = BPETokenizer()
    tok_a.train(corpus, 300)
    tok_b.train(corpus, 300)
    assert_equal(tok_a.vocab_size(), tok_b.vocab_size())
    _assert_ids_equal(tok_a.encode(corpus), tok_b.encode(corpus))


def test_save_load_round_trip() raises:
    var corpus = String("the quick brown fox jumps over the lazy dog")
    var tok = BPETokenizer()
    tok.train(corpus, 300)
    var path = _tmp_path("bpe_roundtrip.bpetok")
    tok.save(path)
    var loaded = BPETokenizer.load(path)
    assert_equal(loaded.vocab_size(), tok.vocab_size())
    # Bit-exact integer comparison of encodings after the round trip.
    _assert_ids_equal(loaded.encode(corpus), tok.encode(corpus))
    var probe = String("brown dog fox")
    _assert_ids_equal(loaded.encode(probe), tok.encode(probe))


def test_train_twice_extends_without_corruption() raises:
    # Training an already-trained tokenizer must extend the vocab, not re-learn
    # a pair it already knows: continuing from the current encoding keeps ranks
    # contiguous so save/load and encode stay consistent.
    var corpus = String("the quick brown fox jumps over the lazy dog. banana")
    var tok = BPETokenizer()
    tok.train(corpus, 280)
    tok.train(corpus, 300)
    assert_equal(tok.vocab_size(), 300)
    # Ranks stay 0..(n-1) and contiguous, so save/load round-trips.
    var path = _tmp_path("bpe_retrain.bpetok")
    tok.save(path)
    var loaded = BPETokenizer.load(path)
    assert_equal(loaded.vocab_size(), 300)
    _assert_ids_equal(loaded.encode(corpus), tok.encode(corpus))
    assert_equal(tok.decode(tok.encode(corpus)), corpus)


def test_register_duplicate_merge_raises() raises:
    var tok = BPETokenizer()
    _ = tok.register_merge(97, 98)
    with assert_raises():
        _ = tok.register_merge(97, 98)  # same pair twice must be rejected


def test_train_below_base_raises() raises:
    var tok = BPETokenizer()
    with assert_raises():
        tok.train(String("hello"), 100)  # < 256 base byte tokens


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
