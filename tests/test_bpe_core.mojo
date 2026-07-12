"""Tests for the byte-level BPE core (BPETokenizer).

These lock the two things a BPE implementation gets subtly wrong: the merge loop
(encode) must always apply the lowest-rank mergeable pair, and the trainer must
learn the merges a human would by hand. The worked example "aaabdaaabac" is the
canonical BPE walkthrough; its exact merges and encoding are asserted.
"""

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
    """Distinct (left, right) pairs map to distinct packed keys; a collision would
    corrupt the merge dictionaries."""
    assert_true(pair_key(97, 98) != pair_key(98, 97))
    assert_true(pair_key(256, 97) != pair_key(97, 256))
    assert_equal(pair_key(0, 0), 0)


def test_no_merges_is_identity_on_bytes() raises:
    """A fresh tokenizer has 256 byte tokens and no merges: one id per byte."""
    var tok = BPETokenizer()
    assert_equal(tok.vocab_size(), 256)
    var ids = tok.encode_bytes(_bytes([104, 105]))  # "hi"
    _assert_ids_equal(ids, [104, 105])
    assert_equal(tok.decode(ids), String("hi"))


def test_train_hand_computed() raises:
    """The Wikipedia BPE walkthrough: 3 merges (a,a)->256, (a,b)->257, (aa,ab)->258
    leave "aaabdaaabac" as [258, 100, 258, 97, 99]."""
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
    """On "abc" with (a,b) rank 0 and (b,c) rank 1, the loop takes the lower-rank
    pair first, giving [ab, c] not [a, bc]."""
    var tok = BPETokenizer()
    var ab = tok.register_merge(97, 98)  # rank 0 -> id 256
    var bc = tok.register_merge(98, 99)  # rank 1 -> id 257
    assert_equal(ab, 256)
    assert_equal(bc, 257)
    _assert_ids_equal(tok.encode_bytes(_bytes([97, 98, 99])), [256, 99])


def test_round_trip_after_training() raises:
    """Byte-level BPE round-trips both the training text and unseen text exactly.
    """
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
    """Decode does not trap on invalid UTF-8: a lone 0xC3 lead byte becomes the
    replacement character U+FFFD."""
    var tok = BPETokenizer()
    var replacement = String(Codepoint.from_u32(UInt32(0xFFFD)).value())
    assert_equal(tok.decode([0xC3]), replacement)


def test_train_is_deterministic() raises:
    """Training the same corpus twice learns identical merges (tie-break: highest
    count, then lowest pair key)."""
    var corpus = String("abracadabra abracadabra banana bandana")
    var tok_a = BPETokenizer()
    var tok_b = BPETokenizer()
    tok_a.train(corpus, 300)
    tok_b.train(corpus, 300)
    assert_equal(tok_a.vocab_size(), tok_b.vocab_size())
    _assert_ids_equal(tok_a.encode(corpus), tok_b.encode(corpus))


def test_save_load_round_trip() raises:
    """A saved tokenizer reloads with identical vocab size and bit-exact encodings.
    """
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
    """Retraining extends the vocab without re-learning known pairs: ranks stay
    contiguous, so save/load and encode remain consistent."""
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
    """Registering the same pair twice is rejected."""
    var tok = BPETokenizer()
    _ = tok.register_merge(97, 98)
    with assert_raises():
        _ = tok.register_merge(97, 98)  # same pair twice must be rejected


def test_train_below_base_raises() raises:
    """A target vocab below the 256 base byte tokens is rejected."""
    var tok = BPETokenizer()
    with assert_raises():
        tok.train(String("hello"), 100)  # < 256 base byte tokens


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
