# Tests for the codepoint-level CharTokenizer.
#
# The char tokenizer is the simplest tokenizer in the pyramid: its vocab is the
# sorted set of unique Unicode codepoints in a corpus. These tests lock the
# invariants a reader relies on — deterministic ids, exact round trips (including
# non-ASCII and non-BMP codepoints), clear errors, and an exact save/load cycle.

from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from std.tempfile import gettempdir

from llm.tokenizer import CharTokenizer


def _tmp_path(name: String) raises -> String:
    var d = gettempdir()
    if not d:
        raise Error("no temp directory available")
    return d.value() + "/" + name


def test_round_trip_ascii() raises:
    var text = String("Hello, World! 123")
    var tok = CharTokenizer.from_text(text)
    var ids = tok.encode(text)
    assert_equal(tok.decode(ids), text)


def test_round_trip_unicode() raises:
    # ä (2-byte), € (3-byte), 🎉 (4-byte, non-BMP) all round-trip exactly.
    var text = String("näïve € 🎉 café")
    var tok = CharTokenizer.from_text(text)
    var ids = tok.encode(text)
    assert_equal(tok.decode(ids), text)


def test_deterministic_ids() raises:
    # Same corpus built twice -> identical vocab and identical encodings; the
    # vocab is ordered by codepoint value, so ids never depend on a seed.
    var text = String("cba cba")
    var tok_a = CharTokenizer.from_text(text)
    var tok_b = CharTokenizer.from_text(text)
    var ids_a = tok_a.encode(text)
    var ids_b = tok_b.encode(text)
    assert_equal(len(ids_a), len(ids_b))
    for i in range(len(ids_a)):
        assert_equal(ids_a[i], ids_b[i])
    # Codepoints present: space(32), a(97), b(98), c(99) -> ids 0..3 in order.
    # So "c" is the largest id and " " (space) is id 0.
    var space_id = tok_a.encode(String(" "))[0]
    var a_id = tok_a.encode(String("a"))[0]
    var c_id = tok_a.encode(String("c"))[0]
    assert_equal(space_id, 0)
    assert_true(a_id < c_id)


def test_vocab_size() raises:
    # "aabbc" has 3 unique codepoints (a, b, c).
    var tok = CharTokenizer.from_text(String("aabbc"))
    assert_equal(tok.vocab_size(), 3)


def test_unknown_char_raises() raises:
    var tok = CharTokenizer.from_text(String("abc"))
    with assert_raises():
        _ = tok.encode(String("z"))  # 'z' was never in the corpus


def test_bad_id_raises() raises:
    var tok = CharTokenizer.from_text(String("abc"))  # vocab_size == 3
    with assert_raises():
        _ = tok.decode([3])  # id 3 is out of range (valid ids are 0..2)


def test_save_load_round_trip() raises:
    var text = String("the quick brown fox — äöü")
    var tok = CharTokenizer.from_text(text)
    var path = _tmp_path("char_tok_roundtrip.chartok")
    tok.save(path)
    var loaded = CharTokenizer.load(path)
    assert_equal(loaded.vocab_size(), tok.vocab_size())
    # Encodings agree id-for-id after a save/load cycle.
    var ids_before = tok.encode(text)
    var ids_after = loaded.encode(text)
    assert_equal(len(ids_before), len(ids_after))
    for i in range(len(ids_before)):
        assert_equal(ids_before[i], ids_after[i])
    assert_equal(loaded.decode(ids_after), text)


def test_load_bad_magic_raises() raises:
    var path = _tmp_path("char_tok_badmagic.chartok")
    with open(path, "w") as f:
        f.write(String("NOTATOKENIZER v1\n0\n"))
    with assert_raises():
        _ = CharTokenizer.load(path)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
