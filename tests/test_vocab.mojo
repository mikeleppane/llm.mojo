# Tests for the toy whitespace Vocabulary.
#
# The headline is the encode/decode round trip: decode(encode(text)) == text for
# space-separated input. The rest pin the small contracts — add is idempotent,
# id_of/token_of are inverses, unknown lookups raise, and repeated words reuse
# their id (the auto-add-once behavior).

from std.testing import assert_equal, assert_raises, TestSuite

from llm.vocab import Vocabulary, new_vocabulary


def test_add_is_idempotent() raises:
    var v = new_vocabulary()
    var a = v.add("hello")
    var b = v.add("hello")
    assert_equal(a, b)
    assert_equal(v.size(), 1)


def test_id_token_round_trip() raises:
    var v = new_vocabulary()
    var id = v.add("world")
    assert_equal(v.token_of(id), "world")
    assert_equal(v.id_of("world"), id)


def test_unknown_token_raises() raises:
    var v = new_vocabulary()
    with assert_raises(contains="unknown token"):
        _ = v.id_of("missing")


def test_token_of_out_of_range_raises() raises:
    var v = new_vocabulary()
    _ = v.add("only")
    with assert_raises(contains="out of range"):
        _ = v.token_of(5)


def test_encode_decode_round_trip() raises:
    var v = new_vocabulary()
    var ids = v.encode("the cat sat")
    assert_equal(len(ids), 3)
    assert_equal(v.decode(ids), "the cat sat")


def test_encode_reuses_ids_for_repeated_words() raises:
    var v = new_vocabulary()
    var first = v.encode("the cat")
    var second = v.encode("the dog")
    assert_equal(first[0], second[0])  # "the" keeps its id
    assert_equal(v.size(), 3)  # the, cat, dog


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
