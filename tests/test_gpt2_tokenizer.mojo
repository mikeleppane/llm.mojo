# Tests for the GPT-2 tokenizer — the capstone of Part V.
#
# The decisive test is parity: for a fixed set of sample strings, our encoder
# must produce byte-exact the same token ids as an independent reference (the
# vendored OpenAI encoder.py in tests/oracles/). We also freeze a few golden id
# sequences so the suite still catches a silently broken oracle and survives
# offline. Pre-tokenization is checked separately so a failure localizes to the
# regex split versus the merge loop.

from std.python import Python, PythonObject
from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from std.tempfile import gettempdir

from llm.tokenizer import (
    GPT2Tokenizer,
    BPETokenizer,
    gpt2_byte_to_unicode,
    gpt2_pre_tokenize,
    GPT2_VOCAB_SIZE,
    END_OF_TEXT_ID,
)

comptime VOCAB_PATH = "data/gpt2/vocab.json"
comptime MERGES_PATH = "data/gpt2/merges.txt"


def _tmp_path(name: String) raises -> String:
    var d = gettempdir()
    if not d:
        raise Error("no temp directory available")
    return d.value() + "/" + name


def _oracle() raises -> PythonObject:
    # The vendored reference encoder (tests/oracles/gpt2_reference_encoder.py).
    Python.add_to_path("tests/oracles")
    return Python.import_module("gpt2_reference_encoder")


def _to_ids(py_list: PythonObject) raises -> List[Int]:
    var out: List[Int] = []
    for value in py_list:
        out.append(Int(py=value))
    return out^


def _load() raises -> GPT2Tokenizer:
    return GPT2Tokenizer.from_files(String(VOCAB_PATH), String(MERGES_PATH))


def _samples() -> List[String]:
    # Plain ASCII, contractions, leading/trailing/interior whitespace, numbers,
    # tabs and newlines, an emoji (non-BMP), accented text, a long word, the
    # empty string. Chosen to exercise every branch of the pre-tokenizer.
    var samples: List[String] = [
        String(""),
        String("Hello world"),
        String("Hello, World!"),
        String("don't you think?"),
        String("  leading spaces"),
        String("trailing spaces   "),
        String("tabs\tand\nnewlines"),
        String("123 456 7890"),
        String("café résumé naïve"),
        String("The quick brown fox jumps over the lazy dog."),
        String("🎉 emoji party 🎈"),
        String("supercalifragilisticexpialidocious"),
        String("GPT-2 uses byte-level BPE."),
        String("Mixed CASE and numb3rs!"),
        String("a"),
    ]
    return samples^


def _assert_ids_equal(got: List[Int], expected: List[Int]) raises:
    assert_equal(len(got), len(expected))
    for i in range(len(expected)):
        assert_equal(got[i], expected[i])


def test_vocab_size_is_50257() raises:
    var tok = _load()
    assert_equal(tok.vocab_size(), 50257)
    assert_equal(GPT2_VOCAB_SIZE, 50257)


def test_byte_unicode_table_is_bijection() raises:
    # 256 entries, all distinct codepoints; the inverse composes to identity.
    var table = gpt2_byte_to_unicode()
    assert_equal(len(table), 256)
    var seen = Dict[Int, Int]()
    for b in range(256):
        var cp = table[b]
        assert_true(
            cp not in seen, "byte->unicode table has a duplicate codepoint"
        )
        seen[cp] = b
    # Inverse(table[b]) == b for every byte.
    for b in range(256):
        assert_equal(seen[table[b]], b)


def test_end_of_text_id() raises:
    # <|endoftext|> is id 50256. It is not special-cased in encode (matching
    # OpenAI's encoder.py); it is the last vocabulary entry, and decoding it
    # yields the literal marker text.
    assert_equal(END_OF_TEXT_ID, 50256)
    var tok = _load()
    assert_equal(tok.decode([END_OF_TEXT_ID]), String("<|endoftext|>"))


def test_pre_tokenize_matches_oracle() raises:
    var tok = _load()
    var oracle = _oracle()
    for s in _samples():
        var ours = gpt2_pre_tokenize(tok.pattern, s)
        var theirs = oracle.reference_pre_tokenize(s)
        assert_equal(
            len(ours), Int(len(theirs)), "chunk count differs for: " + s
        )
        for i in range(len(ours)):
            assert_equal(ours[i], String(theirs[i]))


def test_parity_with_reference_oracle() raises:
    var tok = _load()
    var oracle = _oracle()
    for s in _samples():
        var ours = tok.encode(s)
        var theirs = _to_ids(oracle.reference_encode(s))
        _assert_ids_equal(ours, theirs)


def test_round_trip() raises:
    var tok = _load()
    for s in _samples():
        assert_equal(tok.decode(tok.encode(s)), s)


def test_golden_ids() raises:
    # Frozen from the reference oracle during implementation (never from memory).
    # Guards against a broken oracle and survives fully offline.
    var tok = _load()
    _assert_ids_equal(tok.encode(String("Hello world")), [15496, 995])
    _assert_ids_equal(tok.encode(String("Hello, World!")), [15496, 11, 2159, 0])
    _assert_ids_equal(
        tok.encode(String("GPT-2 uses byte-level BPE.")),
        [38, 11571, 12, 17, 3544, 18022, 12, 5715, 347, 11401, 13],
    )


def test_save_load_round_trip() raises:
    # The GPT-2 merges/vocab persist through our BPETOK v1 format: after a save
    # and load, the underlying BPE encodes every pre-tokenized chunk identically.
    var tok = _load()
    var oracle = _oracle()
    var path = _tmp_path("gpt2_bpe_roundtrip.bpetok")
    tok.bpe.save(path)
    var loaded = BPETokenizer.load(path)
    for s in _samples():
        var chunks = oracle.reference_pre_tokenize(s)
        for chunk in chunks:
            var chunk_str = String(chunk)
            var view = chunk_str.as_bytes()
            var raw: List[UInt8] = []
            for i in range(len(view)):
                raw.append(view[i])
            _assert_ids_equal(
                loaded.encode_bytes(raw), tok.bpe.encode_bytes(raw)
            )


def test_loader_rejects_wrong_vocab_size() raises:
    # A vocab that is not exactly 50257 entries must be rejected, not silently
    # loaded. Build a tiny fixture rather than committing a broken file.
    var vpath = _tmp_path("bad_vocab.json")
    with open(vpath, "w") as f:
        f.write(String('{"a": 0, "b": 1}'))
    var mpath = _tmp_path("bad_merges.txt")
    with open(mpath, "w") as f:
        f.write(String("#version: 0.2\n"))
    with assert_raises():
        _ = GPT2Tokenizer.from_files(vpath, mpath)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
