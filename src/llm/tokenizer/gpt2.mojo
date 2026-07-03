# GPT-2 tokenizer: byte-level BPE with GPT-2's exact vocabulary and merges.
#
# This layer adds the two things the generic BPE core (bpe.mojo) leaves out, so
# that encodings match OpenAI's GPT-2 byte-for-byte:
#   1. A pre-tokenizer regex that splits text into words/numbers/punctuation runs
#      before BPE ever sees it. GPT-2 uses a Unicode-property pattern the Python
#      stdlib `re` cannot express, so this one step calls the `regex` package
#      through Python interop — the one place this code reaches for Python.
#   2. A loader that reads GPT-2's own vocab.json / merges.txt. Those files are
#      written in GPT-2's byte<->unicode alphabet; the loader translates that
#      alphabet back to raw bytes/ids exactly once, here, so the merge loop in
#      bpe.mojo stays pure integer code.
#
# Everything after pre-tokenization is the Mojo BPE core; no Python touches the
# token data.

from std.python import Python, PythonObject

from .bpe import BPETokenizer, pair_key

comptime GPT2_VOCAB_SIZE = 50257
comptime END_OF_TEXT_ID = 50256

# GPT-2's pre-tokenizer pattern. Backslashes are doubled so the literal contains
# a single backslash for the regex engine. In order: the English contractions,
# then a run of letters / numbers / "other" (each optionally led by one space),
# then trailing whitespace. \p{L} and \p{N} are Unicode letter / number classes.
comptime GPT2_PATTERN = (
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+|"
    " ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
)


def gpt2_byte_to_unicode() -> List[Int]:
    # GPT-2's reversible byte -> unicode-codepoint map, as a 256-entry table
    # indexed by byte value. Printable byte values map to themselves; the rest
    # map to codepoints starting at 256, so every byte is a single printable
    # character. Pure Mojo; used only when loading. Allocates; does not raise.
    #
    # The printable ranges are ASCII '!'..'~' (33..126) plus Latin-1
    # '\xa1'..'\xac' (161..172) and '\xae'..'\xff' (174..255) — exactly the set
    # OpenAI's bytes_to_unicode() treats as already-printable.
    var table = List[Int](length=256, fill=-1)
    for b in range(33, 126 + 1):
        table[b] = b
    for b in range(161, 172 + 1):
        table[b] = b
    for b in range(174, 255 + 1):
        table[b] = b
    var n = 0
    for b in range(256):
        if table[b] == -1:
            table[b] = 256 + n
            n += 1
    return table^


# The byte<->unicode map is a fixed property of GPT-2's file format, known
# before the program runs, so build it once at compile time. The loop above is
# evaluated by the compiler and the 256-entry result is frozen into the binary.
# A List is not ImplicitlyCopyable, so a use site cannot read this comptime
# value directly (that would silently materialize a copy); from_files calls
# materialize[BYTE_TO_UNICODE]() to lift the compile-time table into a runtime
# List explicitly. The runtime function above stays as the readable definition
# and the tokenizer parity tests exercise it directly.
comptime BYTE_TO_UNICODE = gpt2_byte_to_unicode()


def gpt2_pre_tokenize(
    pattern: PythonObject, text: String
) raises -> List[String]:
    # The single Python-regex carve-out: run the compiled GPT-2 pattern over
    # `text` and return the matched chunks as Mojo strings. From here on, no
    # Python touches the data. Allocates; raises on any interop failure.
    var matches = pattern.findall(text)
    var chunks: List[String] = []
    for chunk in matches:
        chunks.append(String(chunk))
    return chunks^


def _token_to_bytes(
    token: String, inverse: Dict[Int, Int]
) raises -> List[UInt8]:
    # Translate a vocab token written in GPT-2's unicode alphabet back to the raw
    # bytes it stands for, via the inverted byte<->unicode table.
    var out: List[UInt8] = []
    for codepoint in token.codepoints():
        var code = Int(codepoint)
        var byte = inverse.get(code)
        if not byte:
            raise Error(
                "GPT2Tokenizer: token character is not in the byte<->unicode"
                " table (corrupt vocab)"
            )
        out.append(UInt8(byte.value()))
    return out^


struct GPT2Tokenizer(Movable):
    var bpe: BPETokenizer  # the byte-level BPE core, loaded with GPT-2's tables
    var pattern: PythonObject  # compiled pre-tokenizer regex, built once

    def __init__(out self, var bpe: BPETokenizer, var pattern: PythonObject):
        self.bpe = bpe^
        self.pattern = pattern^

    @staticmethod
    def from_files(
        vocab_path: String, merges_path: String
    ) raises -> GPT2Tokenizer:
        # Load GPT-2's vocab.json and merges.txt into a byte-level BPETokenizer.
        # vocab.json is parsed with Python json (plumbing only); every token is
        # translated from GPT-2's unicode alphabet into raw bytes here in Mojo.
        # Raises with a clear message if the vocab is not exactly 50257 entries,
        # a byte has no id, or a merge line does not resolve. Allocates.
        var json = Python.import_module("json")
        var builtins = Python.import_module("builtins")
        var regex = Python.import_module("regex")

        var handle = builtins.open(vocab_path, "r")
        var vocab_obj = json.load(handle)
        handle.close()

        var declared = Int(py=builtins.len(vocab_obj))
        if declared != GPT2_VOCAB_SIZE:
            raise Error(
                "GPT2Tokenizer.from_files: expected "
                + String(GPT2_VOCAB_SIZE)
                + " vocab entries, got "
                + String(declared)
            )

        # byte<->unicode table and its inverse (codepoint -> byte value). The
        # table is bound at compile time (see BYTE_TO_UNICODE); materialize lifts
        # that frozen value into a runtime List for the inverse-map build below.
        var byte_to_unicode = materialize[BYTE_TO_UNICODE]()
        var unicode_to_byte = Dict[Int, Int]()
        for b in range(256):
            unicode_to_byte[byte_to_unicode[b]] = b

        # Translate every vocab token: build id -> bytes (vocab) and the
        # token-string -> id map used to resolve merges.
        var vocab = List[List[UInt8]](length=GPT2_VOCAB_SIZE, fill=[])
        var stoi = Dict[String, Int]()
        for pair in vocab_obj.items():
            var token = String(pair[0])
            var id = Int(py=pair[1])
            if id < 0 or id >= GPT2_VOCAB_SIZE:
                raise Error(
                    "GPT2Tokenizer.from_files: token id out of range: "
                    + String(id)
                )
            vocab[id] = _token_to_bytes(token, unicode_to_byte)
            stoi[token] = id

        # Every id 0..50256 must have been assigned exactly once. A duplicate or
        # gap would leave a slot empty; catch it rather than encode garbage.
        for id in range(GPT2_VOCAB_SIZE):
            if len(vocab[id]) == 0:
                raise Error(
                    "GPT2Tokenizer.from_files: no token for id "
                    + String(id)
                    + " (duplicate or gap in vocab)"
                )

        # Base byte tokens: raw byte b -> id of its single-character token.
        var byte_to_id = List[Int]()
        for b in range(256):
            var symbol = String(
                Codepoint.from_u32(UInt32(byte_to_unicode[b])).value()
            )
            var id = stoi.get(symbol)
            if not id:
                raise Error(
                    "GPT2Tokenizer.from_files: no vocab id for byte "
                    + String(b)
                )
            byte_to_id.append(id.value())

        # Merges: each line "A B" (in rank order) resolves to ids by string
        # lookup; the merged token is the id of the concatenation "AB". Only the
        # leading "#version" header line is skipped — not every '#'-led line:
        # '#' is an ordinary token byte and several real merges (e.g. "# #")
        # begin with it. .strip() tolerates a stray CRLF on the right token.
        var merge_rank = Dict[Int, Int]()
        var merge_result = Dict[Int, Int]()
        var merges_content = open(merges_path, "r").read()
        var lines = merges_content.split("\n")
        var rank = 0
        for line_index in range(len(lines)):
            var line = String(String(lines[line_index]).strip())
            if line_index == 0 and line.startswith("#version"):
                continue
            if line.byte_length() == 0:
                continue
            var parts = line.split(" ")
            if len(parts) != 2:
                raise Error(
                    "GPT2Tokenizer.from_files: malformed merge line: " + line
                )
            var left_str = String(parts[0])
            var right_str = String(parts[1])
            var merged_str = left_str + right_str
            var left = stoi.get(left_str)
            var right = stoi.get(right_str)
            var merged = stoi.get(merged_str)
            if not left or not right or not merged:
                raise Error(
                    "GPT2Tokenizer.from_files: merge does not resolve: " + line
                )
            var key = pair_key(left.value(), right.value())
            merge_rank[key] = rank
            merge_result[key] = merged.value()
            rank += 1

        var bpe = BPETokenizer()
        bpe.vocab = vocab^
        bpe.byte_to_id = byte_to_id^
        bpe.merge_rank = merge_rank^
        bpe.merge_result = merge_result^

        var pattern = regex.compile(GPT2_PATTERN)
        return GPT2Tokenizer(bpe^, pattern^)

    def vocab_size(self) -> Int:
        return self.bpe.vocab_size()

    def encode(self, text: String) raises -> List[Int]:
        # Pre-tokenize with the GPT-2 regex, then run each chunk's UTF-8 bytes
        # through the BPE merge loop and concatenate. No special-token handling
        # (matches OpenAI's encoder.py). Allocates; raises on interop failure.
        var ids: List[Int] = []
        var chunks = gpt2_pre_tokenize(self.pattern, text)
        for chunk_index in range(len(chunks)):
            var view = chunks[chunk_index].as_bytes()
            var raw: List[UInt8] = []
            for i in range(len(view)):
                raw.append(view[i])
            var chunk_ids = self.bpe.encode_bytes(raw)
            for i in range(len(chunk_ids)):
                ids.append(chunk_ids[i])
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        # Bytes -> lossy UTF-8, delegated to the BPE core.
        return self.bpe.decode(ids)
