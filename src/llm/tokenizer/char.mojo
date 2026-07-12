"""Codepoint-level character tokenizer.

The simplest tokenizer: the vocabulary is the set of unique Unicode codepoints
in a training corpus, ordered by codepoint value, so one codepoint is one token
with no subword structure. Sorting the vocab by codepoint value makes the ids
deterministic — the same corpus always yields the same ids, with no seed or
hash-order dependence.
"""

# Save-format tag (see save/load). Stored codepoints are decimal integers, which
# avoids every newline/whitespace-escaping bug a character-literal format invites.
comptime CHARTOK_MAGIC = "CHARTOK v1"


def _first_codepoint(single: String) raises -> Int:
    """Return the codepoint value of a one-character string.

    Args:
        single: A one-character string.

    Returns:
        Its codepoint value.

    Raises:
        Error: If the string is empty.
    """
    for cp in single.codepoints():
        return Int(cp)
    raise Error("empty string has no codepoint")


struct CharTokenizer(Copyable, Movable):
    """Character tokenizer: an id <-> single-codepoint-string table."""

    var itos: List[String]  # id -> single-codepoint string, sorted by codepoint
    var stoi: Dict[String, Int]  # single-codepoint string -> id

    def __init__(out self):
        """Initialize an empty vocab, populated by from_text or load."""
        self.itos = []
        self.stoi = {}

    @staticmethod
    def from_text(text: String) raises -> CharTokenizer:
        """Build a vocab from the unique codepoints of `text`.

        Codepoints are ordered by value so the ids are deterministic.

        Args:
            text: The training corpus.

        Returns:
            The built tokenizer. Allocates; raises only on internal invariants,
            never on ordinary text.
        """
        var codepoints: List[Int] = []
        for cp in text.codepoints():
            codepoints.append(Int(cp))
        sort(codepoints)

        var tok = CharTokenizer()
        var previous = -1
        for i in range(len(codepoints)):
            var value = codepoints[i]
            if value == previous:
                continue  # skip duplicates; the list is sorted
            previous = value
            var symbol = String(Codepoint.from_u32(UInt32(value)).value())
            tok.stoi[symbol] = len(tok.itos)
            tok.itos.append(symbol)
        return tok^

    def vocab_size(self) -> Int:
        return len(self.itos)

    def encode(self, text: String) raises -> List[Int]:
        """Map each codepoint of `text` to its id.

        Args:
            text: The string to encode.

        Returns:
            The token ids. Allocates.

        Raises:
            Error: If any character is not in the vocab.
        """
        var ids: List[Int] = []
        for cp_slice in text.codepoint_slices():
            var symbol = String(cp_slice)
            if symbol not in self.stoi:
                raise Error(
                    "CharTokenizer.encode: character not in vocab: " + symbol
                )
            ids.append(self.stoi[symbol])
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Map ids back to characters.

        Args:
            ids: Token ids to decode.

        Returns:
            The decoded string. Allocates.

        Raises:
            Error: If any id is outside 0 .. vocab_size-1.
        """
        var out = String("")
        for i in range(len(ids)):
            var id = ids[i]
            if id < 0 or id >= len(self.itos):
                raise Error(
                    "CharTokenizer.decode: id out of range: " + String(id)
                )
            out += self.itos[id]
        return out^

    def save(self, path: String) raises:
        """Write the vocab as CHARTOK v1.

        Format: magic line, vocab size, then one decimal codepoint per line in
        id order.

        Args:
            path: Destination file path (overwritten).
        """
        var text = (
            String(CHARTOK_MAGIC) + "\n" + String(self.vocab_size()) + "\n"
        )
        for i in range(len(self.itos)):
            text += String(_first_codepoint(self.itos[i])) + "\n"
        with open(path, "w") as f:
            f.write(text)

    @staticmethod
    def load(path: String) raises -> CharTokenizer:
        """Restore a tokenizer written by save.

        Args:
            path: File to read.

        Returns:
            The restored tokenizer. Allocates.

        Raises:
            Error: On a bad magic line, a size/line-count mismatch, or an
                unparseable codepoint.
        """
        var content = open(path, "r").read()
        var lines = content.split("\n")
        if len(lines) < 2 or String(lines[0]) != CHARTOK_MAGIC:
            raise Error(
                "CharTokenizer.load: bad magic line (expected '"
                + String(CHARTOK_MAGIC)
                + "')"
            )
        var declared = Int(String(lines[1]))
        var tok = CharTokenizer()
        for i in range(declared):
            var line_index = 2 + i
            if line_index >= len(lines):
                raise Error(
                    "CharTokenizer.load: truncated file (fewer codepoints than"
                    " declared)"
                )
            var value = Int(String(lines[line_index]))
            var codepoint = Codepoint.from_u32(UInt32(value))
            if not codepoint:
                raise Error(
                    "CharTokenizer.load: not a valid codepoint: "
                    + String(value)
                )
            var symbol = String(codepoint.value())
            tok.stoi[symbol] = len(tok.itos)
            tok.itos.append(symbol)
        return tok^
