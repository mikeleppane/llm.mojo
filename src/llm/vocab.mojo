"""A tiny whitespace vocabulary with an encode/decode round trip.

Splits on spaces and assigns each unseen word the next id, establishing the
round-trip contract `decode(encode(text)) == text` for space-separated input.
Unlike the real tokenizers under `tokenizer/`, this toy grows its vocabulary as
it encodes, which is why `encode` takes `mut self` — fine for a demonstration,
wrong for production.
"""

from std.collections import Dict, List


@fieldwise_init
struct Vocabulary(Copyable, Movable):
    """A growable token<->id map backed by a Dict and its inverse List."""

    var token_to_id: Dict[String, Int]  # token -> id
    var id_to_token: List[String]  # id -> token (id is the list index)

    def size(self) -> Int:
        """Number of tokens in the vocabulary."""
        return len(self.id_to_token)

    def add(mut self, token: String) raises -> Int:
        """Return the id for `token`, assigning the next free id if it is new.

        Idempotent: adding the same token twice returns the same id. Mutates self
        (may append to both maps).

        Raises:
            Error: Only because Dict subscript is a raising operation; a present
                token never fails.
        """
        if token in self.token_to_id:
            return self.token_to_id[token]
        var new_id = len(self.id_to_token)
        self.token_to_id[token] = new_id
        self.id_to_token.append(token)
        return new_id

    def id_of(self, token: String) raises -> Int:
        """Look up an existing token's id (read-only).

        Raises:
            Error: On an unknown token.
        """
        if token not in self.token_to_id:
            raise Error("unknown token: " + token)
        return self.token_to_id[token]

    def token_of(self, id: Int) raises -> String:
        """Inverse of id_of.

        Raises:
            Error: If the id is outside [0, size).
        """
        if id < 0 or id >= len(self.id_to_token):
            raise Error("id out of range")
        return self.id_to_token[id]

    def encode(mut self, text: String) raises -> List[Int]:
        """Split on single spaces, auto-adding unseen words.

        Mutates self (grows the vocabulary); allocates and returns the id list.

        Raises:
            Error: Via add (Dict subscript); never on well-formed input.
        """
        var ids = List[Int]()
        for word in text.split(" "):
            ids.append(self.add(String(word)))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Join tokens with single spaces.

        Raises:
            Error: Via token_of on an out-of-range id.
        """
        var text = String("")
        for i in range(len(ids)):
            if i > 0:
                text += " "
            text += self.token_of(ids[i])
        return text


def new_vocabulary() -> Vocabulary:
    """Return an empty vocabulary, hiding its two internal maps from callers."""
    var t2i: Dict[String, Int] = {}
    var i2t = List[String]()
    return Vocabulary(t2i^, i2t^)
