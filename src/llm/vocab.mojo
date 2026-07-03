# A tiny whitespace vocabulary with an encode/decode round trip.
#
# This is the simplest thing that maps tokens to integer ids and back: split on
# spaces, assign each unseen word the next id. It exists to put structs, Dict,
# List, and error handling to work and to establish the round-trip contract
# `decode(encode(text)) == text` for space-separated input — the cheapest strong
# test in the project, reused later for tokenizers, transposes, and checkpoints.
#
# It deliberately coexists with the real tokenizers under `tokenizer/`: those
# freeze their vocabulary after training and handle unknown input by explicit
# byte/subword rules. This toy grows its vocabulary as it encodes, which is why
# `encode` takes `mut self` — fine for a demonstration, wrong for production.

from std.collections import Dict, List


@fieldwise_init
struct Vocabulary(Copyable, Movable):
    var token_to_id: Dict[String, Int]  # token -> id
    var id_to_token: List[String]  # id -> token (id is the list index)

    def size(self) -> Int:
        return len(self.id_to_token)

    def add(mut self, token: String) raises -> Int:
        # Return the id for `token`, assigning the next free id if it is new.
        # Idempotent: adding the same token twice returns the same id. Mutates
        # self (may append to both maps). Raises only because Dict subscript is a
        # raising operation on the pinned Mojo — a present token never fails.
        if token in self.token_to_id:
            return self.token_to_id[token]
        var new_id = len(self.id_to_token)
        self.token_to_id[token] = new_id
        self.id_to_token.append(token)
        return new_id

    def id_of(self, token: String) raises -> Int:
        # Look up an existing token. Raises on an unknown token (read-only).
        if token not in self.token_to_id:
            raise Error("unknown token: " + token)
        return self.token_to_id[token]

    def token_of(self, id: Int) raises -> String:
        # Inverse of id_of. Raises if the id is outside [0, size).
        if id < 0 or id >= len(self.id_to_token):
            raise Error("id out of range")
        return self.id_to_token[id]

    def encode(mut self, text: String) raises -> List[Int]:
        # Split on single spaces, auto-adding unseen words. Mutates self (grows
        # the vocabulary); allocates and returns the id list. Raises via add
        # (Dict subscript), never on well-formed input.
        var ids = List[Int]()
        for word in text.split(" "):
            ids.append(self.add(String(word)))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        # Join tokens with single spaces. Raises via token_of on an out-of-range
        # id — a bad id coming back from a model is a bug worth surfacing.
        var text = String("")
        for i in range(len(ids)):
            if i > 0:
                text += " "
            text += self.token_of(ids[i])
        return text


def new_vocabulary() -> Vocabulary:
    # An empty vocabulary. Separated from the struct so callers never touch the
    # two internal maps directly.
    var t2i: Dict[String, Int] = {}
    var i2t = List[String]()
    return Vocabulary(t2i^, i2t^)
