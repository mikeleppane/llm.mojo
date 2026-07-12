"""Byte-level Byte-Pair Encoding (BPE) — the core algorithm.

Starts from the 256 raw byte values as tokens and repeatedly merges the most
useful adjacent pair into a new token, so it can encode any input with no
out-of-vocabulary character. A token is an `Int` id, `vocab[id]` is its byte
sequence, and a merge of ids (left, right) is keyed by a single packed `Int`
(see `pair_key`) into `merge_rank` and `merge_result`. Keeping the hot loop in
integers, with no unicode string juggling, is what makes the algorithm readable.
"""

# Ids are well under 2**31, so a pair (left, right) packs losslessly into one
# Int: left occupies the high bits, right the low bits. A single Int key lets the
# merge tables be plain Dict[Int, Int] and avoids allocating a string per lookup
# in the innermost loop.
comptime PAIR_BASE = 1 << 32

# Save-format tag (see save/load). Byte sequences are stored as decimal integers.
comptime BPETOK_MAGIC = "BPETOK v1"

# Number of base tokens: one per raw byte value.
comptime N_BYTES = 256


def pair_key(left: Int, right: Int) -> Int:
    """Pack an adjacent (left, right) id pair into one Int key.

    Args:
        left: Left token id.
        right: Right token id.

    Returns:
        A lossless packed key for ids well under 2**31 (ample for GPT-2's
        50257). Does not allocate; does not raise.
    """
    return left * PAIR_BASE + right


struct BPETokenizer(Copyable, Movable):
    """Byte-level BPE tokenizer: vocab, base-byte ids, and merge tables."""

    var vocab: List[List[UInt8]]  # token id -> its byte sequence
    var byte_to_id: List[Int]  # 256 entries: raw byte value -> base token id
    var merge_rank: Dict[Int, Int]  # pair_key -> rank (lower merges first)
    var merge_result: Dict[Int, Int]  # pair_key -> merged token id

    def __init__(out self):
        """Initialize the base vocab: 256 single-byte tokens, no merges."""
        self.vocab = []
        self.byte_to_id = []
        self.merge_rank = {}
        self.merge_result = {}
        for b in range(N_BYTES):
            self.vocab.append([UInt8(b)])
            self.byte_to_id.append(b)

    def vocab_size(self) -> Int:
        return len(self.vocab)

    def register_merge(mut self, left: Int, right: Int) raises -> Int:
        """Append one merge at the next rank and create its token.

        The new id is the current vocab size; its bytes are the left token's
        bytes followed by the right token's.

        Args:
            left: Left token id of the pair.
            right: Right token id of the pair.

        Returns:
            The new merged token id. Mutates self; allocates.

        Raises:
            Error: If the pair was already merged, which would desync the rank
                sequence from merge_rank's length and break save().
        """
        var key = pair_key(left, right)
        if key in self.merge_rank:
            raise Error("BPETokenizer.register_merge: pair already merged")
        var new_id = len(self.vocab)
        self.merge_rank[key] = len(self.merge_rank)
        self.merge_result[key] = new_id
        var merged = self.vocab[left].copy()
        merged.extend(self.vocab[right].copy())
        self.vocab.append(merged^)
        return new_id

    def encode_bytes(self, chunk: List[UInt8]) -> List[Int]:
        """Run the BPE merge loop over one chunk of bytes.

        Starts from one id per byte, then repeatedly finds the present adjacent
        pair with the lowest merge rank and replaces every occurrence with its
        merged id, until no pair is mergeable.

        Args:
            chunk: The raw bytes to encode.

        Returns:
            The token ids. Allocates; does not mutate self; cannot raise.
        """
        var ids: List[Int] = []
        for i in range(len(chunk)):
            ids.append(self.byte_to_id[Int(chunk[i])])

        while len(ids) >= 2:
            # Find the mergeable pair type with the lowest rank in the sequence.
            var best_rank = -1
            var best_key = 0
            for i in range(len(ids) - 1):
                var key = pair_key(ids[i], ids[i + 1])
                var rank_ptr = self.merge_rank.get(key)
                if rank_ptr:
                    var rank = rank_ptr.value()
                    if best_rank == -1 or rank < best_rank:
                        best_rank = rank
                        best_key = key
            if best_rank == -1:
                break  # nothing left to merge

            # Replace every occurrence of that pair with its merged id. The key
            # came from merge_rank, whose keys mirror merge_result, so the
            # lookup always hits — a non-raising get keeps encode_bytes total.
            var merged_id = self.merge_result.get(best_key).value()
            var merged_ids: List[Int] = []
            var i = 0
            while i < len(ids):
                if (
                    i < len(ids) - 1
                    and pair_key(ids[i], ids[i + 1]) == best_key
                ):
                    merged_ids.append(merged_id)
                    i += 2
                else:
                    merged_ids.append(ids[i])
                    i += 1
            ids = merged_ids^
        return ids^

    def encode(self, text: String) -> List[Int]:
        """Encode a string via its UTF-8 bytes through the merge loop.

        The core BPE tokenizer does no pre-splitting (that is GPT-2's job).

        Args:
            text: The string to encode.

        Returns:
            The token ids. Cannot raise: every byte has a base id.
        """
        var bytes_view = text.as_bytes()
        var chunk: List[UInt8] = []
        for i in range(len(bytes_view)):
            chunk.append(bytes_view[i])
        return self.encode_bytes(chunk)

    def decode(self, ids: List[Int]) raises -> String:
        """Concatenate each id's bytes and decode UTF-8 lossily.

        Uses U+FFFD replacement for invalid sequences, since ids may split a
        multi-byte character and decoding must never trap.

        Args:
            ids: Token ids to decode.

        Returns:
            The decoded string. Allocates.

        Raises:
            Error: If an id is out of range.
        """
        var bytes: List[UInt8] = []
        for i in range(len(ids)):
            var id = ids[i]
            if id < 0 or id >= len(self.vocab):
                raise Error(
                    "BPETokenizer.decode: id out of range: " + String(id)
                )
            for b in range(len(self.vocab[id])):
                bytes.append(self.vocab[id][b])
        return String(from_utf8_lossy=Span(bytes))

    def train(mut self, text: String, target_vocab_size: Int) raises:
        """Train merges over the whole byte sequence, minbpe-style.

        Repeatedly counts adjacent pairs, merges the most frequent, and repeats
        until the vocab reaches target_vocab_size. No pre-splitting (the
        teaching variant; GPT-2 trained over pre-split words). Deterministic
        tie-break: highest count wins, then lowest pair key. Mutates self;
        allocates.

        Args:
            text: The training corpus.
            target_vocab_size: Desired final vocab size (>= 256).

        Raises:
            Error: If target_vocab_size is below the 256 base byte tokens.
        """
        if target_vocab_size < N_BYTES:
            raise Error(
                "BPETokenizer.train: target_vocab_size must be >= "
                + String(N_BYTES)
                + " (the base byte tokens)"
            )
        var num_merges = target_vocab_size - self.vocab_size()

        # Working sequence of ids. Encoding with the merges learned so far means
        # already-merged pairs don't reappear, so training an already-trained
        # tokenizer extends its vocab instead of re-learning (and re-registering)
        # a pair it already knows. For a fresh tokenizer this is just the raw
        # per-byte ids.
        var bytes_view = text.as_bytes()
        var raw: List[UInt8] = []
        for i in range(len(bytes_view)):
            raw.append(bytes_view[i])
        var ids = self.encode_bytes(raw)

        for _ in range(num_merges):
            if len(ids) < 2:
                break
            # Count every adjacent pair, tracking the best (most frequent, then
            # lowest pair key) as we go.
            var counts = Dict[Int, Int]()
            for i in range(len(ids) - 1):
                var key = pair_key(ids[i], ids[i + 1])
                counts[key] = counts.get(key, 0) + 1
            var best_key = -1
            var best_count = 0
            for entry in counts.items():
                var key = entry.key
                var count = entry.value
                if count > best_count or (
                    count == best_count and key < best_key
                ):
                    best_count = count
                    best_key = key
            if best_key == -1:
                break  # no adjacent pairs at all

            # Unpack the winning pair and learn it, then rewrite the sequence.
            var left = best_key // PAIR_BASE
            var right = best_key % PAIR_BASE
            var new_id = self.register_merge(left, right)
            var merged_ids: List[Int] = []
            var i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == left and ids[i + 1] == right:
                    merged_ids.append(new_id)
                    i += 2
                else:
                    merged_ids.append(ids[i])
                    i += 1
            ids = merged_ids^

    def save(self, path: String) raises:
        """Write the tokenizer as BPETOK v1.

        Format: magic line, vocab size, one line per token in id order (byte
        count then byte values), then the merge count and one line per merge in
        rank order (left, right, merged). Integers only, so no escaping is
        needed.

        Args:
            path: Destination file path (overwritten).
        """
        var text = (
            String(BPETOK_MAGIC) + "\n" + String(self.vocab_size()) + "\n"
        )
        for id in range(len(self.vocab)):
            var n = len(self.vocab[id])
            text += String(n)
            for b in range(n):
                text += " " + String(Int(self.vocab[id][b]))
            text += "\n"
        # Merges in rank order: invert merge_rank so line order == rank.
        var n_merges = len(self.merge_rank)
        text += String(n_merges) + "\n"
        var merges_by_rank = List[Int](length=n_merges, fill=0)
        for entry in self.merge_rank.items():
            merges_by_rank[entry.value] = entry.key
        for r in range(n_merges):
            var key = merges_by_rank[r]
            var left = key // PAIR_BASE
            var right = key % PAIR_BASE
            var merged = self.merge_result[key]
            text += (
                String(left) + " " + String(right) + " " + String(merged) + "\n"
            )
        with open(path, "w") as f:
            f.write(text)

    @staticmethod
    def load(path: String) raises -> BPETokenizer:
        """Restore a tokenizer written by save.

        Args:
            path: File to read.

        Returns:
            The restored tokenizer. Allocates.

        Raises:
            Error: On a bad magic line or a structurally inconsistent file.
        """
        var content = open(path, "r").read()
        var lines = content.split("\n")
        if len(lines) < 2 or String(lines[0]) != BPETOK_MAGIC:
            raise Error(
                "BPETokenizer.load: bad magic line (expected '"
                + String(BPETOK_MAGIC)
                + "')"
            )
        var vocab_size = Int(String(lines[1]))
        var tok = BPETokenizer()
        tok.vocab = []
        tok.byte_to_id = []
        var cursor = 2
        for _ in range(vocab_size):
            if cursor >= len(lines):
                raise Error("BPETokenizer.load: truncated vocab section")
            var parts = String(lines[cursor]).split(" ")
            var n = Int(String(parts[0]))
            var token: List[UInt8] = []
            for j in range(n):
                token.append(UInt8(Int(String(parts[1 + j]))))
            tok.vocab.append(token^)
            cursor += 1
        # Recover byte_to_id from the single-byte vocab entries. Merged tokens
        # are always >= 2 bytes, so every length-1 entry is a base byte token.
        # This is a permutation for GPT-2 (byte b's id is not b) and the identity
        # for a from-scratch tokenizer; deriving it handles both.
        tok.byte_to_id = List[Int](length=N_BYTES, fill=-1)
        for id in range(len(tok.vocab)):
            if len(tok.vocab[id]) == 1:
                tok.byte_to_id[Int(tok.vocab[id][0])] = id
        for b in range(N_BYTES):
            if tok.byte_to_id[b] == -1:
                raise Error(
                    "BPETokenizer.load: no single-byte token for byte "
                    + String(b)
                )
        # Merge section: count then one "left right merged" per rank.
        if cursor >= len(lines):
            raise Error("BPETokenizer.load: missing merge section")
        var n_merges = Int(String(lines[cursor]))
        cursor += 1
        for r in range(n_merges):
            if cursor >= len(lines):
                raise Error("BPETokenizer.load: truncated merge section")
            var parts = String(lines[cursor]).split(" ")
            var left = Int(String(parts[0]))
            var right = Int(String(parts[1]))
            var merged = Int(String(parts[2]))
            var key = pair_key(left, right)
            tok.merge_rank[key] = r
            tok.merge_result[key] = merged
            cursor += 1
        return tok^
