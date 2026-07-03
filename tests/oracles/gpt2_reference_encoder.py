"""Vendored pure-Python GPT-2 encoder — the parity oracle for the Mojo tests.

This is OpenAI's *original* GPT-2 byte-pair encoding algorithm (from the public
``gpt-2`` repository, ``src/encoder.py``, MIT licensed), lightly adapted to load
the reference files this repo already commits under ``data/gpt2/``. It is an
independent reference implementation — OpenAI's, not ours — so it is a trusted,
independent oracle for the tokenizer parity tests.

It is used **only** by tests. Nothing under ``src/`` imports it. Its sole job is
to answer, for a given input string, "what token ids does canonical GPT-2
produce?" — which the Mojo ``GPT2Tokenizer`` must match exactly.

The only third-party dependency is ``regex`` (for the ``\\p{L}``/``\\p{N}``
pre-tokenizer pattern), which the tokenizer itself needs anyway.

Reference: https://github.com/openai/gpt-2/blob/master/src/encoder.py
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import regex as re

# data/gpt2/ relative to the repo root (this file is tests/oracles/…).
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gpt2"


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> unicode-codepoint map (256 entries).

    Printable byte values map to their own character; the remaining bytes map to
    codepoints starting at 256. This keeps every byte representable as a single
    printable unicode character so BPE can run over strings. Verbatim from
    OpenAI's encoder.py.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    chars = [chr(c) for c in cs]
    return dict(zip(bs, chars))


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Return the set of adjacent symbol pairs in ``word``."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


# GPT-2's pre-tokenizer pattern: contractions, then letter / number / other
# runs, each optionally prefixed by a single space, then whitespace.
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Encoder:
    """OpenAI's GPT-2 encoder over the committed reference files."""

    def __init__(self, encoder: dict[str, int], bpe_merges: list[tuple[str, str]]):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache: dict[str, str] = {}
        self.pat = re.compile(GPT2_PATTERN)

    def bpe(self, token: str) -> str:
        """Apply the merge loop to one unicode-mapped token; return merged symbols
        joined by spaces."""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def pre_tokenize(self, text: str) -> list[str]:
        """The regex pre-split only (exposed so tests can localize failures to the
        split vs. the merge loop)."""
        return re.findall(self.pat, text)

    def encode(self, text: str) -> list[int]:
        bpe_tokens: list[int] = []
        for token in re.findall(self.pat, text):
            token_bytes = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token_bytes).split(" "))
        return bpe_tokens

    def decode(self, tokens: list[int]) -> str:
        text = "".join(self.decoder[token] for token in tokens)
        return bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")


def _load_from(data_dir: Path) -> Encoder:
    with open(data_dir / "vocab.json", encoding="utf-8") as f:
        encoder = json.load(f)
    with open(data_dir / "merges.txt", encoding="utf-8") as f:
        bpe_data = f.read()
    # First line is a "#version: …" comment; the rest are "left right" merges.
    bpe_merges = [tuple(line.split()) for line in bpe_data.split("\n")[1:-1]]
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


@lru_cache()
def get_encoder() -> Encoder:
    """Build the reference encoder from this repo's committed data/gpt2/ files."""
    return _load_from(_DATA_DIR)


# Convenience wrappers so the Mojo interop side can call flat module functions
# (calling a bound method through PythonObject is fiddlier than a free function).
def reference_encode(text: str) -> list[int]:
    return get_encoder().encode(text)


def reference_decode(tokens: list[int]) -> str:
    return get_encoder().decode(list(tokens))


def reference_pre_tokenize(text: str) -> list[str]:
    return get_encoder().pre_tokenize(text)


if __name__ == "__main__":
    enc = get_encoder()
    for sample in ["Hello world", "don't", "  leading", "123", "café"]:
        print(repr(sample), "->", enc.encode(sample))
