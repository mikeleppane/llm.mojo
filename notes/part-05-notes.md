# Part V — Tokenization: build notes

Raw material for the chapter: what was decided, what broke, and what surprised me
while implementing the tokenizers. Not published as-is.

## What shipped

- `src/llm/tokenizer/char.mojo` — `CharTokenizer`, codepoint-level.
- `src/llm/tokenizer/bpe.mojo` — `BPETokenizer`, byte-level BPE core + trainer.
- `src/llm/tokenizer/gpt2.mojo` — `GPT2Tokenizer`, GPT-2 vocab/merges + regex
  pre-tokenizer.
- `data/gpt2/{vocab.json,merges.txt}` + `scripts/download_gpt2_files.py`
  (provenance: source URLs + pinned SHA-256).
- `tests/oracles/gpt2_reference_encoder.py` — vendored OpenAI encoder, oracle only.
- Four test files; the capstone is GPT-2 parity against the oracle over a fixed
  sample set, plus frozen golden ids.

## Design choices that held up

- **BPE core is integers, not unicode-mapped strings.** Tokens are `Int` ids,
  `vocab[id]` is a `List[UInt8]`, merges live in two `Dict[Int, Int]` keyed by a
  packed pair key (`left * 2**32 + right`). GPT-2's byte↔unicode alphabet only
  ever appears in the loader, translated back to raw bytes exactly once. The
  merge loop then reads as pure integer algorithmics — which is the whole point
  of teaching it. The alternative (mirror OpenAI's `encoder.py` and merge over
  strings) drags UTF-8/byte-indexing juggling through the hot loop.
- **Save format is integers-only text**, `CHARTOK v1` / `BPETOK v1`. Storing
  codepoints/bytes as decimal integers sidesteps every newline/whitespace
  escaping bug a character-literal format would invite.
- **The oracle is OpenAI's own `encoder.py` algorithm, vendored**, not tiktoken
  or transformers. It needs only `json` + `regex` (which the pre-tokenizer needs
  anyway), reads the same committed files, and is an independent implementation —
  so parity against it is meaningful and stays offline.

## What broke / surprised me

- **The "aaabdaaabac" golden was wrong in my head, not in the code.** I wrote the
  expected encoding as `[258, 100, 258, 99]`, dropping the trailing `a`. The
  canonical Wikipedia walkthrough ends at `"XdXac"` = `[258, 100, 258, 97, 99]`
  (5 symbols). The test caught my arithmetic slip — exactly what the hand-computed
  oracle is for. Lesson: compute the golden twice, independently.
- **`byte_to_id` is a permutation for GPT-2, not the identity.** The from-scratch
  tokenizer has `vocab[i] == [i]` and `byte_to_id[i] == i`, so my first
  `BPETokenizer.load` just rebuilt `byte_to_id` as identity — and every
  from-scratch save/load test passed. GPT-2's `vocab.json` orders the single-byte
  tokens differently (id 0 is `"!"`, byte 33), so identity was wrong and the
  GPT-2 save/load round trip failed. Fix: derive `byte_to_id` from the length-1
  vocab entries on load (merged tokens are always ≥2 bytes, so length-1 entries
  are exactly the base bytes). Handles both cases without a format change.
- **`TestSuite`'s per-test timing display is unreliable in mojo 1.0.0b2.** It
  printed `[70.977]`, `[309.171]` for tests whose real cost is milliseconds; the
  actual `time` wall-clock for the whole GPT-2 file is ~4 s. `GPT2Tokenizer.from_files`
  measured with `perf_counter_ns` is ~70 ms. Don't trust the bracketed numbers;
  measure with `time` if you care.

## Mojo API facts pinned by compile probes (1.0.0b2)

- **String → bytes:** `s.as_bytes()` returns an indexable `Span[UInt8]`.
- **bytes → String:** three keyword constructors —
  `String(from_utf8_lossy=Span(list))` (U+FFFD on invalid, used by `decode`),
  `String(from_utf8=...)` (strict), `String(unsafe_from_utf8=...)` (no check).
  No hand-rolled UTF-8 decoder needed; the lossy path was the contingency plan
  and turned out unnecessary.
- **PythonObject → Int:** `Int(py=obj)`, *not* `Int(obj)` — `PythonObject` does
  not conform to `Intable` here.
- **`PythonObject` as a struct field** works with `Movable` and `^` transfer; a
  compiled `regex` pattern can be held on the tokenizer and reused per call.
- **`comptime` constants re-export** cleanly through `__init__.mojo`
  (`from .gpt2 import GPT2_VOCAB_SIZE`), so the struct-field fallback was not
  needed.
- **`match` is reserved** — a `for match in ...:` loop fails to parse; rename the
  loop variable.
- **Codepoints:** `s.codepoints()` yields `Codepoint` (`Int(cp)` for the value);
  `Codepoint.from_u32(UInt32(v)).value()` builds one; `String(codepoint)` makes
  the one-character string. `sort(list)` is a free function.
- **Temp files:** `from std.tempfile import gettempdir` → `/tmp`; used for
  save/load round-trip tests.

## Deferred / out of scope (as agreed)

- No special-token handling in `encode` (matches OpenAI's `encoder.py`);
  `END_OF_TEXT_ID = 50256` is exposed for later parts.
- No performance work on the merge loop (naive rescan per merge; chunks are short
  after pre-splitting). Performance is a later part's concern.
- Restoring Parts I–IV: separate chore before Part VI.
