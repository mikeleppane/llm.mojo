# Progress

Build status per part of the from-scratch Transformer. "Test command" is what
proves the part green on a fresh checkout (`pixi install` first).

| Part | Title | Status | Test command | Date |
|------|-------|--------|--------------|------|
| I | Foundations & config | absent (to restore before VI) | — | — |
| II | Vocabulary | absent (to restore before VI) | — | — |
| III | Tensors & ops | absent (to restore before VI) | — | — |
| IV | Utilities (rng, math) | absent (to restore before VI) | — | — |
| V | Tokenization | ✅ green | `pixi run test` | 2026-07-03 |
| VI+ | Model & training | not started | — | — |

## Notes

- **Parts I–IV are not in the repo.** `src/llm/` holds the tokenizer package and
  package `__init__` files only; there is no `config`, `vocab`, `tensor`, or
  `utils` code yet. Part V sits at the bottom of the dependency graph
  (`tokenizer → data → training`) and needs none of it, so it stands alone. The
  Parts I–IV foundation must be restored before Part VI, which depends on it.
- **Part V deliverables:** `CharTokenizer`, byte-level `BPETokenizer` (merge loop
  + didactic trainer), `GPT2Tokenizer` (GPT-2 vocab/merges, regex pre-tokenizer,
  byte↔unicode table), save/load for all three, and GPT-2 parity proven against a
  vendored OpenAI reference encoder. See [notes/part-05-notes.md](notes/part-05-notes.md).

## Test suites (Part V)

| File | Covers |
|------|--------|
| `tests/test_smoke.mojo` | toolchain + `-I src` import path |
| `tests/test_char_tokenizer.mojo` | codepoint vocab, round trips, save/load, errors |
| `tests/test_bpe_core.mojo` | merge loop, rank order, trainer (hand-computed), save/load |
| `tests/test_gpt2_tokenizer.mojo` | vocab size 50257, byte↔unicode bijection, oracle parity, goldens, save/load |
