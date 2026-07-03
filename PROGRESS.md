# Progress

Build status per part of the from-scratch Transformer. "Test command" is what
proves the part green on a fresh checkout (`pixi install` first).

| Part | Title | Status | Test command | Date |
|------|-------|--------|--------------|------|
| I | Foundations & config | absent (to restore) | — | — |
| II | Vocabulary | absent (to restore) | — | — |
| III | Tensors & ops | absent (to restore before VII) | — | — |
| IV | Utilities (rng, math) | partial (`Rng` only) | `pixi run test` | 2026-07-03 |
| V | Tokenization | ✅ green | `pixi run test` | 2026-07-03 |
| VI | Dataset pipeline | ✅ green | `pixi run test` | 2026-07-03 |
| VII+ | Model & training | not started | — | — |

## Notes

- **Parts I–IV are still mostly absent.** `src/llm/` holds the tokenizer and data
  packages plus a minimal `utils/random.mojo` (`Rng` only); there is no `config`,
  `vocab`, or `tensor` code yet. Parts V–VI sit at the bottom of the dependency
  graph (`tokenizer → data → training`) and need none of that, so they stand
  alone. The Parts I–IV foundation — `config`, `vocab`, and especially `tensor`
  (float math) plus the Box–Muller/Xavier extensions to `Rng` — must be restored
  before Part VII, which needs float tensors for the bigram table.
- **Part V deliverables:** `CharTokenizer`, byte-level `BPETokenizer` (merge loop
  + didactic trainer), `GPT2Tokenizer` (GPT-2 vocab/merges, regex pre-tokenizer,
  byte↔unicode table), save/load for all three, and GPT-2 parity proven against a
  vendored OpenAI reference encoder. See [notes/part-05-notes.md](notes/part-05-notes.md).
- **Part VI deliverables:** a seeded `Rng` (LCG), and the tokenizer-agnostic data
  layer — `load_text`, `TokenDataset` + `train_val_split`, flat `[B, T]`
  `TokenBatch`, and `BatchLoader` (sliding windows, seeded epoch shuffle,
  remainder-drop) with `overfit_batch`. Tiny Shakespeare is committed for offline
  tests. See [notes/part-06-notes.md](notes/part-06-notes.md).

## Test suites (Parts V–VI)

| File | Covers |
|------|--------|
| `tests/test_smoke.mojo` | toolchain + `-I src` import path |
| `tests/test_char_tokenizer.mojo` | codepoint vocab, round trips, save/load, errors |
| `tests/test_bpe_core.mojo` | merge loop, rank order, trainer (hand-computed), save/load |
| `tests/test_gpt2_tokenizer.mojo` | vocab size 50257, byte↔unicode bijection, oracle parity, goldens, save/load |
| `tests/test_rng.mojo` | LCG goldens, same-seed determinism, `next_below` range, shuffle permutation |
| `tests/test_dataset.mojo` | train/val split arithmetic + partition, corpus load + missing-file error |
| `tests/test_token_batch.mojo` | flat `[B, T]` layout, shape/bounds checks |
| `tests/test_batch_loader.mojo` | window shapes, shift-by-one, seeded epochs, coverage, remainder drop, end-to-end |
