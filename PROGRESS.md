# Progress

Build status per part of the from-scratch Transformer. "Test command" is what
proves the part green on a fresh checkout (`pixi install` first).

| Part | Title | Status | Test command | Date |
|------|-------|--------|--------------|------|
| I | Foundations & config | ✅ green | `pixi run test` | 2026-07-03 |
| II | Vocabulary | ✅ green | `pixi run test` | 2026-07-03 |
| III | Tensors & ops | ✅ green | `pixi run test` | 2026-07-03 |
| IV | Utilities (rng, math) | ✅ green (`Rng` + float draws, timing) | `pixi run test` | 2026-07-03 |
| V | Tokenization | ✅ green | `pixi run test` | 2026-07-03 |
| VI | Dataset pipeline | ✅ green | `pixi run test` | 2026-07-03 |
| VII | Tiny bigram LM | ✅ green | `pixi run test` | 2026-07-03 |
| VIII | Architecture family | ✅ green (preset + exact param count + comptime pin) | `pixi run test` | 2026-07-03 |
| IX+ | GPT model & training | not started | — | — |

## Notes

- **Foundation restore (Parts II–III) landed on branch `foundation-restore`.**
  `config` (GPTConfig, TrainingConfig), `vocab` (toy whitespace Vocabulary), the
  `tensor` package (Tensor2D/3D, elementwise/matmul/softmax/cross-entropy ops,
  argmax, Xavier init), the `Rng` float draws (uniform/normal), and benchmark
  timing helpers. Every chapter deviation is recorded in
  [notes/part-07-notes.md](notes/part-07-notes.md). This unblocks Part VII, which
  needs float tensors for the bigram table.
- **Part V deliverables:** `CharTokenizer`, byte-level `BPETokenizer` (merge loop
  + didactic trainer), `GPT2Tokenizer` (GPT-2 vocab/merges, regex pre-tokenizer,
  byte↔unicode table), save/load for all three, and GPT-2 parity proven against a
  vendored OpenAI reference encoder. See [notes/part-05-notes.md](notes/part-05-notes.md).
- **Part VI deliverables:** a seeded `Rng` (LCG), and the tokenizer-agnostic data
  layer — `load_text`, `TokenDataset` + `train_val_split`, flat `[B, T]`
  `TokenBatch`, and `BatchLoader` (sliding windows, seeded epoch shuffle,
  remainder-drop) with `overfit_batch`. Tiny Shakespeare is committed for offline
  tests. See [notes/part-06-notes.md](notes/part-06-notes.md).

- **Part VII deliverables:** `BigramLM` (a single `[V, V]` logits table, filled
  either by `from_counts` with Laplace smoothing or trained from zeros/random),
  `loss_and_grad` with the fused `p − onehot` scatter-add, `perplexity`,
  `sgd_step`, the single-batch `train_bigram`, and `sample_categorical`. New
  `models/` package + scope. Trains char-level on tiny Shakespeare
  (`examples/bigram_shakespeare.mojo`); the `q → u` bigram is pinned. See
  [notes/part-07-notes.md](notes/part-07-notes.md).

- **Part VIII deliverables:** `GPTConfig.gpt2_124m()` (the reference GPT-2 small
  preset) and `GPTConfig.parameter_count()` (the exact GPT-2-layout total,
  124,439,808, derived in the docstring), pinned at compile time by
  `check_gpt2_contract()`. Plus two behavior-frozen metaprogramming cleanups:
  derived `comptime` constants replacing the magic mantissa literal in
  `utils/random.mojo`, and the GPT-2 byte↔unicode table bound at compile time in
  `tokenizer/gpt2.mojo` (`materialize`d at use). No layers, traits, or new
  packages — those are Part IX. See [notes/part-08-notes.md](notes/part-08-notes.md).

## Test suites (Parts II–VII)

| File | Covers |
|------|--------|
| `tests/test_smoke.mojo` | toolchain + `-I src` import path |
| `tests/test_config.mojo` | GPTConfig/TrainingConfig validate (both paths), d_head, param counts, Writable |
| `tests/test_vocab.mojo` | encode/decode round trip, add idempotence, unknown/out-of-range raises |
| `tests/test_tensor2d.mojo` | shape/offset, set-get, ones/zeros/full, checked access, from_rows |
| `tests/test_tensor3d.mojo` | nested row-major offset layout, set-get, checked access |
| `tests/test_elementwise.mojo` | add (+ mismatch raise), scale, transpose round trip |
| `tests/test_matmul.mojo` | hand-computed matmul, ijk/ikj agreement, matvec, mismatch raises |
| `tests/test_softmax.mojo` | rows sum to 1, stability at 1000-magnitude logits, temperature limits |
| `tests/test_cross_entropy.mojo` | uniform → log V, grad sums to 0, logsumexp stability |
| `tests/test_grad_check.mojo` | finite-difference check of cross_entropy_grad |
| `tests/test_finite_difference_step.mojo` | h-selection study (x³ truncation error = h²) |
| `tests/test_argmax.mojo` | max index, deliberate first-wins tie |
| `tests/test_timing.mojo` | median (odd/even), sort-in-place, hand-computed GFLOP/s |
| `tests/test_bigram.mojo` | zeros→log V, hand-computed count model, finite-diff gradient, grad rows sum to 0, perplexity=exp(loss) |
| `tests/test_bigram_training.mojo` | monotone decrease, deterministic-batch overfit, convergence to count optimum, training determinism, tiny-Shakespeare loss drop |
| `tests/test_sampler.mojo` | degenerate/one-hot, seed determinism, support-respect, invalid-probs raises, q→u plausibility pin |
| `tests/test_char_tokenizer.mojo` | codepoint vocab, round trips, save/load, errors |
| `tests/test_bpe_core.mojo` | merge loop, rank order, trainer (hand-computed), save/load |
| `tests/test_gpt2_tokenizer.mojo` | vocab size 50257, byte↔unicode bijection, oracle parity, goldens, save/load |
| `tests/test_rng.mojo` | LCG goldens, same-seed determinism, `next_below` range, shuffle permutation, float draws (uniform/normal), Xavier shape/determinism/no-NaN |
| `tests/test_dataset.mojo` | train/val split arithmetic + partition, corpus load + missing-file error |
| `tests/test_token_batch.mojo` | flat `[B, T]` layout, shape/bounds checks |
| `tests/test_batch_loader.mojo` | window shapes, shift-by-one, seeded epochs, coverage, remainder drop, end-to-end |
