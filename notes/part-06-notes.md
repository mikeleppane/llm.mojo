# Part VI — Dataset Pipeline: build notes

Raw material for the chapter: what was decided, what broke, and what surprised me
while building the data layer. Not published as-is.

## What shipped

- `src/llm/utils/random.mojo` — `Rng`, a minimal seeded LCG (Knuth MMIX
  constants): `next_u64`, `next_below`, in-place Fisher–Yates `shuffle`.
- `src/llm/data/` — the tokenizer-agnostic data layer:
  - `corpus.mojo` — `load_text` with an actionable missing-file error.
  - `dataset.mojo` — `TokenDataset`, `TrainValSplit`, `train_val_split`.
  - `batch.mojo` — `TokenBatch`, a flat row-major `[B, T]` container.
  - `loader.mojo` — `BatchLoader` (sliding windows + seeded epoch shuffle) and
    `overfit_batch`.
- `data/tinyshakespeare/input.txt` (committed, 1,115,394 bytes) +
  `scripts/download_tinyshakespeare.py` (provenance: source URL + pinned
  SHA-256).
- Four test files: `test_rng`, `test_dataset`, `test_token_batch`,
  `test_batch_loader` (the last one carries the end-to-end integration test).

## Design choices that held up

- **Epoch = seeded permutation of window starts.** Enumerate all starts once,
  Fisher–Yates-shuffle them with `Rng(seed)`, slice groups of B. The three
  properties the roadmap wanted fall out for free and are all exactly testable:
  same seed → identical batch sequence, every window seen once per epoch, and a
  real `num_batches()`. A fresh loader iterates in natural (unshuffled) order, so
  `start_epoch` is only needed when you actually want a shuffle — which is what
  makes `overfit_batch` a trivial "first batch, no shuffle" reuse of the loader.
- **The shift-by-one lives in the window, not in a later step.** A window at `s`
  is `ids[s .. s+T]` (T+1 tokens); inputs are the first T, targets the last T. A
  reader never has to hold "and then we shift" in their head — the two arrays are
  already aligned at construction.
- **Synthetic corpus `ids == positions` for the loader tests.** With
  `ids = [0, 1, 2, ...]`, `input_at(b, t)` is literally the window start plus t,
  so every expected value is arithmetic, not a magic constant. The shift-by-one
  and coverage tests read as proofs rather than fixtures.
- **`TokenBatch` accessors funnel through one bounds-checked `_flat_index`.** The
  check can't drift between `input_at` and `target_at` because there is only one
  copy of it.

## What broke / surprised me

- **The float split trap.** The plan wrote the split index as
  `floor(len * (1 - val_fraction))`. Under IEEE-754, `1.0 - 0.1 ==
  0.8999999999999999`, so `Int(100 * 0.8999…) == 89` — a 90/10 split silently
  becomes 89/11. Fixed by computing the *val* count directly,
  `val_count = floor(len * val_fraction)` (`100 * 0.1 == 10.0000…2 → 10`), and
  taking train as the remaining prefix. Same integer whenever the arithmetic is
  exact, correct integer when it is not. Lesson: never derive a count from
  `1 - fraction` in floating point when you can derive it from `fraction`.
- **You cannot `^` a field out of a live struct.** `BatchLoader(split.train^, …)`
  failed with "field 'split.train.ids' destroyed out of the middle of a value,
  preventing the overall value from being destroyed." A partial move leaves the
  aggregate un-destroyable. `.copy()` the field instead (`TokenDataset` is
  `Copyable`). Now recorded in AGENTS.md and the coding skill.
- **UInt64 arithmetic wraps, it does not trap** (the §7 risk). Probed before
  writing the LCG: `0*A+C`, `42*A+C`, and the next step all matched a Python
  `% 2**64` oracle exactly, so the recurrence needs no explicit masking. The
  golden test would have caught a non-wrapping multiply instantly.
- **The unreliable `TestSuite` timer again.** `test_end_to_end_tinyshakespeare`
  printed `[39.763]`; the real cost of the whole file by `time` is ~3 s
  wall-clock. It encodes all 1.1M codepoints of the corpus once — genuinely the
  slowest test in the repo, but nowhere near 40 "seconds." Same 1.0.0b2 quirk
  Part V flagged; don't trust the bracketed numbers.

## Determinism, proven

The RNG goldens are frozen from an independent Python computation of the
recurrence (not from memory): seed 0 → `[1442695040888963407,
1876011003808476466, 11166244414315200793]`, seed 42 → `[10481999410520546993,
4159066171780167020, 7615522811268512075]`. Everything downstream that is
"random" is really seeded and replayable, and the loader test asserts two
same-seed loaders produce byte-identical batches across a full epoch.

## Deferred / out of scope (as agreed)

- **No Box–Muller / Xavier on `Rng` yet.** Gaussian sampling and weight init
  belong on this same generator but aren't needed until the model-parameter
  layer; they will be added to `random.mojo`, not a new file. When the Parts I–IV
  chapter code arrives, its `Rng` reconciles *into* this one (same LCG, same
  constants), so the merge is additive.
- **No streaming / mmap datasets.** Tiny Shakespeare fits in memory ~1000× over;
  the whole corpus is a single `List[Int]`.
- **No random-offset (nanoGPT-style) sampling.** The permutation epoch is the one
  shipped; a step-based sampler can be added later as a thin alternative if the
  training loop wants it.
