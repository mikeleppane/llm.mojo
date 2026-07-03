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
  same seed → identical batch sequence, every window *eligible* each epoch (with
  a seed-dependent remainder of `num_windows % B` dropped — full once-only
  coverage holds exactly when `B` divides `num_windows`), and a real
  `num_batches()`. A fresh loader iterates in natural (unshuffled) order, so
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

## What external review caught

Two independent reviews ran against the branch diff: Codex (GPT-5.5, high
reasoning) and Claude Opus 4.8 (xhigh).

- **[Codex #1, High — accepted, already fixed] `start_epoch` was not
  seed-idempotent.** It shuffled `self.order` in place without restoring the
  natural order first, so a second `start_epoch(seed)` shuffled the previous
  epoch's permutation — making the order a function of the whole epoch history,
  not just the seed. I had independently found and fixed this before the reviews
  returned (extract `_window_starts`, rebuild before each shuffle; regression
  test re-seeds after a full epoch). Both the reviewer and I landed on the same
  hole, which is a good sign the test now guards it.
- **[Codex #2, Medium — accepted as a doc fix] `next_below`/`shuffle` claimed
  exact uniformity.** The modulo reduction is only exactly uniform when `n`
  divides 2⁶⁴; the docstrings said "uniform" while also documenting the bias —
  contradictory. Reworded to "near-uniform" pointing at the bias note. The bias
  itself is accepted (negligible at our sizes; a rejection loop would clutter
  teaching code), so this was wording, not behavior.
- **[Codex #3, Low — accepted] `TokenBatch` accepted non-positive dimensions.**
  It checked only the flat length, so `batch_size=-2, seq_len=-3` (product 6)
  passed against a 6-element array, leaving accessors with ranges like `[0, -2)`.
  `BatchLoader` guards itself, but `TokenBatch` is public and must enforce its
  own invariant. Added `batch_size >= 1` / `seq_len >= 1` checks + a test.
- **Clean areas confirmed by Codex:** window bounds and shift-by-one (`s + T`
  stays in range; inputs `ids[s .. s+T-1]`, targets `ids[s+1 .. s+T]`), the
  `data → utils` layering direction, and no syntax-contract violations. Codex
  could not run the suite (its sandbox blocked loopback networking); the review
  was static against the diff, which is fine — the suite is green locally and in
  a committed-files-only fresh tree.

### Opus 4.8 (xhigh) — no correctness bugs; 6 Low/Nit findings

Opus hand-verified the seed-42 golden and the windowing math and found **no
correctness bugs** — windowing, shift-by-one, split arithmetic, determinism,
layering, and the syntax contract all clean. The six Low/Nit items:

- **[#1 Low, accepted — doc]** The note said "every window seen once per epoch,"
  which is only true when `B ∣ num_windows`; with remainder-drop an epoch visits
  `num_batches × B` windows. Tightened the note wording above.
- **[#2 Low, accepted — doc]** Natural-order iteration permanently drops the same
  corpus tail every epoch. Added a paragraph to the `BatchLoader` module docstring
  telling training loops to `start_epoch(seed + epoch)` each epoch (which also
  reshuffles which windows fall in the dropped remainder).
- **[#3 Low, accepted — test]** Remainder-drop was only tested in a divisible
  case, and the remainder test asserted only the batch *count*. Strengthened
  `test_num_batches_drops_remainder` to collect the window starts and assert
  exactly 8 *distinct* windows appear (no duplicate, no over-run).
- **[#4 Nit, rejected]** "`sort` used with no visible import." It is genuinely
  prelude-provided in 1.0.0b2: the merged Part V `char.mojo` calls `sort(...)`
  with zero imports and is green. Adding an explicit import only in the new tests
  would be inconsistent with the shipped codebase, so no change.
- **[#5 Nit, accepted]** Two goldens exceed 2⁶³−1 and only compile because the
  `IntLiteral` flows straight into `UInt64(...)`. Added a comment in
  `test_rng.mojo` warning not to route them through an `Int` intermediate.
- **[#6 Nit, partially accepted — test]** Added the two useful uncovered edges:
  empty `ids` into `train_val_split` (raises) and `overfit_batch` on a
  too-small dataset (raises). The third (`TokenBatch` with a zero dimension) is
  already covered — the Codex-driven positivity check now makes it raise at
  construction, so Opus's "constructs, then accessors raise" description was
  superseded by that fix.

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
