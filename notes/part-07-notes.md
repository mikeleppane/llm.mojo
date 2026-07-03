# Part VII — Tiny Bigram LM: build notes

Raw material for the chapter(s): what was decided, what broke, and every place
the restored foundation code deviates from the chapter markdowns. Part VII ships
on two branches — `foundation-restore` (Parts II–III foundation) then
`part-07-bigram` (the model). Not published as-is; the chapter markdowns get
corrected *from* this file.

---

## Phase 0 — foundation restore (branch `foundation-restore`)

Restored the Parts II–III code from `docs/chapters/02-*.md` and `03-*.md`,
treating every snippet as starter code that had never compiled. What shipped,
file by file: `config.mojo` (GPTConfig, TrainingConfig), `vocab.mojo` (toy
whitespace Vocabulary), `tensor/{tensor2d,tensor3d,ops,init_weights}.mojo`,
`utils/{random,timing}.mojo` additions, `examples/config_summary.mojo`,
`benchmarks/bench_matmul.mojo`, and their tests.

### Deviations from the chapter code (blog gold — the chapters get fixed to match)

1. **`Writable.write_to` signature.** The chapters write
   `def write_to[W: Writer](self, mut writer: W)`. That parametric form is stale;
   the current syntax (verified by compiling a probe) is
   `def write_to(self, mut writer: Some[Writer])` — `Some[Writer]` is the builtin
   existential, not a bare generic parameter. Used the current form in
   `GPTConfig`.

2. **`Vocabulary.add`/`.encode` must be `raises`.** The chapter marks both
   non-raising. On the pinned Mojo (1.0.0b2) `Dict.__getitem__` is a *raising*
   operation, so `return self.token_to_id[token]` forces `add` to be `raises`,
   which cascades to `encode` (it calls `add`). Recorded on the methods that a
   present-key lookup never actually fails — the `raises` is a language
   artifact, not a real failure mode. This is a genuine chapter bug: the code as
   printed does not compile.

3. **`xavier_2d` moved to the tensor layer.** The chapter (§15) places
   `xavier_2d` in `utils/random.mojo`, but it imports `Tensor2D` — and the
   dependency graph runs `utils → tensor`, so a utils module importing a tensor
   points *up* the graph (a cycle risk). Moved it to
   `tensor/init_weights.mojo`; the RNG it draws from stays in utils, which the
   tensor layer imports downward. File named `init_weights.mojo` to avoid
   clashing with the package `__init__.mojo`.

4. **`argmax` promoted into the library.** The chapter (II §7) defines `argmax`
   only inside a test file (duplicated). Promoted it to `tensor/ops.mojo`
   (first-wins ties, strict `>`) with its tie test — greedy decoding later needs
   it in the library, not copy-pasted per test.

5. **RNG reconciliation, additive only.** Part VI's `Rng` (LCG, Knuth MMIX
   constants, frozen goldens) is live. The chapter (§15) shows a *fresh* `Rng`
   plus a `new_rng(seed)` factory that rewrites seed 0 to `0x9E3779B9…` "to avoid
   the degenerate all-zero state." That guard was **deliberately not ported**:
   an LCG with an odd increment `C` has full period from *every* state including
   0 (state 0 → first output `C`), so seed 0 is not degenerate — and Part VI's
   seed-0 golden test pins exactly that first output. Silently swapping the seed
   would be the kind of hidden magic AGENTS.md bans, and would break the golden.
   So: kept Part VI's struct and seed semantics verbatim; *added* `uniform()`
   (top-53-bit construction), `uniform_range()`, and `normal()` (Box–Muller with
   the `1e-300` log guard). `new_rng` is not ported — construct `Rng(seed)`
   directly. The frozen goldens still pass unchanged (the merge is correct by
   that test).

6. **Trait lists.** Chapters use the short `struct T(Copyable)` (Copyable implies
   Movable). The existing repo writes `(Copyable, Movable)` explicitly
   (`Rng`, `TokenBatch`). Matched the repo for consistency, not the chapter.

7. **`median_ns` upper-middle median.** Kept the chapter's deliberate
   simplification (returns the upper-middle element for an even count instead of
   averaging the two middles) — it never matters for a benchmark median — and
   pinned it with an even-count test so the behavior is a documented choice, not
   an accident.

8. **Benchmark landed, not skipped.** The chapter marks the timer `sketch`
   pending symbol verification. `std.time.perf_counter_ns` compiles and runs on
   1.0.0b2, so `benchmarks/bench_matmul.mojo` landed for real (outside the test
   gate — it measures, does not assert). Debug-build numbers show ikj only
   marginally ahead of ijk at 64/128/256; the dramatic gap needs a release build
   and larger sizes (performance chapters), so no ratio is asserted.

9. **`__init__` re-exports.** Added `llm/__init__.mojo` (config, vocab) and
   `tensor/__init__.mojo` (all tensor symbols), and extended `utils/__init__.mojo`
   with the timing helpers — the package tables of contents the chapters describe
   but the scaffold left empty.

### Probed before landing (verified facts)

- `from std.math import exp, log, sqrt, cos, pi` — all present.
- `from std.math import isnan` — present (used in the RNG finite-value test).
- `write_to(self, mut writer: Some[Writer])` — compiles; `String.write(cfg)`
  renders it.
- `std.time.perf_counter_ns` — present.

### External review triage (foundation-restore)

Two independent read-only reviews ran against `git diff main...foundation-restore`:
Codex (GPT-5.5, high reasoning) and Claude Opus 4.8 (xhigh). They converged
strongly — **every finding accepted**, several raised by both. Both confirmed
the syntax contract, the numerics, the layering, and (critically) that the
additive `Rng` merge leaves the frozen Part VI goldens byte-for-byte unchanged.

Each accepted finding got a failing test first, then the fix:

- **[Codex High — accepted] `softmax_row_temperature` overflowed at extreme T.**
  It divided by `temperature` *before* the max subtraction, so a near-zero T
  sent a large logit to `+inf` and the softmax returned `inf/inf = NaN`
  (`softmax_row_temperature([1000, 0], 1e-307)`). The chapter explicitly (and
  wrongly) claims stability "carries over for free" from the wrapper form. Fixed
  by not delegating: subtract the row max first and divide the *difference*,
  `exp((x_i - max)/T)` — algebraically identical, but every exponent is ≤ 0 so
  nothing overflows. This is a genuine chapter correction (blog gold). Test:
  `test_extreme_low_temperature_is_stable`.
- **[Opus Medium — accepted] Float draws had no value goldens.** `uniform()`,
  `normal()`, and `xavier_2d` were only tested for range/finiteness/determinism,
  so a stub `return 0.5` / `return 0.0` / all-zeros would pass. Added independent
  goldens derived from the frozen `next_u64` values: `Rng(0).uniform() =
  (1442695040888963407 >> 11) / 2**53 = 0.0782086…`, `Rng(42).uniform() =
  0.5682303…`, and a Box–Muller oracle `Rng(0).normal(0,1) = 1.8121678…`, plus a
  distinct-values check so a stuck generator fails.
- **[Codex Medium / Opus Low — accepted] `GPTConfig.validate` ignored `dropout`.**
  A probability outside `[0, 1)` passed. Added the bound + `test_dropout_*`.
- **[Codex Medium — accepted] `xavier_2d` accepted non-positive fans** (divide by
  zero / degenerate shape). Made it `raises` and reject `fan_in/fan_out <= 0`.
- **[Codex Medium / Opus Low — accepted] `softmax_rows` read `scores[r, 0]` with
  no `cols == 0` guard** (out-of-bounds on a 0-column tensor, inconsistent with
  `softmax_row`'s empty handling). Guarded; returns the empty tensor unchanged.
- **[Opus Low — accepted] `cross_entropy_grad` skipped the target range check**
  that `cross_entropy_one` performs (silent out-of-bounds write). Made it
  `raises` with the same guard, so loss and gradient reject bad targets
  symmetrically.
- **[Opus Nit — accepted] `matmul_ikj`'s shape-mismatch guard was untested.**
  Added `test_matmul_ikj_shape_mismatch_raises`.
- **[Codex Low / Opus Nit — accepted, partial] Float `==` in tests.** Converted
  computed/stored-float assertions to `assert_almost_equal` (house tolerance
  habit) across the tensor/ops tests. Kept exact `==` only where
  bit-reproducibility is the actual property under test — the RNG determinism
  tests (`test_uniform_deterministic`, `test_normal_deterministic`,
  `test_xavier_deterministic`) — with a comment saying why. Both reviewers
  proposed exactly this split.

No findings were rejected.

---

## Phase 1 — the bigram LM (branch `part-07-bigram`)

_(pending)_
