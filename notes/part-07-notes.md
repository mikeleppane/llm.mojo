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

### Rejected review findings

_(none yet — external review pending)_

---

## Phase 1 — the bigram LM (branch `part-07-bigram`)

_(pending)_
