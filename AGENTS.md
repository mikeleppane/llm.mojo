# AGENTS.md — mojo-llm-from-scratch

Build a small decoder-only Transformer language model **from scratch in Mojo** —
tokenizer, tensors, attention, training loop, and generation, with no ML
framework underneath. The repo is the companion codebase to a written guide (an
Obsidian/GitHub blog series), so it is read at least as often as it is run.

This file is the **source of truth** for how to work in this repo. The skills
under [`.agents/skills/`](.agents/skills/) go deeper on specific moments (git,
coding, tests, review, architecture); when a skill and this file disagree, **this
file wins**, and the skill should be updated to match.

## Golden rule: this is teaching code

The product is **understanding you can trust**. Every rule below serves that.
When two options tie, pick the one a reader learns more from and can verify:

- **Correctness before speed.** A clear, correct, tested implementation lands
  first. Optimizations come later, behind a benchmark and a passing test, and
  never by making the code unreadable.
- **Every shape is documented.** Tensor code carries `# [B, T, C]`-style shape
  comments. A reader must be able to follow the dimensions without running it.
- **Every module, struct, and public function has a Google-style docstring.**
  Triple-quoted, short (what it does and why), with `Args:` / `Returns:` /
  `Raises:` folding in the four facts a caller needs (shapes, mutates, allocates,
  raises). Not `#`-comment doc blocks. The full format and a worked example are in
  [mojo-coding-guidance](.agents/skills/mojo-coding-guidance/SKILL.md#docstrings--google-style-triple-quoted-mandatory).
- **If it matters, it is tested.** Correctness is checkable, not asserted in
  prose. See [test-driven-development](.agents/skills/test-driven-development/SKILL.md).
- **No magic.** No undocumented constants, no unexplained clever tricks, no
  copying PyTorch architecture without explaining the shapes.

## Mojo, not Python

Mojo evolves fast and pretrained models emit obsolete syntax. **The global
`mojo-syntax` skill is the authority on Mojo syntax** — consult it before writing
or reviewing any Mojo, and prefer it over your own recollection. Do not restate
its rules here; this section only names the project's standing consequences of it.

The **syntax contract** for all runnable Mojo in this repo (a chapter is not
publishable until its code passes these — see *Publishing* below):

- `def`, never `fn`. Add `raises` explicitly when a function can raise.
- `comptime`, never `alias` or `@parameter` (for constants, type aliases,
  compile-time branches/loops).
- `var`, never `let`. Argument conventions are `read` (default) / `mut` / `var` /
  `out` / `deinit` — never `inout` / `owned` / `borrowed`.
- Imports are `std.`-prefixed: `from std.testing import ...`, not
  `from testing import ...`. Prelude types (`Int`, `String`, `List`, …) need no
  import.
- Tests use `TestSuite` discovery run with `mojo run` — the `mojo test`
  subcommand was removed.
- No stdlib `Tensor[T]`. This project defines its own tensors under
  `src/llm/tensor/`.
- Never compare floats with `==` **for a numerical result** — use
  `assert_almost_equal` with a dtype-sized tolerance. Exact/bitwise equality is
  correct only where *exactness itself is the contract*: deterministic RNG
  replay, a checkpoint bit-for-bit round trip, the gradient-doubling test (two
  backward passes give exactly `2×`), zero-preservation (a masked position is
  exactly `0.0`), and integer-valued counts. In those cases assert exact
  equality on purpose and say why — do not "fix" such a test by loosening it to
  a tolerance (that hides the very regression it guards). See
  [test-driven-development](.agents/skills/test-driven-development/SKILL.md).

Beyond this contract, the pinned toolchain (`1.0.0b2`) has a set of
compiler/formatter/stdlib gotchas that have each cost real debugging time. They
are **catalogued in [docs/mojo-1.0.0b2-notes.md](docs/mojo-1.0.0b2-notes.md)** —
consult it before non-trivial Mojo. The categories, so you know whether your task
is affected without opening the file:

- **Ownership, copies, transfers** — you can't `^` a single field out of a live
  value; binding a `List[T]` element copies it; read a `var value`'s shape before
  moving it; a temporary can't bind to a `mut` arg or have its field used
  in-place.
- **`comptime`** — it evaluates user structs and non-raising methods (how a
  parameter-count invariant becomes a *build* failure); a `comptime assert` only
  fires where its function is called; a non-`ImplicitlyCopyable` comptime value
  needs `materialize[T]()`; `std.math` transcendentals are comptime-legal.
- **Raising** — `Dict` subscript raises, so any `Dict`-indexing helper must be
  `raises` even when the key can't miss.
- **Exact-gradient accumulation** — one fully-formed `+=` per backward call (not
  a running `+=` in a loop), and sum a two-path Parameter into one delta, or the
  bit-exact gradient-doubling test fails.
- **Formatter and reserved words** — `mojo format` rejects a bare reassignment to
  a local named `out` (or any convention keyword); `ref` can't be a variable name
  at all. A file can pass `test-fast` yet fail `fmt-check` for this alone.
- **Binary and string I/O** — the `Float64 ↔ bits` bitcast idiom (no
  `Float64.from_bits`); `String.split` yields `StringSlice` + a trailing empty
  slice; there is no `"rb"` open mode (use `read_bytes()`), and don't materialize
  a ~500 MB payload as one `String`.
- **SIMD, pointers, parallelism** — `List[Float64].unsafe_ptr()` +
  `ptr.load[width=W]`/`.store`/`.reduce_add` (no storage redesign needed);
  `simd_width_of` from `std.sys`; `parallelize` wants an `@parameter def(Int)`
  worker and manual block partitioning; write parallel output through a raw
  pointer, not a `mut` capture; `memcpy` is keyword-only; multi-accumulator SIMD
  reassociates (classify Class A vs B), SIMD-over-columns does not.

## Toolchain and the quality floor

Everything runs through **pixi** (see [pixi.toml](pixi.toml)). There is no `make`.

```bash
pixi install            # set up the environment from pixi.lock
pixi run mojo-version   # print the pinned Mojo version
pixi run hello          # examples/hello.mojo
```

**The floor before you call any change done** — run in this order, all green:

```bash
pixi run fmt            # mojo format (rewrites in place)
pixi run test           # the canonical green gate (see below)
```

`pixi run test` is **the** gate — locally and in CI. It runs the smoke test
first, then every test *except* the files in `SLOW_6554` (currently only
`tests/test_seq_tasks.mojo`), which trip Mojo #6554 and can hang a run for
minutes; it prints a loud `SKIPPED (Mojo #6554)` line per excluded file every
run, so the exclusion is visible, never silent. "The suite is green" always means
green under `pixi run test`. `pixi run test-fast` is a retained alias of it (same
exclusion), kept only so older docs and muscle memory don't break. `pixi run
test-full` (`RUN_SLOW=1`) is the strict superset that *also* runs the SLOW_6554
files; it is **not** the gate (it can hang) — its one job is the
toolchain-upgrade check (see "The #6554 compile stall" below). The one currently
excluded file is a Part XII lab test; the lab is a frozen teaching layer, so this
is a known, accepted coverage gap rather than missing coverage of the main line.

**Two validation tiers, and the merge gate.** `pixi run test` is *Tier 1*: the
hermetic doll-house suite — no weights, no network, runs on every change and in
CI. It proves the *code*. It cannot prove the *model*: bugs that surface only on
real weight distributions and real token statistics (a BPE edge case, a numerical
drift a doll-house tensor never accumulates, an off-by-one at context position
1023) need the real 124M weights, which will never live in git or CI. That is
*Tier 2*: `pixi run gauntlet` runs `examples/gpt2_gauntlet.mojo` against
`checkpoints/gpt2-124m.bin` (failing with a converter-pointing error if the
weights are absent), checking a curated multi-prompt set against frozen float64
goldens. It runs locally in minutes. **From Part XIX onward a change is mergeable
to main when Tier 1 is green, `pixi run gauntlet` is green, and `pixi run
fmt-check` passes** — re-run `test` and `gauntlet` on `main` after the merge
commit. `pixi run build-examples` (compile-check every example and benchmark, no
weights) rounds out the floor: examples are the guide's artifacts and bit-rot
silently otherwise.

CI runs `pixi run fmt-check` (format then `git diff --exit-code`, so it fails on
a diff instead of editing), `pixi run test`, and `pixi run build-examples`. It
does **not** run the gauntlet (that needs the weights). Locally you fix formatting
with `pixi run fmt`; never commit code that fails format or tests. Run a single
test directly while iterating:

```bash
pixi run mojo run -I src tests/test_softmax.mojo
```

The `-I src` flag puts the `llm` package on the import path so tests and examples
can `from llm.tensor import matmul`. During development `-I src` against the
source tree is convenient (edits take effect immediately), but it is **slow at
scale**: `mojo run -I src tests/X.mojo` recompiles *and re-optimizes at `-O` the
whole `llm` source tree, inlined into each test binary*, every run. For a test
that pulls the whole model into one function (a training loop, an end-to-end
finite-difference check) LLVM can grind on the giant monomorphized functions for
**minutes per file** — this is the dominant cost, not your CPU (the compiler is
single-threaded on that phase; the machine is fine).

**Precompile the package to make the suite fast.** `scripts/test_all.sh` builds
`build/llm.mojopkg` once (~1 s) and runs every test with `-I build`, so a test
compiles only its own small file against a *binary* dependency and the library's
optimizer passes run once, not per test — the suite drops from tens of minutes to
a couple. When iterating on one test by hand, do the same:

```bash
pixi run mojo precompile src/llm -o build/llm.mojopkg   # after any src/ change
pixi run mojo run -I build tests/test_softmax.mojo      # ~seconds, optimized
```

Rebuild the package after editing anything under `src/`; forget to, and tests run
against stale library code. Two operational gotchas learned the hard way:
**never SIGKILL a `mojo` compile mid-flight** (the module cache is only written on
a clean exit, so the next run pays full cost again), and **don't stack concurrent
cold compiles** — the editor's LSP already runs a background `mojo` on every save,
so a couple is fine but five will starve each other.

**The #6554 compile stall, and how to keep the TDD loop fast.** Even against the
precompiled package a few test files spend *minutes* in the compiler before a
sub-second run. Root cause is Mojo #6554: `TestSuite`'s comptime `discover_tests`
builds a thin-function-pointer dispatch table over the module's functions, and
the compile cost of that table balloons with the function count (list-literal
initializers make it worse). It is not your machine and not `-O` — it reproduces
at `-O0` on a precompiled package. Mitigations, in order of preference:

- **Keep a test module's function count low.** This is the real lever. If a file
  starts stalling, split it into smaller test files (each gets a smaller table)
  rather than piling more `def test_*` into one module. Prefer append-helper
  builders over long inline `[a, b, …]` `List[Int]` literals.
- **The default gate skips them.** `pixi run test` skips the files listed in
  `SLOW_6554` in `scripts/test_all.sh` and runs everything else, printing a loud
  `SKIPPED (Mojo #6554)` line per excluded file — this is the canonical gate,
  locally and in CI. `pixi run test-full` (`RUN_SLOW=1`) is the superset that also
  runs the SLOW_6554 files; run it (or better, the single file standalone) only
  when you're actually changing one:
  `pixi run mojo run --no-optimization -I build tests/<file>`.
- When you add a file that trips the stall, append it to `SLOW_6554` (its ONE
  home) and note the standalone run time, so the loop stays fast for the next part.

**Toolchain-upgrade trigger.** The `SLOW_6554` exclusion is a workaround for an
upstream compiler bug, not a permanent fact. When you bump the pinned Mojo version
(`pixi.toml` / `pixi.lock`), run `pixi run test-full` **once**: if #6554 is fixed
upstream the slow files now run in normal time — retire the exclusion list
entirely (empty `SLOW_6554`, drop `test-full` and this trigger). Until then,
`test-full` exists only for this check; running it in a normal loop will hit the
very stall it isolates.

The standing `SLOW_6554` member is **`tests/test_seq_tasks.mojo`** (a Part XII lab
test), and its #6554 stall is severe enough to effectively **hang** a run, not just
cost a slow minute — so in practice **never invoke it** (not even via `test-full`,
unless you are running the toolchain-upgrade check above), and take the pre-merge
green gate with `pixi run test`, which runs the whole suite except that one file
(this is also what CI runs). "The suite is green" always means green with
`test_seq_tasks` excluded; do not try to fix, delete, or wait it out (it is a
frozen lab layer anyway). Because the lab is quarantined off the main line, this
one excluded file is an accepted coverage gap, not a hole in the model's tests.

## Layout

```text
src/llm/          the library — reusable implementation, one package per concern
  config.mojo       model/training configuration
  vocab.mojo        toy whitespace vocabulary (an early chapter's example)
  tokenizer/        char + BPE + GPT-2 tokenizers
  data/             text datasets, batch iteration
  tensor/           Tensor2D/Tensor3D and ops (matmul, softmax, ...)
  nn/               parameter, linear, embedding, norm, activation, mlp, optim (AdamW math)
  transformer/      masks, attention, positional, block, gpt, gpt2_weights (loader)
  models/           standalone models (bigram); not nn layers, not the GPT
  training/         loss, optimizer, schedule, trainer, checkpoint
  generation/       sampler, generate  (KV cache is Part XVII — not yet built)
  lab/              Part-XII encoder-decoder lab — quarantined off the main line
  utils/            random (seeded RNG), timing
examples/         runnable demonstrations (not core logic)
tests/            correctness checks (TestSuite)
benchmarks/       performance measurement (after correctness)
docs/             the written guide chapters
  plans/            internal refactor plans (working documents, never published)
scripts/          test_all.sh and other dev scripts
checkpoints/      generated artifacts — gitignored, never committed
build/            compiled .mojopkg output — gitignored
```

Every importable package directory needs an `__init__.mojo`; without it the
import fails. `__init__.mojo` re-exports the package's public surface and holds
no executable top-level code — treat it as the package's table of contents.

**Where code belongs:** reusable implementation → `src/llm/`; teaches usage →
`examples/`; proves correctness → `tests/`; measures performance →
`benchmarks/`. A snippet in a notebook is a draft until it lands in one of those.

## Dependency layering — one direction only

Dependencies flow from foundational to high-level. **Lower layers never import
higher layers.** A cycle (e.g. `utils` importing `transformer`) makes code
impossible to test in isolation — treat one as a bug.

**This is the authoritative allowed-dependency graph.** Every arrow is an
*allowed import direction* — a package may import from lower layers, never from a
higher one. It reflects the actual import edges in `src/llm/` (grep the
`from llm.` lines to check). README, ARCHITECTURE.md, and the skills all defer to
this graph rather than restating it.

```text
Layer 0  utils   config   vocab*   tokenizer*      (no internal imports)
Layer 1  tensor (uses utils)        data (uses utils)
Layer 2  nn (uses tensor)           models (uses tensor, data)
Layer 3  transformer (uses nn, config)
Layer 4  training (uses transformer, models, data, config)   generation (uses transformer)
Layer 5  lab (uses transformer, training, nn)   — quarantined Part-XII encoder-decoder
```

Notes on the graph:

- **`config`** feeds `transformer` and `training` (not `nn`). **`utils`** and
  **`tensor`** are the widely-shared foundation.
- **`vocab`** and **`tokenizer`** are self-contained today (`*`): nothing under
  `src/` imports them yet — each is exercised only by its own test/example until
  the guide reaches the chapter that consumes it. They are still Layer 0 (they
  import nothing internal).
- **`lab`** is a frozen teaching layer that sits *above* `training`; nothing
  imports `lab`. Keeping it at the top is what lets it be quarantined.

When you add an import that points "up" the graph, stop: the dependency belongs
somewhere else, or the thing you need should move down. See
[improve-architecture](.agents/skills/improve-architecture/SKILL.md).

**The `GPT` parameter walk is a load-bearing contract.** With no framework
parameter dict, the model's fixed traversal order (wte once, wpe, each block's 12
parameters in layer order, ln_f) IS the registry: the optimizer state (m/v),
gradient clipping, the checkpoint format, and export/import all index that ONE
order. Every walk method (`parameter_shapes`, `parameter_decay_flags`,
`grad_norm`, `scale_grads`, `export/import_parameters`, `export_gradients`,
`apply_adamw`, `zero_grad`, `apply_sgd`) must visit the same parameters in the
same order, wte exactly once (weight tying → one Parameter). The per-block 12 live
once in `transformer/block.mojo` so the order is authored in a single place; order
drift between methods is the named failure mode, guarded by the shape/count
reconciliation, the decay-partition inventory, the checkpoint round-trip, and the
against-oracle optimizer run. Optimizer *math* lives in `nn/optim.mojo`
(Parameter-level, layering-legal for `transformer/` to call); optimizer *state* is
trainer-owned, never stored in `Parameter`.

## Testing

The pyramid: many cheap unit tests (tensors, ops, masks, config), some component
tests (attention/block forward), a few integration tests (overfit-one-batch,
generation smoke). The highest-value integration test is **overfit-one-batch** —
a correct model + loop drives the loss on one small batch to near zero; if it
can't, the loss, gradient, or optimizer is broken.

Tolerances: `1e-9`–`1e-12` for `Float64` reference math, `1e-4`–`1e-5` for
`Float32` model math. Details and patterns (oracle, golden, finite-difference
gradient checks) live in
[test-driven-development](.agents/skills/test-driven-development/SKILL.md).

**`TestSuite`'s per-test timing display is unreliable on the pinned Mojo
(1.0.0b2)** — the bracketed `[70.977]`/`[309.171]` numbers can be off by orders
of magnitude from the real cost (a whole file that runs in ~4 s of `time`
wall-clock may print hundreds of "seconds"). Don't diagnose a "slow test" from
those numbers; measure with `time pixi run mojo run -I src tests/<file>.mojo`.

**The validation gauntlet (Tier 2) and the golden lifecycle.** The suite is
hermetic and doll-house-scale; it structurally cannot catch a bug that needs real
weights. `examples/gpt2_gauntlet.mojo` (`pixi run gauntlet`) closes that gap: it
runs a curated multi-prompt set (`data/gauntlet/prompts.txt`, spanning unicode,
code, punctuation, the 1024-token boundary) against frozen float64 goldens
(`data/gauntlet/goldens.txt`), checking tokenization / argmax / top-5 EXACTLY and
probe logits / mean NLL at `1e-6`. Cross-implementation checks stay at the logit
level with tolerance; token-*sequence* exactness is only ever our-vs-our
(`generate` vs `generate_cached`), so no genuine near-tie can flake the gate. The
goldens are generated by `scripts/gpt2_gauntlet_reference.py` (NumPy f64, reusing
the existing tokenizer and forward oracles) and carry a `sha256=<hash>` header
pinning the exact `.bin` that produced them.

The lifecycle is doctrine, mirroring the numerical-edge-case policy: **a red
gauntlet after a code change indicts THE CHANGE, not the goldens.** Re-pinning
`goldens.txt` is legitimate only with documented evidence in the part's notes —
either the oracle side changed (a new `.bin` or a converter fix, made visible by
the sha256 header) or a near-tie logit delta at ~`1e-13` scale is shown. "The new
number looks close enough" is never evidence. Goldens regenerate ONLY via the
script, never by hand. When a later part finetunes, the BASE model's gauntlet must
still pass on `main`; finetuned weights get their OWN artifacts and never overwrite
the base goldens.

## Commits

Conventional Commits with a **required scope**, atomic, imperative subject ≤72
chars, a body explaining *why*. **No AI/assistant attribution anywhere** — no
`Co-Authored-By` for an AI, no "Generated with" line, no 🤖. The full rules are in
[git-conventions](.agents/skills/git-conventions/SKILL.md).

**No internal-plan references anywhere** — not in commit messages, PR bodies,
docstrings, or code comments. The plans under `docs/plans/` are gitignored and
unpublished, so `plan D3`, `decision D4`, `§5`, `per the plan`, and the like
dangle for anyone reading the repo. State the reason itself, not the document
that recorded it. External prior art (`minbpe-style`, `nanoGPT`, a paper) is
fine — the ban is on this repo's private planning artifacts. (See
[git-conventions](.agents/skills/git-conventions/SKILL.md).)

**Scope vocabulary** (this list is authoritative; keep it in sync with the layout
as modules emerge):

| Scope | Area |
| ----- | ---- |
| `scaffold` | repo skeleton, dirs, pixi/tooling bootstrap |
| `config` | `src/llm/config.mojo` |
| `vocab` | `src/llm/vocab.mojo` |
| `tokenizer` | `src/llm/tokenizer/` |
| `data` | `src/llm/data/` |
| `tensor` | `src/llm/tensor/` |
| `nn` | `src/llm/nn/` |
| `transformer` | `src/llm/transformer/` |
| `models` | `src/llm/models/` (standalone models, e.g. bigram) |
| `lab` | `src/llm/lab/` (Part-XII encoder-decoder lab, quarantined) |
| `training` | `src/llm/training/` (loss, optimizer, schedule, trainer) |
| `checkpoint` | checkpoint save/load format |
| `generation` | `src/llm/generation/` |
| `utils` | `src/llm/utils/` |
| `examples` | `examples/` |
| `bench` | `benchmarks/` |
| `test` | test infrastructure (`scripts/test_all.sh`, shared helpers) — a module's own tests use type `test` + that module's scope, e.g. `test(tensor): …` |
| `docs` | guide chapters, README, docstrings |
| `build` | `pixi.toml`, `pixi.lock`, packaging |
| `ci` | `.github/workflows/` |
| `skills` | `.agents/skills/` |

Use a module's own scope when a change touches one module. Reserve broad edits
for when they genuinely span the spine.

**Grammar: the *type* is one of the fixed set** (`feat`, `fix`, `refactor`,
`perf`, `docs`, `test`, `bench`, `build`, `ci`, `chore` — see
[git-conventions](.agents/skills/git-conventions/SKILL.md)). `examples`,
`skills`, and `lab` are **scopes, not types** — so a docs change to a skill is
`docs(skills): …`, an example gains a feature as `feat(examples): …`, and lab
code is refactored as `refactor(lab): …`. Never use `examples` or `skills` as a
leading type. **Merge commits** are exempt from the `type(scope): subject`
grammar — keep the default `Merge …` subject (or a short descriptive one).

## Ask first (do not do these silently)

Stop and confirm before:

- **Changing the on-disk checkpoint format** (field layout, version integer,
  dtype). Old checkpoints must stay loadable or fail with a clear error — never
  read garbage. Version the format and store shapes with tensors.
- **Bumping the pinned Mojo version** in `pixi.toml` / `pixi.lock`. CI must match
  local; a bump can silently change syntax that's valid.
- **Adding a runtime dependency** or reaching for Python interop where native
  Mojo would do.
- **Weakening a test tolerance** to make a failing test pass, or deleting/skipping
  a test to get to green.
- **Changing a public API** used by examples/tests (rename, signature change) —
  use a `!` commit and a `BREAKING CHANGE:` footer.

## Publishing (guide chapters)

This repo backs a written series, so an extra gate applies to guide code:

1. Write the chapter; audit every Mojo snippet against the syntax contract above
   and the `mojo-syntax` skill. Mark pseudocode as ```text```, not ```mojo```.
2. Extract runnable code into `examples/` and `tests/`.
3. `pixi run fmt` → `pixi run test` → `pixi run build-examples` (and `pixi run
   gauntlet` when the change could affect the 124M forward).
4. Only then commit. A chapter with unverified `mojo` code blocks is not done.

## Skills index

- [git-conventions](.agents/skills/git-conventions/SKILL.md) — commits, scopes, PRs, no-AI-attribution.
- [mojo-coding-guidance](.agents/skills/mojo-coding-guidance/SKILL.md) — how to write library Mojo here (shapes, modules, docstrings, numerics).
- [test-driven-development](.agents/skills/test-driven-development/SKILL.md) — failing-test-first, TestSuite, oracle/golden/overfit tests.
- [code-review-and-quality](.agents/skills/code-review-and-quality/SKILL.md) — multi-axis review before merge.
- [improve-architecture](.agents/skills/improve-architecture/SKILL.md) — layering, deep modules, refactor plans.
- **`mojo-syntax`** (global skill) — the authority on current Mojo syntax.
