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
- Never compare floats with `==`. Use `assert_almost_equal` with a tolerance.
- You **cannot transfer a single struct field** out of a live value:
  `s.field^` fails with "destroyed out of the middle of a value". `.copy()` the
  field (or move the whole `s`). Owned (`var`) args likewise need `^`/`.copy()`
  at the call site — `List`/`Dict`/user structs are not `ImplicitlyCopyable`.
- **`Dict` subscript (`d[key]`) raises** on the pinned Mojo, so any function
  that indexes a `Dict` must itself be `raises` — even a lookup guarded by
  `if key in d` that can never actually miss. (Pretrained code and the chapter
  drafts routinely mark such helpers non-raising; they will not compile.)
- **`comptime` evaluates user structs and non-raising methods.** A user struct
  constructs in a comptime context and a non-raising method runs on it at
  compile time (`comptime cfg = GPTConfig.gpt2_124m()` then
  `comptime assert cfg.parameter_count() == N`). This is how a pure arithmetic
  invariant becomes a *build* failure, not just a test failure. Raising
  functions cannot be comptime-evaluated, so the method must be a plain `def`
  with no `raises`.
- **A `comptime assert` only fires where its function is *called*.** An
  uncalled function's `comptime assert` is never evaluated (a deliberately
  false one compiles clean). Give every contract pin a live call site — a test
  that calls it — or it enforces nothing.
- **A comptime value that isn't `ImplicitlyCopyable` can't be read at runtime
  directly.** `comptime T = build_list()` compiles, but a bare use site fails
  with "cannot materialize comptime value ... because it is not
  'ImplicitlyCopyable'". Lift it explicitly with `materialize[T]()` — the build
  runs at compile time, the read copies the frozen result. A bare use is a
  compile error, never a silent copy.

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
pixi run test           # smoke test first, then the whole suite
```

CI runs `pixi run fmt-check` (format then `git diff --exit-code`, so it fails on
a diff instead of editing) and `pixi run test`. Locally you fix formatting with
`pixi run fmt`; never commit code that fails format or tests. Run a single test
directly while iterating:

```bash
pixi run mojo run -I src tests/test_softmax.mojo
```

The `-I src` flag puts the `llm` package on the import path so tests and examples
can `from llm.tensor import matmul`. During development prefer `-I src` against
the source tree (edits take effect immediately); a compiled `build/llm.mojopkg`
is a later, optional distribution step.

## Layout

```text
src/llm/          the library — reusable implementation, one package per concern
  config.mojo       model/training configuration
  vocab.mojo        vocabulary
  tokenizer/        char + BPE tokenizers
  data/             text datasets, batch iteration
  tensor/           Tensor2D/Tensor3D and ops (matmul, softmax, ...)
  nn/               parameter, linear, embedding, norm, activation, mlp
  transformer/      masks, attention, positional, block, model
  models/           standalone models (bigram); not nn layers, not the GPT
  training/         loss, optimizer, trainer, checkpoint
  generation/       sampler, kv_cache, generate
  utils/            random, math, timing, logging
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

```text
utils  →  tensor  →  nn  →  transformer  →  { training, generation }
config + vocab  →  nn, transformer
tokenizer  →  data  →  training
tensor, data  →  models  →  { training, generation }
```

When you add an import that points "up" the graph, stop: the dependency belongs
somewhere else, or the thing you need should move down. See
[improve-architecture](.agents/skills/improve-architecture/SKILL.md).

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
| `training` | `src/llm/training/` (loss, optimizer, trainer) |
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
3. `pixi run fmt` → `pixi run test` → build representative examples.
4. Only then commit. A chapter with unverified `mojo` code blocks is not done.

## Skills index

- [git-conventions](.agents/skills/git-conventions/SKILL.md) — commits, scopes, PRs, no-AI-attribution.
- [mojo-coding-guidance](.agents/skills/mojo-coding-guidance/SKILL.md) — how to write library Mojo here (shapes, modules, docstrings, numerics).
- [test-driven-development](.agents/skills/test-driven-development/SKILL.md) — failing-test-first, TestSuite, oracle/golden/overfit tests.
- [code-review-and-quality](.agents/skills/code-review-and-quality/SKILL.md) — multi-axis review before merge.
- [improve-architecture](.agents/skills/improve-architecture/SKILL.md) — layering, deep modules, refactor plans.
- **`mojo-syntax`** (global skill) — the authority on current Mojo syntax.
