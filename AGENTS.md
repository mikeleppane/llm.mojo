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
  This includes a returned struct's field: `list.append(result.output^)` on an
  `AttentionResult` fails; use `result.output.copy()`.
- **Binding a `List[T]` element to a local copies it** when `T` is `Copyable`
  but not `ImplicitlyCopyable` (`Tensor2D`, user structs): `var part = parts[i]`
  fails with "cannot be implicitly copied". Read the scalar fields you need off
  `parts[i]` and subscript `parts[i][...]` directly, or `.copy()` when you truly
  need to own the element.
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
- **`std.math` transcendental/root functions run in a comptime context.** Not
  just exact integer arithmetic — `comptime SQRT_2_OVER_PI = sqrt(2.0 / pi)`
  compiles and is bit-identical to the hand-spelled literal (verified with a
  `comptime assert` against `0.7978845608028654`). So a derived numeric constant
  whose derivation is its documentation can be bound at compile time, not only a
  literal. A constant that is *fitted*, not derived (e.g. GELU's `0.044715`), has
  no expression to bind — name and cite it instead.
- **A struct constructor that takes `var value` and needs the value's shape must
  read the shape *before* the move.** `self.grad = zeros_2d(value.rows,
  value.cols)` then `self.value = value^` works; reordering (move first, read
  after) uses a destroyed value. This is the ownership shape of every layer
  factory here (`Parameter`, and the `init_random`/`init_default` factories that
  build tensors then hand them to `Parameter`).
- **A gradient that must accumulate to an *exact* double needs one `+=` per
  backward call, not a running `+=` inside a loop.** Backward accumulates into
  `Parameter.grad` (`+=`, never `=`) so two paths through one Parameter sum — and
  a per-layer test pins that two backward passes yield bit-for-bit `2×` the
  grads. That exact equality is real only if each call adds a single fully-formed
  delta: a running `grad[j] += …` *inside* a per-row/element loop interleaves the
  second call's partial sums with the first call's stored result, which rounds
  differently than `2·grad1`. So accumulate the call's contribution into locals,
  then add the finished delta to `grad` once (LayerNorm's `dγ`/`dβ`). Grads formed
  by one matmul and added once (Linear's `dW`) already satisfy this. A gradient
  can be numerically *correct* and still fail the doubling test — that is the test
  earning its keep, since a doubling off by ulps is a future weight-tying bug.
- **A Parameter fed by *two* paths in one backward call must sum them into one
  delta before the single `+=`.** The weight-tying case the rule above warned
  about: `GPT`'s tied token table gets gradient from the head matmul *and* the
  embedding gather in the same `backward`. Adding them as two separate `+=` makes
  two calls accumulate as `((h+g)+h)+g`, which is not bit-identical to `2·(h+g)`
  (float addition is not associative), so the exact-doubling test fails. Combine
  the paths into one `[V, C]` delta, then add it to `grad` once — one fully-formed
  `+=` per call, per the rule above. The gradient *value* is the same either way;
  only the doubling distinguishes them, which is why the doubling test is separate
  from the finite-diff.
- **A temporary cannot bind to a `mut` argument.** `f(mut rng: Rng)` called as
  `f(Rng(0))` fails — an rvalue has no mutable storage to borrow. Bind a named
  `var rng = Rng(0)` and pass that, even when the callee will not actually mutate
  it on this path (e.g. an eval-mode `forward_cached` that draws no rng but whose
  signature still takes `mut rng`).
- **Bind a temporary's non-`ImplicitlyCopyable` field to a local before using
  it.** `some_call(...).cache` where `cache: AttentionCache` errors ("cannot be
  implicitly copied … consider transferring with `^`"), because the temporary is
  destroyed and the field can't be transferred out of it. Bind the whole result
  first (`var fwd = some_call(...); fwd.cache`), which borrows the field. Same
  family as the single-field-transfer rule above.
- **`mojo format` rejects a plain reassignment to a local named `out`** — `out =
  expr` fails with `Cannot parse` even though the compiler accepts it, because
  `out` is the argument-convention keyword and the formatter's parser reads
  `out = …` as a malformed signature. `out += …`, `out[i] = …`, and `return out`
  all format fine; only the bare reassignment trips it. A file can compile and
  pass `pixi run test` yet fail `pixi run fmt-check` for this alone. Don't name a
  reassigned local `out` (use `acc`/`result`); the same guard applies to the
  other convention keywords (`mut`, `read`, `owned`, `ref`, `deinit`). Reserved
  words also surprise as ordinary identifiers: `ref` cannot be a variable name at
  all (`unexpected token in expression`).
- **A Float64 ↔ its IEEE-754 bit pattern:** `x.to_bits[DType.uint64]()` and
  `bitcast[DType.float64, 1](SIMD[DType.uint64, 1](bits))[0]` (import `bitcast`
  from `std.memory`). `Float64.from_bits` does not exist on the pinned Mojo. This
  is how the checkpoint stores exact values (a hex UInt64 per Float64) so a resume
  round-trips bit-for-bit instead of through a re-rounded decimal.
- **`String.split(sep)` yields `StringSlice`, not `String`**, so a helper typed
  `List[String]` needs an explicit `String(slice)` per element; and a
  newline-terminated file always leaves a trailing empty slice, which a parser
  must drop (or it surfaces as a spurious blank-line error rather than the real
  "truncated file").
- **Default `List[T]()` arguments are legal** (`init_m: List[Tensor2D] =
  List[Tensor2D]()`), the clean way to make an optional list parameter without an
  overload.

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
- **Don't run known-slow files in the red/green loop.** `SKIP_SLOW=1 pixi run
  test` (aka `pixi run test-fast`) skips the files listed in `SLOW_6554` in
  `scripts/test_all.sh` and still runs everything else; the default `pixi run
  test` keeps full coverage for CI. Run a skipped file standalone only when you're
  actually changing it: `pixi run mojo run --no-optimization -I build tests/<file>`.
- When you add a file that trips the stall, append it to `SLOW_6554` and note the
  standalone run time, so the loop stays fast for the next part.

The standing `SLOW_6554` member is **`tests/test_seq_tasks.mojo`** (a Part XII lab
test), and its #6554 stall is severe enough to effectively **hang** a run, not just
cost a slow minute — so in practice **never invoke it**, and take the pre-merge
green gate with `pixi run test-fast` (`SKIP_SLOW=1`), which runs the whole suite
except that one file. "The suite is green" always means green with
`test_seq_tasks` excluded; do not try to fix, delete, or wait it out (it is a
frozen lab layer anyway).

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
