---
name: mojo-coding-guidance
description: Mojo implementation and review guidance for the mojo-llm-from-scratch repo — how to write clear, correct, tested, well-shaped library Mojo for a from-scratch Transformer. Use every time you write, modify, refactor, or review Mojo in this codebase — shape discipline, numerical correctness, docstrings, module boundaries, error handling, and allocation all matter here. Apply on every Mojo edit, not only when the user asks for "clean code". Defers to the global mojo-syntax skill for language syntax and to AGENTS.md for project rules.
---

# Mojo Coding Guidance (mojo-llm-from-scratch)

How to write library Mojo in this repo: modern, clear, correct, tested,
**shape-documented** code that teaches. This is educational code — a reader must
be able to follow the dimensions and the math without running it, and be able to
*trust* it because a test proves it.

## Sources of truth (read these first)

- **Language syntax → the global `mojo-syntax` skill.** Mojo evolves fast and
  pretrained models emit obsolete syntax. That skill is the authority on `def`
  vs `fn`, `comptime` vs `alias`/`@parameter`, argument conventions
  (`read`/`mut`/`var`/`out`/`deinit`), `std.`-prefixed imports, lifecycle
  methods, `Writable`, SIMD, strings, and pointers. **Do not rely on your own
  recollection of Mojo syntax; consult it.** This skill does not restate it.
- **Project rules → [AGENTS.md](../../../AGENTS.md).** The golden rule, the
  syntax contract, the quality floor, the dependency layering, and the
  "Ask first" boundaries. AGENTS.md wins over this skill.
- **When in doubt, compile it.** `pixi run mojo run -I src <file>`. The syntax
  moves; a green build is the only proof a snippet is current.

The rest of this skill is the project's *coding* contract on top of that syntax.

---

## The floor

Before any Mojo change is done:

```bash
pixi run fmt      # mojo format — never hand-format, let the tool decide
pixi run test     # the file you touched must be covered and green
```

`mojo format` is the arbiter of layout: line breaks, import wrapping,
indentation. Don't argue with it in review — if it reformats your code, that's
the house style.

---

## Shape discipline — the most important convention

Every tensor-shaped value carries its shape, and every tensor function states
its input and output shapes. This is non-negotiable in this repo.

Canonical shape letters (keep them consistent across files):

```text
B = batch size        T = sequence length     C = model dimension (d_model)
H = number of heads   D = head dimension       V = vocabulary size
```

File header on every math-heavy file:

```mojo
# Scaled dot-product and multi-head attention.
# Shapes:
#   x:      [B, T, C]
#   q,k,v:  [B, H, T, D]
#   scores: [B, H, T, T]
```

Function contract on every public tensor function — shapes in, shapes out, and
the four facts a caller needs:

```mojo
# Row-wise numerically stable softmax.
# Input:  scores [rows, cols]
# Output: probs  [rows, cols]  (each row sums to ~1)
# Allocates a new tensor; does not mutate the input; does not raise.
def softmax_rows(scores: Tensor2D) -> Tensor2D:
    ...
```

The four facts: **shapes**, whether it **mutates**, whether it **allocates**,
whether it can **raise**. Prefer a real docstring on public APIs (the doc tooling
extracts it) and a shape comment inside dense loops.

Short math symbols (`q`, `k`, `scores`) are fine **only** next to a shape
comment. Everywhere else use descriptive names.

---

## Naming

Clear over clever. Types `UpperCamelCase` (`Tensor2D`, `GPTConfig`); functions,
methods, variables `snake_case` (`softmax_rows`, `d_head`); compile-time
constants `UPPER_SNAKE_CASE` (`comptime CONTEXT_LENGTH = 128`).

| Concept | Name |
|---|---|
| model dimension | `d_model` |
| head dimension | `d_head` |
| number of heads | `n_heads` |
| sequence length | `seq_len` (or `T` in local math) |
| vocabulary size | `vocab_size` |
| context length | `context_length` |
| batch size | `batch_size` |
| learning rate | `learning_rate` |

Never shadow the reserved convention words `ref`, `mut`, `out`, `deinit`,
`read`, `var` — not as parameter names, not as locals. Rename to `reference`,
`expected`, etc. (See the `mojo-syntax` skill for why.)

---

## Numerical correctness

This is a numerics project; the subtle bugs are numerical.

- **Never compare floats with `==`.** Use `assert_almost_equal` with a tolerance
  sized to the precision: `1e-9`–`1e-12` for `Float64` reference math,
  `1e-4`–`1e-5` for `Float32` model math. (See
  [test-driven-development](../test-driven-development/SKILL.md).)
- **Softmax and log-sum-exp are computed stably** — subtract the row max before
  `exp`. A "correct" naive softmax overflows on real logits; the stable form is
  the reference implementation, not an optimization.
- **Be explicit about dtype.** Mojo does not implicitly convert between numeric
  *variables* — write `Float32(my_int) * scale`, `Int(my_uint)`. Literals are
  polymorphic and adapt to context (`var a: Float32 = 0.5`), so don't wrap them.
  Decide the model's working dtype (`Float32`) and the reference dtype
  (`Float64`) deliberately and keep tests aware of which they compare against.
- **Guard divisions and logs.** Add the epsilon where the math needs it (layer
  norm denominator, cross-entropy `log`), and document the epsilon's value and
  why — it is exactly the kind of magic constant AGENTS.md forbids leaving
  unexplained.

---

## Error handling

- `def` does **not** imply `raises`. Add `raises` explicitly when a function can
  fail, and omit it (compiler-enforced) when it cannot — the signature is a
  contract the reader relies on.
- **Validate shapes at the boundary and raise a clear error**, don't let a
  mismatch become an out-of-bounds crash deep in a loop. "matmul: inner
  dimensions 64 and 32 disagree" teaches; a segfault does not.
- Invalid configuration raises at construction (`GPTConfig` with
  `d_model % n_heads != 0`), so the failure is early and named. Cover these with
  `assert_raises` tests.
- Use `comptime assert` for invariants knowable at compile time (place it inside
  a function body — it is illegal at module/struct scope).

---

## Memory and allocation

- **Say whether a function allocates** in its docstring. Allocation is the main
  cost in the hot path; a reader tracking performance needs to know without
  reading the body.
- Prefer the safe types (`List`, `Span`, `Pointer`, `OwnedPointer`,
  `ArcPointer`) over raw `UnsafePointer`. When a tensor genuinely owns heap data
  via `UnsafePointer`, give the field an explicit origin
  (`UnsafePointer[Self.T, MutUntrackedOrigin]`), allocate with `alloc[T](n)`, and
  free it in `__del__`. A tensor that allocates must have a destructor that frees
  — prove the round-trip with a test that constructs and drops many.
- Reach for SIMD in the ops layer *after* a scalar version is correct and tested,
  never before. The scalar version stays as the reference the SIMD version is
  checked against.

---

## Module boundaries

- One responsibility per file. When a file starts owning two things (attention
  *and* masks), split it — the split usually reveals a hidden dependency you can
  now make explicit. Detailed guidance and refactor planning live in
  [improve-architecture](../improve-architecture/SKILL.md).
- **Dependencies flow one direction** (the authoritative graph is in AGENTS.md):

  ```text
  utils  →  tensor  →  nn  →  transformer  →  { training, generation }
  config + vocab  →  nn, transformer
  tokenizer  →  data  →  training
  ```

  Never import "up" the graph. If you need something from a higher layer, it
  belongs lower.
- **`__init__.mojo` is the package's public surface** — re-export the clean names
  (`from .ops import matmul, softmax_rows`) so callers write
  `from llm.tensor import matmul`, and you can move files inside the package
  without breaking them. `__init__.mojo` holds no executable top-level code.
- Keep the public surface small; a struct's fields are an implementation detail —
  expose methods, not raw internals, where it aids the reader.

---

## Language gotchas that bite in this repo

These are the current-Mojo footguns most likely to hit numeric/collection code.
The `mojo-syntax` skill has the full list; these are the ones that recur here:

- **`List` has no variadic constructor.** Use bracket literals: `var v = [1, 2, 3]`,
  `var w: List[Float32] = [1.0, 2.0]`. `List[Int](1, 2, 3)` does not compile.
- **`List[T]` rejects negative indices** at compile time — use `lst[len(lst) - 1]`,
  not `lst[-1]`.
- **String is UTF-8 and byte-indexed.** `len(s)` is deprecated on `String` —
  use `s.byte_length()` or `s.count_codepoints()` (they differ for non-ASCII).
  Index with `s[byte=i]` (returns a `StringSlice`; wrap in `String(...)` to
  own). No `s[0:10]` slice syntax. Rarely needed in the math layers, common in
  the tokenizer.
- **Explicit copy/transfer** for non-`ImplicitlyCopyable` types (`List`, `Dict`,
  most user structs): `d.copy()` or transfer with `^`. `return my_struct` errors
  until you transfer or add the conformance.
- **Self-qualify struct parameters** inside a struct: `var data: Self.T`, not
  `var data: T`.
- **Iterate `Dict` entries directly**: `for e in d.items(): print(e.key, e.value)`
  — no `[]` deref.

---

## Review checklist for a Mojo change

- [ ] `pixi run fmt` clean, `pixi run test` green
- [ ] Every tensor function states input/output shapes + mutate/allocate/raise
- [ ] Short symbols only next to a shape comment; else descriptive names
- [ ] No float `==`; tolerances match the precision
- [ ] Softmax / log-sum-exp / norm computed stably; epsilons documented
- [ ] `raises` present iff the function can raise; shape mismatches raise clearly
- [ ] Allocating functions say so; `UnsafePointer` owners free in `__del__`
- [ ] Imports point down the dependency graph; `__init__.mojo` re-exports the surface
- [ ] Syntax matches the `mojo-syntax` skill (no `fn`/`let`/`alias`/`@parameter`)
- [ ] New behavior has a test that would fail without it
