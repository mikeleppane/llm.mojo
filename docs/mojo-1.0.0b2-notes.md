# Pinned-Mojo notes (1.0.0b2)

Compiler, formatter, and standard-library gotchas that have each cost real
debugging time in this repo. They are **specific to the pinned toolchain**
(`mojo ==1.0.0b2`, see [pixi.toml](../pixi.toml)); a version bump can change or
retire any of them, so re-verify against a green build after bumping the pin.

This file is the long tail of the syntax contract in
[AGENTS.md](../AGENTS.md#mojo-not-python). The short, non-negotiable rules
(`def`/`var`/`comptime`, `std.`-prefixed imports, no float `==` for numerical
results, …) live there; the notes below are the "why did this not compile /
format / round-trip" catalogue. **Consult this file before non-trivial Mojo** —
ownership-heavy code, anything using `comptime`, a hand-written backward pass, or
binary/serialization I/O. When in doubt, compile it: a green build under the pin
is the only proof a snippet is current.

The global `mojo-syntax` skill remains the authority on *current* Mojo syntax;
where it and this file disagree, the **pinned compiler wins** for this repo,
because it is what CI runs.

---

## Ownership, copies, and transfers

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
- **A struct constructor that takes `var value` and needs the value's shape must
  read the shape *before* the move.** `self.grad = zeros_2d(value.rows,
  value.cols)` then `self.value = value^` works; reordering (move first, read
  after) uses a destroyed value. This is the ownership shape of every layer
  factory here (`Parameter`, and the `init_random`/`init_default` factories that
  build tensors then hand them to `Parameter`).
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
- **Default `List[T]()` arguments are legal** (`init_m: List[Tensor2D] =
  List[Tensor2D]()`), the clean way to make an optional list parameter without an
  overload.

## Compile-time evaluation (`comptime`)

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

## Raising

- **`Dict` subscript (`d[key]`) raises** on the pinned Mojo, so any function
  that indexes a `Dict` must itself be `raises` — even a lookup guarded by
  `if key in d` that can never actually miss. (Pretrained code and the chapter
  drafts routinely mark such helpers non-raising; they will not compile.)

## Exact-gradient accumulation

These two are numerical-correctness contracts, not style; the exact-doubling
tests they describe are examples of the "exactness is the contract" case in
[AGENTS.md](../AGENTS.md#mojo-not-python) and
[test-driven-development](../.agents/skills/test-driven-development/SKILL.md).

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

## Formatter and reserved words

- **`mojo format` rejects a plain reassignment to a local named `out`** — `out =
  expr` fails with `Cannot parse` even though the compiler accepts it, because
  `out` is the argument-convention keyword and the formatter's parser reads
  `out = …` as a malformed signature. `out += …`, `out[i] = …`, and `return out`
  all format fine; only the bare reassignment trips it. A file can compile and
  pass `pixi run test-fast` yet fail `pixi run fmt-check` for this alone. Don't name a
  reassigned local `out` (use `acc`/`result`); the same guard applies to the
  other convention keywords (`mut`, `read`, `owned`, `ref`, `deinit`). Reserved
  words also surprise as ordinary identifiers: `ref` cannot be a variable name at
  all (`unexpected token in expression`).

## Binary and string I/O

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
- **Raw binary file reads: `open(path, "r").read_bytes()`** returns the whole
  file as a `List[UInt8]`. There is **no `"rb"` mode** — passing it raises
  `invalid mode: "rb". Can only be one of {"r", "w", "rw", "a"}`; `read_bytes()`
  is a method on the ordinary text handle, not a binary-open flag. Reconstruct a
  little-endian float32 from four bytes with
  `bitcast[DType.float32, 1](SIMD[DType.uint32, 1](b0 | b1<<8 | b2<<16 | b3<<24))[0]`
  and widen with `Float64(f32)` — the widening is EXACT (every float32 is a
  float64), so a released-weights loader round-trips the on-disk f32 bit-for-bit
  (the GPT2W loader pins this with a probe). Scan for the header's newline and
  break at it rather than reading the whole file into a `String` — a 498 MB
  payload should never materialize as one string.
