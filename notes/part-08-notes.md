# Part VIII — Architecture Family: build notes

Raw material for the chapter: the binding family decisions later parts justify
their shape against, the metaprogramming policy the project now inherits, and
the two comptime probe results (new territory for this repo on 1.0.0b2). Ships
on branch `part-08-architecture`. Deliberately small: a preset, an exact
parameter count, a compile-time contract pin, and two behavior-frozen
metaprogramming cleanups. No layers, no traits, no `nn/` or `transformer/`
packages — those are Part IX.

---

## Why no trait hierarchy

The tempting deliverable — a `Module` trait (`forward()`, `num_parameters()`)
every future layer implements — was rejected, and the reasoning is recorded
because it will be asked again:

1. **The signatures genuinely differ.** `Embedding.forward` takes integer ids
   and returns floats; attention takes a mask; dropout takes a mode and an
   `Rng`. A trait wide enough to cover them all says nothing usable; a trait per
   shape is a hierarchy with one implementer each.
2. **No consumer exists yet.** The first code that treats layers uniformly is
   the optimizer (iterate parameters) and the backward-pass wiring, both later
   parts. Designing the interface before its consumer is guessing.
3. **The backward pass would force a redesign anyway.** A `forward`-shaped trait
   gains a `backward` obligation later, with cached activations whose types
   differ per layer.

What replaces it: Part IX introduces a plain `Parameter` struct (value tensor +
gradient tensor) as the shared building block; uniform treatment of parameters
stays a later (optimizer) concern. Traits earn their place when they make
components interchangeable in tests, not as architecture theater.

## Binding family decisions (later parts justify their shape against these)

- **The main line is decoder-only** in the GPT-2 layout: pre-LN, learned
  positional embeddings, weight tying (LM head shares the token-embedding
  matrix). Encoder-only and encoder-decoder are explained in the chapter, but
  only the encoder-decoder gets code, in the Part XII lab.
- **Attention must be one core serving both self- and cross-attention.** Part X
  builds scaled-dot-product attention so the same core takes Q from one source
  and K/V from another; the Part XII lab consumes the cross variant, so the
  split is a Part X design requirement, not a later retrofit.
- **Masks are data passed into attention, never baked into it.** Causal and
  padding masks compose; attention receives a mask argument.
- **Modern variants enter as opt-in config flags, off in GPT-2 mode.** RoPE,
  RMSNorm, SwiGLU (Part XX) become `GPTConfig` fields that default to the GPT-2
  behavior; `GPTConfig` does not grow those fields until then.

## The parameter count as the architecture, committed early

`parameter_count()` is the whole architecture written as arithmetic before a
single layer exists. Its total, 124,439,808, is the independently published
GPT-2 124M figure, so the arithmetic cannot reproduce it with a missing term —
one absent bias vector moves the number. Deriving the total (rather than
asserting it bare) pins every structural decision now:

- biases present on every linear (QKV, attention projection, both MLP matrices);
- LayerNorm with both weight and bias, two per block plus a final one;
- learned positional embeddings sized to `context_length` (not sinusoidal);
- the LM head tied to the token embedding (contributes 0).

The per-block cost collapses to the constant `12*C^2 + 13*C` (12 from the four
weight matrices 3+1+4+4, 13 from the bias/norm vectors 2+3+1+2+4+1). A test
pins the delta of adding one block to that constant, so the formula's structure
is tested, not only its total. When later parts build the real model, their own
count must reconcile with this formula and weight loading must fill exactly
these tensors; any drift breaks a test written here.

`dropout` in the preset is 0.1, GPT-2's *training-time* value. It lives in the
preset because it is the architecture's stated number; evaluation disables
dropout at the layer level (the layer reads a mode flag), so inference parity is
unaffected by this field. Documented here and in the preset docstring so the
0.1-vs-0.0 question does not resurface as a parity bug later.

## Metaprogramming policy (the project inherits this)

**Use now (this part):**
- **Comptime for derived constants.** Magic literals that are exact functions of
  known quantities become named `comptime` constants whose derivation is the
  documentation: `INV_2_POW_53 = 1.0 / Float64(1 << 53)` and `TWO_PI = 2.0 * pi`
  in `utils/random.mojo`. Exact arithmetic, so outputs are bit-identical.
- **Comptime contract pins.** Non-raising pure functions over the config can be
  asserted at compile time, failing the *build* on drift (see the probe below).
- **Comptime tables for fixed file-format facts.** The GPT-2 byte<->unicode map
  is known before the program runs, so it is built once at compile time (see the
  probe below).

**Reserve for later (recorded so parts inherit the intent):**
- Part IX: GELU-style derived constants at compile time, e.g.
  `SQRT_2_OVER_PI = sqrt(2.0 / pi)` (probe `std.math.sqrt` in comptime first).
- **Float32 migration mechanism.** The precision switch will be a *type
  parameter* on the tensors (`Tensor2D[dtype: DType]`), not a global scalar
  alias. A global `comptime Scalar = Float32` alias was analyzed and **rejected**:
  the test suite compares Float32 model math against Float64 oracles, and a
  single global scalar cannot let both dtypes coexist in one test. Parameterizing
  the tensor over dtype keeps the oracle at Float64 while the model runs Float32.
  `Float64` stays literal until that part.
- Part XVIII: `comptime for` unrolling, SIMD width as an inferred parameter, tile
  sizes as parameters, `comptime if` for optional GPU paths.
- The `where`-clause constraint system is available for any parametric API that
  emerges; no current consumer.

**Rejected outright:**
- **Model dimensions as struct parameters** (`GPTModel[d_model, ...]`). Weights
  load at runtime, tests vary configs freely, and per-shape specialization of the
  whole model trades compile time for nothing a reader learns from. Dimensions
  stay runtime values; parameters are for specialization, dynamic values belong
  in arguments.
- **Test goldens as comptime-derived values.** Goldens must be computed
  independently of the code under test (that is what makes them an oracle);
  deriving them at compile time from the same arithmetic would destroy that
  independence.

## Comptime probe results on 1.0.0b2 (new territory — blog material)

Both probes were run in scratch before committing, because struct-in-comptime
and comptime `List` materialization were unverified on the pinned toolchain.

### Probe 1 — struct construction + method call in comptime: **works**

```mojo
comptime GPT2 = GPTConfig.gpt2_124m()
comptime assert GPT2.parameter_count() == 124_439_808, "..."
```

A user struct constructs cleanly in a comptime context, and a non-raising method
runs on it at compile time. The primary branch was taken: the contract pin lives
in `check_gpt2_contract()` (a `comptime assert` is illegal at module scope, so it
sits in a function). Two further findings:

- **An uncalled function's `comptime assert` does not fire.** A deliberately
  false assert in an uncalled function compiles fine. The assert is only
  evaluated where the function is *called*, so the pin needs a live call site —
  the config test calls `check_gpt2_contract()`, which makes the *test build*
  enforce the contract. Verified by flipping the golden to 124_439_809: the test
  file then fails to compile ("function instantiation failed" at the assert),
  not at runtime.
- No fallback needed. The plan's fallback (comptime-evaluate the arithmetic
  through a free function over plain Ints) was not required; struct-in-comptime
  is enough. The runtime `TestSuite` test stays anyway, for suite visibility and
  because it guards the runtime materialization edge.

### Probe 2 — comptime-built `List` at module scope: **works, with a wrinkle**

`comptime BYTE_TO_UNICODE = gpt2_byte_to_unicode()` builds the 256-entry table at
compile time. But a use site cannot read it directly:

```
error: cannot materialize comptime value of type 'List[Int]' to runtime
       because it is not 'ImplicitlyCopyable'
       note: use 'materialize' to explicitly materialize the value
```

`List` is not `ImplicitlyCopyable`, so the compiler refuses to silently copy the
comptime value into runtime memory. The fix the compiler itself suggests works:
`materialize[BYTE_TO_UNICODE]()` lifts the frozen table into a runtime `List`
explicitly. So the table build (the loop that finds the non-printable bytes) runs
once at compile time and `from_files` just materializes the constant. The runtime
`gpt2_byte_to_unicode()` stays as the readable definition (tokenizer parity tests
call it directly); the comptime binding is derived from it. Tokenizer parity was
re-run green after the change — the same 256 values reach the loader either way.

The general lesson for this repo: **comptime can build a non-`ImplicitlyCopyable`
allocating value, but reading it at runtime requires an explicit
`materialize[...]()`** — a bare use is a compile error, not a silent copy.

## Review triage

One external reviewer (Claude Opus 4.8, xhigh reasoning), read-only,
non-interactive, over `git diff main...part-08-architecture`. Verdict: **diff
clean** — no blockers, no should-fix. The reviewer independently recomputed the
parameter total to 124,439,808 term-by-term, confirmed the per-block constant
`12C^2 + 13C`, and checked both cleanups for bit-identity — including the one
real risk on the `TWO_PI` change: `*` is left-associative, so the original
`cos(2.0 * pi * u2)` already grouped as `(2.0*pi)*u2`, and precomputing the
product performs the same two multiplies in the same order (goldens hold).

Two optional nits, both **fixed**:

1. `test_parameter_count_embedding_share` comment claimed the exact count minus
   its terms "leaves nothing over," but the assertion only checked
   `count > embed`. Strengthened to reconstruct the full total from its
   documented parts (`embed + T*C + L*(12C^2+13C) + 2C`) and assert exact
   equality, so the test now proves what the comment says.
2. `assert_true(cfg.dropout == 0.1)` used float `==` in a test. The house rule
   bans float `==`; switched to `assert_almost_equal`, consistent with the rest
   of the numeric suite. (Safe either way — exact literal round-trip — but the
   rule is the rule.)

No findings rejected; there were none to reject.
