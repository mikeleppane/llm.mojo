# Part IX — NN building blocks: build notes

Raw material for the chapter. Part IX creates the `nn/` package: every layer the
Transformer is assembled from except attention (Part X). **Forward passes only** —
the backward pass, with finite-difference gradient checks, is a later part. Ships
on branch `part-09-nn`, kept after merge.

The package imports `tensor` and `utils` only (the layering graph runs
`utils → tensor → nn`); it does not touch `config`, `tokenizer`, `data`, or
`models`. Dimensions reach the layers as runtime constructor arguments, never as
imports or struct parameters (the architecture-family decision to keep model
dimensions out of the type system stands).

---

## The shape everything shares

Layers operate on `Tensor2D` shaped `[N, C]`, where `N` is a flattened `B·T`.
Every layer here is position-independent — Linear, LayerNorm, GELU, and dropout
all act per row — so there is no need for a 3D layer variant. The model (a later
part) owns the `[B, T, C]` view and flattens to `[B*T, C]` before calling layers,
which keeps the tested `matmul` the only matmul and avoids duplicate kernels.

`Parameter` (a `value` tensor plus a same-shaped zeros `grad`) lands now rather
than at the backward pass. Layers own `Parameter`s, not bare tensors, so the
backward pass adds *methods* and the optimizer gets its uniform surface without a
rewrite. This is the concrete cash-out of the earlier "no `Module` trait"
decision: a shared parameter type, not a shared behavior trait.

## Design decisions as built

- **Weight convention `[out, in]`, forward `x @ W^T + b`.** Output channel `o` is
  the contiguous row `W[o, :]`, matching `xavier_2d`'s `[fan_out, fan_in]` layout.
  The forward transposes the weight and reuses the tested `matmul` — no second
  kernel. (GPT-2's TensorFlow checkpoint stores Conv1D kernels transposed as
  `[in, out]`; that is a transpose-at-load concern for the weight-loading part,
  never a reason to bend the library convention.)
- **GELU is the tanh approximation**, not the erf-exact form:
  `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`. GPT-2's released weights were
  trained against this exact form; the erf variant differs in the 4th decimal (at
  `x = 1`, `0.8411920` vs `0.8413447`) and would drift logit parity. The frozen
  goldens are computed from the tanh formula and a dedicated test rejects the erf
  value, so the wrong variant fails loudly.
- **LayerNorm uses biased variance** (÷C, not ÷C-1) and eps 1e-5, with eps inside
  the sqrt. This is what GPT-2 / PyTorch `nn.LayerNorm` compute. The 3x4 oracle
  golden is computed biased and a test explicitly rejects the unbiased row-0
  values, so a ÷(C-1) regression can't slip through. A constant row (variance 0)
  exercises the eps path — it maps to the bias exactly instead of dividing by zero.
- **Dropout is inverted dropout with the mode as an argument.** Training keeps each
  element with probability `1-p` and scales survivors by `1/(1-p)`; eval is the
  identity. Crucially, eval mode and `p == 0` short-circuit *before any rng draw*,
  so disabling dropout never perturbs the seeded generator — a twin-generator test
  pins this. Raises on `p` outside `[0, 1)` (`p = 1` divides by zero in the scale).
- **Init policy: GPT-2's `normal(0, 0.02)`** for the layer factories, biases zero;
  `GPT2_INIT_STD` is a single named constant in `linear.mojo` that `embedding.mojo`
  imports (one source of truth, no drift). `xavier_2d` stays untouched as the
  earlier teaching artifact — a different scheme, and the factories say so.
- **Embedding is one struct used twice** (token and positional), with a public
  `table` Parameter so a later part can tie the LM head to it. `forward` gathers
  rows by id and raises on any id `< 0` or `>= V` *before* writing that row.
- **MLP hidden width is an explicit argument**, never a hardcoded 4x. GPT-2 passes
  `4C` later; the block doesn't assume the ratio.

## The sqrt-in-comptime probe (new territory — blog material)

The scheduled probe for this part: does `std.math.sqrt` run in a `comptime`
context on the pinned 1.0.0b2, so `comptime SQRT_2_OVER_PI = sqrt(2.0 / pi)`
compiles? Part VIII verified struct construction and `List` building in comptime;
this was the one remaining unknown for the GELU constant.

**Result: it works.** Run in scratch before committing:

```mojo
comptime SQRT_2_OVER_PI = sqrt(2.0 / pi)
# ...
comptime assert SQRT_2_OVER_PI == 0.7978845608028654, "sqrt(2/pi) != literal"
```

Both the constant binding and a `comptime assert` comparing it to the literal
compiled and ran clean; `SQRT_2_OVER_PI` printed `0.7978845608028654`, bit-identical
to the hand-spelled literal. So the primary branch was taken: `nn/gelu.mojo` binds
the constant at compile time, and the derivation *is* the documentation — no
fallback literal was needed. The general lesson for the repo: **`std.math`
transcendental/root functions evaluate in a comptime context on 1.0.0b2**, so
derived numeric constants (not just exact integer arithmetic, as in Part VIII) can
be bound at compile time.

`GELU_CUBIC = 0.044715` is *not* derivable — it is a fitted constant from the GELU
paper (Hendrycks & Gimpel, 2016) — so it is named and cited rather than bound from
an expression. `LAYERNORM_EPS = 1e-5` is GPT-2's value, named at module scope.

## Oracle goldens

`tests/oracles/nn_reference.py` (NumPy, float64) computes reference values for
GELU (tanh formula), LayerNorm (biased variance, eps 1e-5), a small Linear, and
the tiny MLP composition. It is run once by hand; its printed numbers are frozen
as literals into the Mojo test files with a comment pointing back to the script.
Nothing under `src/` or the test suite imports it — tests stay offline, and the
goldens are an *independent* oracle (reference math here, implementation in
`src/llm/nn/`, compared only through the frozen literals). This mirrors the
tokenizer's `gpt2_reference_encoder.py`. Tolerances are 1e-12; floats are never
compared with `==`.

## Deviations from plan

None of substance. The plan's signatures were built as written. One small
addition beyond the letter of the plan: each layer's `forward` and each
`init_random` carries an explicit shape/dimension guard that raises with a clear
message (e.g. `Linear.forward` checks the input feature count), consistent with
the tensor layer's existing "checked on demand" error style.

## Review triage

Dual external review, read-only and non-interactive, over `git diff
main...part-09-nn`: Codex (GPT-5.5, high reasoning) and Claude Opus 4.8 (xhigh).
Both independently confirmed the load-bearing math — GELU tanh vs erf, LayerNorm
biased vs unbiased denominator and eps placement, dropout eval-mode rng
discipline, the `[out, in]` convention, and oracle independence. Opus recomputed
the goldens from a fresh Python session (not the repo's own oracle) and matched
every frozen literal. Opus verdict: **approve**. Codex verdict: **request
changes** on the items below.

Findings triaged (all fixed; each fix got a failing test first):

1. **Dropout accepted NaN `p`** (Codex). The old guard `p < 0.0 or p >= 1.0` let
   a NaN through (every comparison with NaN is false), which would zero the whole
   output while still consuming `N·C` rng draws. Rewrote the guard as
   `not (p >= 0.0 and p < 1.0)`, which raises on NaN. Test: `test_p_nan_raises`.
2. **`p == 0.0` float `==`** (Codex flagged as a contract violation; Opus judged
   it a correct, intended sentinel). The behavior was right, but the house rule
   bans float `==`. Since the range guard already excludes `p < 0`, the no-op
   sentinel is now written `p <= 0.0` — same behavior, no `==`. Covered by the
   existing `test_p_zero_in_training_is_identity_and_untouched_rng`.
3. **Linear didn't validate bias shape** (Codex). A hand-built `Linear` with a
   bias narrower than `out` read out of bounds at the per-column add. `forward`
   now validates the bias is `[1, out]` and raises a clear layer error. Test:
   `test_bias_shape_mismatch_raises`. (Unreachable via `init_random`, which always
   builds the right shape — a direct-misconstruction hardening.)
4. **LayerNorm didn't validate bias shape** (both Codex and Opus, the latter as a
   non-gating nit). Same class as (3): a shorter bias than weight read out of
   bounds at the per-column shift. `forward` now validates weight and bias are
   both `[1, C]`. Test: `test_bias_shape_mismatch_raises`.

No findings rejected. The two "consistency check" tests Codex could have flagged
(MLP manual composition, `gelu_rows` vs scalar) are backed by independent oracle
goldens elsewhere in the same files, which Opus explicitly confirmed — they
supplement the oracle, they don't stand in for it.
