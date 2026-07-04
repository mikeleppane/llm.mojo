# Part XI — Backpropagation by hand: build notes

Every layer Parts IX–X built now has a hand-derived backward, and every backward
is proven against a central finite difference of its own forward. No autograd —
that is the project's founding constraint. This part is where the finite-diff
technique that `test_grad_check.mojo` demonstrated on one function scales to the
whole stack. The deliverable is as much the checks as the code: a subtly wrong
gradient still trains (loss still falls), it just learns worse, silently.

## The finite-difference convention (stated once, in full)

Every backward test in this part follows the same three-step recipe, repeated
inline per file (no shared closure-taking checker — closure behavior under the
pinned 1.0.0b2 is an unforced risk, and the repetition is cheap):

1. **Projected scalar loss.** Pick a fixed, asymmetric cotangent `d_out` and form
   `L(x) = sum(d_out ⊙ f(x))`. Differentiating this one scalar checks the whole
   Jacobian *action* (the vector-Jacobian product the backward computes), not one
   output entry at a time. The analytic side is exactly `backward(cache, d_out)`.
2. **Central difference, `h = 1e-5`.** `dL/dx_i ≈ (L(x+h·e_i) − L(x−h·e_i)) / 2h`.
   The step comes from the Part III study (`test_finite_difference_step.mojo`):
   for a smooth `f` the central difference overshoots the true derivative by
   exactly `h²`, so `1e-5` sits in the sweet spot between truncation (`~h²`) and
   Float64 cancellation (`~ε/h`).
3. **Mixed tolerance.** `|analytic − numeric| ≤ 1e-7 + 1e-5·|numeric|`: an
   absolute floor for entries near zero (where relative error is meaningless
   after cancellation) plus a relative term for large entries. Looser than the
   `1e-9`–`1e-12` exact-math policy on purpose — a finite difference is an
   approximation, not an oracle. **Loosening a tolerance to make a gradient pass
   is forbidden**; a failing check means the formula is wrong (suspect the
   formula first, the test second, the tolerance last).

Two backwards are linear given their cache (dropout given its mask; softmax's
uniform-row case), so those get a tight `1e-9`/`1e-12` check instead — an exact
map deserves an exact test.

## Explicit per-layer caches; `forward` never changes

Each layer gains `forward_cached(x) -> <Layer>Forward` (the output plus a
`<Layer>Cache` holding exactly what backward needs) and
`backward(mut self, cache, d_out) -> d_input`. The original `forward` is left
untouched — it is the inference path, and generation should never pay for
caching. Explicit cache structs (over layers storing hidden mutable state) make
the data flow visible: a chapter can point at `LayerNormCache.rstd` and say "this
is why we saved it," a stale cache becomes a wrong *argument* rather than hidden
state, and a test can build a cache directly. **The contract, in every backward
docstring: a cache is valid only for the forward call that produced it.**

The caches turned out minimal. `LinearCache` is just `x` — both `dW = d_out^T @ x`
and `dx = d_out @ W` are built from `x` and `d_out`, and neither touches the
output. `LayerNormCache` stores `x` plus the per-row `mean` and `rstd` (two
scalars per row is cheaper than the full normalized `x̂`, which backward
recomputes). `EmbeddingCache` is the `ids`. `DropoutResult`'s mask *is* the
cache. `AttentionCache` is `q, k, v, weights` — not the mask (constant data, no
gradient).

## Gradients accumulate (`+=`), never overwrite

Every parameter backward does `grad += ...`, so two backward passes without a
`zero_grad()` between them sum. This is not pedantry: a later part ties the LM
head to the token embedding, and weight tying only works if two backward paths
through the one `Parameter` add their contributions. A dedicated per-layer test
pins it now — run backward twice, assert the grads exactly doubled — so tying
cannot silently overwrite half its gradient later.

**Exact doubling requires care about summation order** (a lesson the LayerNorm
test taught the hard way). `Linear` doubles bit-for-bit trivially: it forms the
full `dW` with a matmul, then adds it once (`grad = 0 + dW`, then
`grad = dW + dW == 2·dW`). LayerNorm's first draft accumulated `dγ_j += …` *inside*
the per-row loop, so the second call's running sum interleaved with the grad
already there — `((grad1 + t0) + t1) + t2` rounds differently than `2·grad1`, and
the exact-equality test failed. Fix: accumulate this call's `dγ`/`dβ` into locals
across the rows, then add the finished delta to the `Parameter` once. One
fully-formed `+=` per call per entry is what makes doubling exact. This is now an
AGENTS.md lesson.

## The two derivations that are chapter material

### Softmax row Jacobian

For one row `p = softmax(s)`, the Jacobian is `∂p_i/∂s_j = p_i(δ_ij − p_j)`. The
VJP contracts the upstream `dW` against it, column by column:

```
dS_j = Σ_i dW_i · p_i(δ_ij − p_j)
     = p_j dW_j − p_j Σ_i dW_i p_i
     = p_j (dW_j − Σ_i dW_i p_i).
```

So `dS = W ⊙ (dW − rowsum(dW ⊙ W))`. Two things fall out of this form. It takes
the **output** `W`, not the input scores — which is exactly what attention
already cached. And the subtracted `rowsum` is one scalar shared across the row,
so a blocked entry (`W ≈ 0`) both contributes `≈0` to it and receives `≈0` back:
masked positions leak no gradient, for free, with no special-casing of the mask.

### LayerNorm's three-term dx

The most-fumbled hand-written backward. Forward, per row over `C` features:
`μ = mean(x)`, `v = mean((x−μ)²)` (biased), `r = 1/√(v+eps)`, `x̂_j = (x_j−μ)r`,
`y_j = γ_j x̂_j + β_j`. Write `a_j = dL/dx̂_j = d_out_j γ_j`. Both `μ` and `r`
depend on *every* `x_k`, so `x̂` has to be differentiated through them:

```
∂μ/∂x_k = 1/C
∂v/∂x_k = (2/C)(x_k − μ)                 [Σ_j(x_j − μ) = 0 kills the μ term]
∂r/∂x_k = −½(v+eps)^{-3/2} ∂v/∂x_k = −(r²/C) x̂_k
∂x̂_i/∂x_k = (δ_ik − 1/C) r + (x_i − μ) ∂r/∂x_k
          = r[ δ_ik − 1/C − (1/C) x̂_i x̂_k ].
```

Contracting `a` against this Jacobian gives

```
dx_k = Σ_i a_i ∂x̂_i/∂x_k = r ( a_k − mean(a) − x̂_k · mean(a ⊙ x̂) ).
```

The two subtracted terms are **projections**: `mean(a)` removes the component
along the ones vector (the mean carries no gradient), and `x̂_k·mean(a⊙x̂)` removes
the component along `x̂` (the scale carries no gradient). That gives a purely
analytic test that catches a dropped term without any finite difference:
`Σ_k dx_k = 0` **exactly** (drop `mean(a)` and it becomes `r·C·mean(a)`, O(1)),
and `Σ_k dx_k x̂_k = 0` up to eps (drop the `x̂` term and it becomes
`r·C·mean(a⊙x̂)`, O(1)). The x̂-orthogonality is only approximate because with
`eps>0`, `Σ x̂_k² = C·v/(v+eps) ≠ C`; the residual is `r·C·mean(a⊙x̂)·eps/(v+eps)`,
`~1e-5` here — far below the O(1) a dropped term would produce, so `atol=1e-3`
distinguishes them cleanly while the ones-orthogonality holds at `1e-10`.

## Attention backward: where the scale lands

Core forward (pinned order): `S = qk^T`, `scaled = S/√D`, `+ mask`, `softmax → W`,
`out = Wv`. Reversing it, with `dO = d_out`:

```
dV = W^T @ dO                              (off the value matmul — NOT scaled)
dW = dO @ V^T
dS = softmax_rows_backward(W, dW)          (mask is constant; the add passes dW through)
dQ = (dS @ K) / √D
dK = (dS^T @ Q) / √D
```

The `1/√D` scale (`D = d_head`, the same value the forward used) folds into
`dQ`/`dK` **once** and never touches `dV`. Getting that wrong — scaling `dV`, or
scaling twice — is a classic bug; the finite-diff on all three under `no_mask`
and `causal_mask` catches it, and a fully-blocked-key test pins the no-leak
property directly (that key's `dk` and `dv` are `≈0`). MHA backward is pure
plumbing reversal: `proj.backward`, split `d_concat` into per-head `[T,D]` slices,
per-head core backward, `concat_cols` the per-head `dq`/`dk`/`dv` back to `[T,C]`,
reassemble the fused `[T,3C]` gradient in `Q|K|V` order, `qkv.backward`.
`slice_cols`/`concat_cols` are exact inverses (pinned in Part X), so the column
bookkeeping round-trips.

## The batched cross-entropy lands here

`cross_entropy_rows(logits [N,V], targets) -> Float64` is the mean of
`cross_entropy_one` over rows; `cross_entropy_rows_backward -> (softmax−onehot)/N`.
It is the top of every backward chain (the chain test needs a scalar loss to
differentiate) and later parts consume it unchanged. The `1/N` mean factor keeps
the gradient scale independent of batch size and is pinned by a test that doubles
the rows and checks the per-row gradient halves. The bigram's fused
`loss_and_grad` is left untouched — it is Part VII's own teaching artifact.

## The chain test trains

`test_backprop_chain` composes the real `Embedding → LayerNorm → MLP →
cross_entropy_rows` into a tiny classifier (`V = C`, so the MLP output is the
logits) and checks the wiring two ways: the gradient of the loss with respect to
the **embedding table** — the furthest-back parameter, reached only by threading
every backward in the stack — matches a finite difference of the whole forward;
and twenty full-batch SGD steps (`lr = 0.3`) drive the loss strictly down. That
the chain-rule wiring demonstrably *trains* before Part XII builds the real model
is the point. Attention stays checked in isolation (its own file) so the chain
test stays cheap and its failures stay localized; attention joins a full chain
when a Transformer block exists to hold it.

## Finite-diff checks that failed during development (blog material)

- **LayerNorm exact-doubling.** The dx, dγ, dβ finite-diff checks and the
  orthogonality test all passed on the first run — the *formula* was right — but
  the exact-doubling accumulation test failed. Cause was not the math but the
  summation order (see "Gradients accumulate" above): a per-row `grad +=` inside
  the loop interleaves the second call's partial sums with the first call's
  result, so `grad2 ≠ 2·grad1` bit-for-bit. Accumulating into a local and adding
  one finished delta per call fixed it. Worth keeping: the gradient was *correct*
  and the check still caught a real defect (a doubling that is off by ulps is a
  future weight-tying bug), which is exactly why the doubling test exists
  separately from the finite-diff test.

Every other backward's finite-diff passed on first correct compile; no formula
was wrong on the first try (the derivations were done on paper first, in the
docstrings).

## Deviations from plan

None of substance. The plan's signatures and test list were followed as written.
Minor additive choices the plan left implicit:

- Added a `<Layer>Forward` struct (output + cache) for **every** layer including
  Embedding and the attention core, for a uniform `forward_cached` return shape,
  and a `scaled_dot_product_attention_cached` wrapper so the core has a cached
  forward parallel to the struct layers (MHA's `forward_cached` uses it per head).
- The no-gradient-leak check uses a `key_padding_mask` that fully blocks one key
  column (a genuinely never-attended key, whose `dk`/`dv` are `≈0`) rather than
  trying to find an unattended key under a square causal mask, where none exists
  (every key `j` is attended by at least query `j`). The causal case is covered
  by the full finite-diff under `causal_mask`.

## Mojo lessons this part

- **`out` is a reserved argument keyword** — it cannot be a parameter name.
  `def sum_product(cot, out)` fails to parse; rename to `output`. (Same family as
  `ref`/`mut`/`var`/`deinit`/`read`.)
- **A field of a temporary struct can't be bound directly** when the field type
  is not `ImplicitlyCopyable`: `scaled_dot_product_attention_cached(...).cache`
  errors. Bind the whole result first (`var fwd = …; fwd.cache`), which borrows
  the field instead of transferring it out of a value about to be destroyed.
- **A tuple of move-only structs is awkward to return.**
  `def make_layers() -> (Embedding, LayerNorm, MLP)` failed to resolve the tuple
  constructor. Building the layers inline in each test (replaying the same seeded
  rng sequence for identical initial parameters) was simpler than fighting it.
- **Exact-doubling accumulation depends on summation order** — accumulate a
  call's parameter-grad delta into locals and add it once, not with a running
  `+=` inside an inner loop. (Promoted to AGENTS.md.)

## Review triage

Dual external review over `git diff main...part-11-backprop`, both reviewers
asked to re-derive the math, not just read it. Both **independently confirmed
all four hard derivations correct** (LayerNorm three-term dx, softmax VJP,
attention core scale/transpose placement, MHA fused-QKV plumbing) plus Linear,
Embedding scatter, GELU, MLP, and cross-entropy — no swapped Q/K/V block, no
stray transpose, no misplaced `1/√D`, no `=`-instead-of-`+=`, no stale cache.
Codex (GPT-5.5, high) raised 1 finding; Opus (4.8) raised 2 SHOULD-FIX + 3 NIT.
The two SHOULD-FIX items were real and are fixed; triage below.

- **SF-1 — dropout backward is not the VJP of the eval forward (both reviewers).
  FIXED.** Eval-mode forward is the identity (all-ones mask, no scale), but
  `dropout_backward(mask, p, d_out)` recomputed `1/(1-p)` from the passed `p`, so
  pairing the eval cache with the forward's own `p` gave `d_out/(1-p)`, not
  `d_out`. My `test_eval_and_p_zero_backward_is_identity` had dodged it by passing
  `p=0.0` to backward — a *different* `p` than the forward. Fix: cache the applied
  scale. `DropoutResult` now carries `inv_keep` (1.0 in eval / p=0, `1/(1-p)` in
  training) and `dropout_backward(mask, inv_keep, d_out)` consumes it — no `p` to
  get wrong, forward and backward are exact inverses in every mode. New tests pin
  eval-identity using the *cached* scale (`test_eval_backward_is_identity_with_
  cached_scale`) and p=0 identity.
- **SF-2 — Embedding scatter used a running `+=` inside the loop (Opus). FIXED.**
  The exact anti-pattern the LayerNorm fix (and the new AGENTS.md rule) warns
  against — and I'd missed applying the lesson to Embedding. For a repeated id the
  per-element `table.grad[idx,j] += …` interleaves the second backward call's
  partial sums with the first call's stored result, so the exact-`==` doubling
  drifts by a ulp on adversarial data (Opus measured ~22% of random repeated-id
  deltas). My original doubling test passed only by luck of its values. A quick
  search found a deterministic 3-value case that drifts by 1 ulp; I pinned it as
  `test_repeated_id_doubling_is_bit_exact` (**confirmed failing before the fix**),
  then fixed backward to sum each distinct id's contribution into a per-id local
  and add the finished delta to `table.grad` once — bit-exact doubling for any
  data, mirroring LayerNorm. The scratch is sized to the distinct touched ids
  (linear scan over the short id list), not the whole vocabulary, so untouched
  rows cost nothing.
- **N-1 — LayerNorm derivation typo `∂r/∂x_k = -(r/C)x̂_k` (Opus). FIXED.** Missing
  a factor of `r`: the correct intermediate is `-(r²/C)x̂_k` (the final Jacobian
  and the code were already right — only this one line was off). Fixed in both
  `layernorm.mojo` and this file. This is exactly why the review brief asked both
  reviewers to re-derive: a teaching derivation with a wrong intermediate step
  mis-teaches even when the code is correct.
- **N-2 — `gelu_rows_backward` test mirrors the implementation (Opus). ACCEPTED
  AS-IS.** The rows test asserts `got == d_out ⊙ gelu_derivative(x)`, which is what
  the function computes. Kept deliberately: `gelu_derivative` is independently
  finite-difference checked against `gelu` in the same file, and the plan's design
  for GELU is precisely "the scalar derivative is the verified unit; rows is the
  elementwise map." So the elementwise wiring and the scalar derivative are both
  covered without redundancy. Noted so the mirror pattern is not copied where the
  scalar isn't separately validated.
- **N-3 — copy churn in the `_cached` wrappers (Opus). ACCEPTED AS-IS.** The
  caches must outlive the forward temporaries, so the copies are deliberate;
  correctness-before-speed means the inference `forward` stays copy-free and a
  `^`-transfer pass for caller-owned inputs is a later perf concern, not a Part XI
  change.

Both fixes were TDD: the failing test came first (dropout eval-identity with the
cached scale; the deterministic embedding-doubling drift), then the fix, then
green. Gates re-run clean after triage.
