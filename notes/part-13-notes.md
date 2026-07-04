# Part XIII — The GPT-2 model: build notes

The convergence part. Every layer Parts IX–XI built and proved, and the block
assembly Part XII rehearsed on a lab model, comes together here into the real
decoder-only GPT: `TransformerBlock` (GPT-2's pre-LN block, self-attention only)
and `GPT` (embeddings → L blocks → final LayerNorm → weight-tied head). The
walked parameter inventory reconciles with the Part VIII formula that already
comptime-pins 124,439,808.

Only two things were genuinely new; everything else was assembly of proven
forwards and backwards. Both are written out below as chapter material.

## The weight-tied head: one Parameter, two gradient paths

The language-model head has no matrix of its own. The logits are

```
logits = h @ wte.table^T          # h = ln_f(x), wte.table is [V, C]
```

— GPT-2 reuses the token-embedding table, transposed, as the output projection,
with no bias. So `parameter_count()` counts the head as 0, and the model's
largest matrix does double duty: it embeds tokens at the bottom and unembeds
them at the top.

In backward that means `wte.table` receives gradient through **two** paths that
**sum** into its one `Parameter.grad`:

```
d_table (head path)   = d_logits^T @ h        [V, T] @ [T, C] -> [V, C]
d_h                   = d_logits @ table      [T, V] @ [V, C] -> [T, C]   (flows down the stack)
... ln_f, blocks, embedding dropout ...
d_table (gather path): scatter-add d_emb rows into table.grad by token id
```

The head path reaches **every** row of the table (the logits depend on all of
`wte` through `h @ wte^T`), while the gather path reaches only the rows for the
token ids actually fed. The model-level finite-difference of the table grad is
the test that both paths are present and summing: perturbing an *unused* row
(e.g. token 0 or 2 when the ids are 1/3/4) checks the head path alone; perturbing
a *used* row checks head + gather together. A missing head path, a missing
gather path, or `=` instead of `+=` is off by a whole term, and the finite-diff
catches it.

### Why the two paths are combined into one delta

Part XI's exact-doubling contract has a sharp consequence here. Two backward
passes without a `zero_grad()` between them must double every grad *bit-for-bit*
— that is the property weight tying was built to exploit. But float addition is
not associative: if each backward added the head delta and the gather delta as
**two** separate `+=` into `table.grad`, then after two calls the accumulation
would be

```
((h + g) + h) + g          which is NOT bit-identical to   2·(h + g).
```

So `GPT.backward` sums the head delta and the gather scatter into **one** `[V, C]`
delta (`d_table`), then adds that finished delta to `table.grad` **once** per
call. One fully-formed `+=` per call → exact doubling. This is the same lesson
LayerNorm's `dγ`/`dβ` and Embedding's repeated-id scatter taught, now applied to
the tied weight it was always aimed at. The scatter into `d_table` mirrors
`Embedding.backward` (repeated ids sum); `wpe`, reached by a single path, uses
`Embedding.backward` directly. A dedicated model-level doubling test pins this,
and `test_gpt.mojo`'s tied-head test pins that the head is exactly `h @ wte^T`
with no bias.

*(Deviation from the plan's literal wording:* D4 described "the embedding
gather's backward later scatter-adds into the SAME `wte.table.grad`" — i.e. a
second `+=`. Written that way the tied weight would fail bit-exact doubling for
the reason above. The combined-delta form computes the identical gradient value
— the finite-diff still passes — while making the doubling exact. Kept.)*

## Attention-weight dropout: a backward that is composition, not new math

GPT-2 drops the post-softmax attention weights before they weight the values:
`output = dropout(W) @ v`. This landed as an **additive** train path over the
frozen scaled-dot-product core (`scaled_dot_product_attention_train` and its
backward, plus `MultiHeadAttention.forward_cached_train`/`backward_train`), so no
existing signature changed and every earlier caller keeps compiling.

The forward's pinned order gains exactly one step:

```
scores -> scale -> + mask -> softmax -> W -> dropout_cached(W) -> dropped_W
output = dropped_W @ v
```

The backward is a composition of two proven pieces, and the only subtlety is
**which** weight tensor feeds **which** term:

```
dropped_W = W ⊙ mask · inv_keep            (reconstructed from the cache)
dV        = dropped_W^T @ dO               (the value matmul saw the DROPPED weights)
d_droppedW = dO @ V^T
dW        = dropout_backward(mask, inv_keep, d_droppedW)   (undo the drop)
dS        = softmax_rows_backward(W, dW)   (on the PRE-dropout W — what softmax made)
dQ = (dS @ K)·s,  dK = (dS^T @ Q)·s        (exactly as the frozen core, s = 1/√D)
```

`dV` is fed the **dropped** weights (that is what multiplied `v` in the forward);
`softmax_rows_backward` is fed the **pre-dropout** `W` (that is what the softmax
produced). Swapping them is the classic attention-dropout backward bug, and the
finite-diff through the dropped core catches it. The cache stores `{q, k, v,
weights, drop_mask, inv_keep}` — the pre-dropout `W` for the softmax backward,
the mask and scale to reconstruct the dropped `W` for `dV`.

With `training = False` (or `p = 0`) `dropout_cached` returns an all-ones mask,
`inv_keep = 1.0`, and draws no rng, so the train path degenerates to the proven
one **exactly** — outputs and all gradients equal. A test pins that equivalence
at the core and MHA levels; the block and model inherit it.

### The re-seeded-rng finite-diff convention

With `training = True` the forward draws a dropout mask, so a naive central
difference would compare two *different* masks and measure noise. But the mask
depends only on the draw sequence — one uniform per weight entry, in fixed
row-major order — and never on `q`/`k`/`v`. So **re-seeding the rng identically
before every forward call** in the finite-diff loop replays the identical mask
for `x + h` and `x − h`. The check then differentiates the mask-fixed map (linear
in the fixed mask), which is exactly what the backward computes. This is the only
finite-diff wrinkle unique to this part; every check that does not target dropout
runs with `training = False` and needs no such care.

## GPT-2's residual-init scaling

`GPT.init_random` draws every weight from `normal(0, 0.02)`, then scales each
block's attention `proj` weight and MLP `down` weight **in place** by `1/√N`,
`N = 2·n_layers` = the number of residual additions (std 0.02 → 0.02/√(2L),
≈ 0.00408 at L = 12). These are the residual-*feeding* projections; scaling them
keeps the residual stream's variance from growing linearly with depth. `qkv` and
MLP `up` stay at 0.02.

Scaling *after* drawing (rather than threading a per-layer std through
`Linear.init_random`) keeps the rng draw stream identical to the unscaled layout
and every layer-factory signature untouched — the deliberate trade the plan
called for. `test_gpt.mojo` pins the sample std of `proj`/`down` to the scaled
value and of `qkv`/`up` to 0.02, so scaling the *wrong* matrices fails a band.

## The dropout placement, and where the rng is unreachable

The cached path IS the training path — dropout lives only there. The plain
`forward` is the inference path and takes no rng argument at all, so applying
dropout at inference is *unrepresentable*, not merely tested against. GPT-2's
three sites, all driven by the single `cfg.dropout`:

1. **embedding dropout** on `wte(ids) + wpe(pos)`, at the model level;
2. **attention-weight dropout** inside each block's self-attention train core;
3. **residual dropout** on each sublayer's *branch* before the residual add —
   `x + dropout(sublayer(ln(x)))`. The skip path is **never** dropped.

`test_block.mojo` pins the placement two ways: a zeroed-sublayer block reproduces
its input `x` *exactly* even under `training = True` with `p = 0.9` (if the skip
were dropped, or dropout hit the residual sum, `x` would come back sparsified),
and a real block's `training = True` output demonstrably differs from inference
(the branch dropout is not inert).

## The one-directional layering forced an inlined SGD update

`GPT.apply_sgd` / `TransformerBlock.apply_sgd` perform the `p -= lr·grad` update
via a small `sgd_parameter` helper in `transformer/block.mojo`, **not** by
importing `training.optimizer.sgd_step`. The dependency layering is
`nn → transformer → {training, generation}`: `transformer/` sits *below*
`training/`, and a lower layer must never import a higher one. The lab's
`zero_grad`/`apply_sgd` helpers could call `sgd_step` because the lab package
sits above `transformer/`; the main-line model cannot. The update is a one-liner,
so inlining it is cheap and keeps the layering clean.

*(Deviation from the plan's wording:* D5 said "via the free `sgd_step`". Taken
literally that is an upward import and a layering violation the review brief
explicitly asks to flag. Inlining the identical update resolves it. Kept.)*

## Reconciliation: the walk meets the formula

`GPT.parameter_count_actual()` walks every Parameter and sums `value.size()`,
counting the tied `wte` **once** (the head owns no Parameter).
`test_gpt.mojo` asserts it equals `GPTConfig.parameter_count()` on two tiny
configs — one symmetric (V=10, C=16, L=2, H=2), one asymmetric (V=7, C=12, L=3,
H=3, dropout 0.1) — and that a double-counted `wte` would exceed the formula by
exactly `V·C`. That transfers Part VIII's comptime pin of 124,439,808 to the real
tensors without allocating them at full size.

The full-size build is `examples/gpt2_inventory.mojo` (D7): it constructs
`gpt2_124m()` (~2 GB resident), prints the component-by-component inventory, and
asserts the walked total equals both the formula and the literal 124,439,808. Run
manually — never in the suite. Its output:

```
GPTConfig(vocab_size=50257, context_length=1024, d_model=768, n_layers=12, n_heads=12, dropout=0.1)
Allocating the 124M preset (~2 GB resident)...

Component                         Parameters
------------------------------------------------
token embedding    (wte, V*C)     38597376
positional embed   (wpe, T*C)     786432
per block          (x 12 )          7087872
all blocks                        85054464
final LayerNorm    (ln_f, 2C)     1536
LM head            (tied)                 0
------------------------------------------------
walked total                      124439808
formula (GPTConfig)               124439808

Reconciled: walked == formula == 124,439,808 (GPT-2 124M).
```

The rows sum exactly: 38,597,376 (wte) + 786,432 (wpe) + 85,054,464 (12 blocks)
+ 1,536 (ln_f) + 0 (tied head) = 124,439,808.

## Finite-diff / equivalence / smoke checks that failed during development

Kept as blog material (a wrong wire that still *trains*, just worse, is the whole
reason these checks exist).

- **Nothing's finite-diff was wrong on first correct compile.** Every backward in
  this part is a composition of Part XI backwards whose math was already proven,
  so the gradients were right the first time. What the checks earned their keep on
  was the *accumulation grouping*, not the math:
- **wte exact doubling would have failed with the plan's literal two-`+=` form.**
  Reasoned out before writing (float addition is not associative, so
  `((h+g)+h)+g ≠ 2·(h+g)`), the tied weight's two paths were combined into one
  delta added once — so the model-level doubling test passes bit-for-bit. This is
  the Part XI LayerNorm/Embedding lesson landing exactly where it was always
  aimed. Had the two-`+=` form shipped, `test_model_exact_doubling` would have
  caught it (a doubling off by ulps is a real defect, which is why the doubling
  test is separate from the finite-diff).
- **The overfit smoke converged on the first real config** (V=6, C=8, L=2, H=2,
  T=5, lr=0.5, 120 steps): dropout=0 loss falls monotonically at the checkpoints
  and lands well below log V; the dropout=0.1 run still learns. No hyperparameter
  tuning was needed to paper over a wire, which — given every gradient is
  finite-difference-checked upstream — is the expected outcome.

## Deviations from plan

- **wte gradient accumulation: two combined into one delta** (not the literal
  two-`+=`), for bit-exact doubling — see above. Same gradient value.
- **`apply_sgd` inlines the update** rather than importing `sgd_step`, forced by
  the one-directional layering — see above.
- **The block/model forward oracle uses a shared deterministic `fill` pattern**
  for weights rather than ~150 hand-typed literals. The pattern
  (`v = (((k+base)·37+11) mod 101)/100 − 0.5`, integer-exact in Float64) is
  authored test input, independent of the model; the NumPy oracle computes the
  forward from it independently, and the frozen output goldens are what the test
  checks. The block forward golden matching confirms both the Mojo `fill` matches
  NumPy bit-for-bit *and* the forward wiring is correct.
- Everything else matches the plan's signatures, module list, and test intent.
  Nothing under `tensor/`, `nn/`, `lab/`, `training/`, `generation/`, or
  `config.mojo` changed (D1 honored) — verified with `git diff main --stat`.

## Mojo lessons this part

- **A field of a struct binds by copy for non-`ImplicitlyCopyable` types even on
  a plain read.** `var w = train.cache.weights` fails ("cannot be implicitly
  copied"); `.copy()` fixes it. Same family as the AGENTS.md single-field rule,
  seen again binding cache fields in tests.
- **A temporary cannot bind to a `mut` argument.** The eval path draws no rng but
  the `forward_cached(ids, training, mut rng)` signature still needs one, so a
  test must pass a *named* `var rng = Rng(0)`, not `forward_cached(..., Rng(0))`
  — an rvalue has no mutable storage to borrow.
- **Imports must be at module scope**, not inside a function body — a helper that
  reached for `from llm.tensor.ops import cross_entropy_rows` inside its body
  failed to parse; hoist it to the top.
- **`GPTConfig` is `Copyable` but not `ImplicitlyCopyable`**, so passing a `read`
  `cfg` into the `GPT` constructor needs `cfg.copy()` (the same rule as every
  Tensor2D transfer).

## Review triage

Dual external review over `git diff main...part-13-gpt2-model`, both read-only,
both asked to VERIFY THE ASSEMBLY AND THE TYING by re-deriving the math (tied
two-path gradient, attention-dropout backward, dropout placement, the walk, the
frozen-layer rule), not just reading it.

- **Codex (GPT-5.5, high): no blocking defects, zero findings to fix.** Eight
  ranked *confirm-correct* items, each re-derived with the concrete failure it
  avoids: the tied `wte` gradient combines both paths into one `[V,C]` delta
  before a single `+=` (so two backward passes double bit-exactly, not
  `((h+g)+h)+g`); the attention-dropout backward feeds the reconstructed *dropped*
  weights to `dV` and the *pre-dropout* `W` to `softmax_rows_backward`; dropout
  wraps each branch with the skip entering `add` raw; `wpe` sized
  `context_length` with positions from row 0; the walk reaches every Parameter
  and counts tied `wte` once; residual-init scaling hits exactly `attn.proj` and
  `mlp.down`; named `T=0`/`T>ctx` errors; and the oracle is independent NumPy math
  (shared only the deterministic `fill` values). Full text:
  `docs/plans/part-13-review-codex.md` (gitignored).
- **Claude Opus 4.8 (xhigh): 0 blocker, 0 should-fix, 1 nit.** It re-derived the
  three adversarial targets and *refuted* each (i.e. confirmed correct): the tied
  doubling (`d_table` is a deterministic function of `(cache, d_logits)`, so two
  calls give `D ⊕ D = 2D` in IEEE), the attention-dropout weight routing (a swap
  would fail the independent finite-diff), and dropout placement (inference
  `forward` takes no rng and calls only the non-`_train` cores, so dropout is
  *structurally* unreachable at inference). Also confirmed the walk/reconciliation,
  residual scaling, named errors, the frozen-layer rule and the inline-SGD
  upward-import justification, and no syntax drift. Full text:
  `docs/plans/part-13-review-opus.md` (gitignored).

Nit triage (the only actionable finding from either reviewer):

- **NIT-1 — the `zero_grad`/`apply_sgd` coverage test spot-checked only 4 of 26
  Parameters (Opus). FIXED.** `test_zero_grad_and_apply_sgd_reach_every_parameter`
  filled all 26 grads to 1.0 but asserted only four values moved (no bias among
  them), so a future edit dropping e.g. `sgd_parameter(self.ln_f.bias, lr)` from
  `apply_sgd` would slip through — defeating the purpose of a coverage test. Not a
  current defect (all three walks are complete, verified by both reviewers).
  Strengthened: the test now snapshots *every* Parameter tensor (weights and
  biases, in the walk order), and after a step with all grads at 1.0 asserts every
  entry moved by exactly `-lr`. A skipped `sgd_parameter` call now leaves its
  tensor unmoved and fails. Re-ran green.

Both reviewers independently confirmed the two genuinely-new pieces (the tied
two-path gradient and the attention-dropout backward composition) correct by
re-derivation. No math was wrong; the single fix hardened a test.
