# Part XIV — Training: build notes

The part that turns the Part XIII model from "demonstrably learns" (a plain-SGD
overfit smoke) into "trains": AdamW with decoupled decay and the GPT-family
selective-decay partition, a warmup+cosine learning-rate schedule, global-norm
gradient clipping, versioned bit-exact checkpoints, and the real `train_gpt` loop
over Part VI's `BatchLoader`. The two design problems both come from the
project's founding constraints (no autograd, concrete structs, one-directional
layering), and both are chapter material.

## Where optimizer state lives: the walk order IS the parameter registry

AdamW carries two moment tensors (`m`, `v`) per parameter across steps. A
framework hides this behind a parameter dict; here the Parameters are scattered
inside nested concrete structs (`GPT` → `blocks[i]` → `attn.qkv.weight`, …). The
resolution (a user-approved decision): the model's **fixed traversal order is the
registry**. One documented walk — wte (once — weight tying means one Parameter),
wpe, then each block's 12 parameters in layer order (ln1 w/b, attn qkv w/b, attn
proj w/b, ln2 w/b, mlp up w/b, mlp down w/b), then ln_f w/b — serves every
consumer:

- `parameter_shapes()` sizes the optimizer state and the checkpoint;
- `apply_adamw(m, v, …)` indexes trainer-owned parallel `List[Tensor2D]` m/v in
  that order;
- `grad_norm()` / `scale_grads()` sum and scale every gradient in that order;
- `export_parameters()` / `import_parameters()` and `export_gradients()` copy in
  that order for checkpoint IO and inspection;
- the checkpoint format writes and reads floats in that order.

Optimizer state is **trainer-owned**, not stored in `Parameter`. Rejected
alternative: `m`/`v` fields inside `Parameter`. It would simplify the call sites
but taxes every non-training use (the lab, inference, a loaded model) with 2×
dead memory and couples Part IX's teaching struct ("a Parameter is a value and
its gradient") to one particular optimizer — and still would not solve ordering
for the checkpoint.

**Order drift between walk methods is the named failure mode.** The per-block
12-parameter walk lives once in `transformer/block.mojo` (each walk method
delegates to the block, mirroring the existing `zero_grad`/`apply_sgd`), so the
order is authored in one place. Three tests guard drift: `parameter_shapes`
reconciles with `parameter_count_actual` (a double-counted wte would inflate the
float total); the decay-partition inventory is pinned explicitly; and the
against-oracle 2-step run compares `apply_adamw` to a flat reference that drives
the (independently oracle-verified) `adamw_update` over the model's OWN published
walk metadata (`export_parameters` + `export_gradients` + `parameter_decay_flags`)
in a plain index loop. If `apply_adamw` visits parameters in a different order,
assigns a decay flag inconsistent with `parameter_decay_flags`, or threads its
m/v off-by-one, the two diverge — and the second step is essential because an
m/v off-by-one is invisible while all state is still zero.

## Where optimizer math lives: nn/optim.mojo (the layering resolution)

The dependency layering runs `nn → transformer → {training, generation}`, so
`GPT` (in `transformer/`) cannot import from `training/`. Part XIII had already
hit this: its `apply_sgd` performed `p -= lr·grad` via a `sgd_parameter` helper
*inlined in `transformer/block.mojo`* rather than importing
`training.optimizer.sgd_step` upward (the lab could call `sgd_step` because
`lab/` sits above `transformer/`; the main-line model cannot).

This part establishes the permanent home. The per-Parameter update math moves to
a new `nn/optim.mojo` — `nn/` owns `Parameter`, sits below `transformer/`, and is
a legal import target. `sgd_update(mut p, lr)` is the SGD step; `GPT.apply_sgd`
and `TransformerBlock.apply_sgd` now delegate to it, **behavior-frozen** (every
Part XIII test that exercises `apply_sgd` passes unchanged). `adamw_update`
joins it. `training/optimizer.mojo`'s free `sgd_step` (over a bare `Tensor2D`)
stays untouched — the bigram consumes it. So the two "sgd" functions coexist
honestly: `nn/optim.sgd_update` (Parameter-level, for the model) and
`training/optimizer.sgd_step` (Tensor2D-level, for the bigram).

## AdamW: decoupled decay, bias correction, selective decay

The per-tensor update (`nn/optim.adamw_update`, derived in its docstring):

```
m ← β₁·m + (1−β₁)·g            v ← β₂·v + (1−β₂)·g²
mhat = m/(1−β₁ᵗ)               vhat = v/(1−β₂ᵗ)          (t starts at 1)
value ← value − lr·( mhat/(√vhat + eps) + weight_decay·value )
```

Three things the tests are aimed at:

- **Bias correction starts at t=1.** With m=v=0, after one step m₁=(1−β₁)g and
  the denominator 1−β₁¹ = 1−β₁ cancels it exactly, so mhat=g and vhat=g²; the
  update's adaptive term is g/(|g|+eps) ≈ sign(g). A t<1 argument raises. An
  off-by-one (t starting at 0, or the correction skipped) is caught by the
  hand-computed step 1 and the multi-step oracle goldens.
- **Decay is DECOUPLED** — the "W" in AdamW. `weight_decay·value` is added to the
  update directly, never folded into g or the moments (that would be Adam+L2, a
  different algorithm). The observable pin: with g=0 the moments stay *exactly*
  zero yet the value still shrinks to value·(1−lr·wd). If decay were coupled
  through the gradient, g=0 would leave the value untouched.
- **Selective decay, GPT-family convention.** Weight decay applies to matrices
  (every Linear weight, wte, wpe) and never to vectors (every bias, every
  LayerNorm weight and bias). Per block that is 4 decayed matrices + 4 undecayed
  biases + 4 undecayed LN vectors; plus wte, wpe decayed and ln_f undecayed. The
  partition is pinned as an explicit inventory, and `apply_adamw` passes
  `weight_decay` to the matrices and `0.0` to the vectors — the same partition
  `parameter_decay_flags` reports.

Defaults (`AdamWConfig.gpt2_defaults()`): β₁ 0.9, **β₂ 0.95** (the GPT-training
value, NOT Adam's 0.999 habit — pinned by a test), eps 1e-8, weight_decay 0.1,
grad_clip 1.0.

## Schedule and clipping

`lr_at(step, peak, warmup, max_steps, min_lr)` is pure arithmetic: linear warmup
from 0 over `warmup` steps, then a cosine from peak to `min_lr` across
`[warmup, max_steps]`, `min_lr` held after. The warmup boundary is continuous
(cosine progress 0 at step == warmup gives exactly peak), and `warmup = 0`
degenerates cleanly to "start at peak" with no divide-by-zero (the warmup branch
is `step < warmup`, which is `step < 0` when warmup is 0, so the division is
unreachable). Every golden is hand-computable.

Clipping is the global (whole-model) L2 norm, not per-tensor:
`clip_grad_norm(gpt, max_norm)` computes `gpt.grad_norm()` (sqrt of the sum of
squares over every gradient entry, wte once) and, if it exceeds `max_norm`,
scales every gradient by `max_norm/norm` — bringing the global norm to exactly
`max_norm` while preserving direction. Below the threshold it is an exact no-op
(no tensor touched); a zero gradient never reaches the division. Applied AFTER the
batch's gradient accumulation and BEFORE the optimizer step. The hand-computed
test plants 3, 4, 12 on three DIFFERENT parameters and checks the single combined
norm is √169 = 13 (a per-tensor norm would report 3, 4, or 12, never 13).

## Checkpoint format (versioned, bit-exact)

`training/checkpoint.mojo`, format version 1:

```
line 0:          "GPTCKPT 1"                 magic + version
line 1:          N                            parameter tensor count
line 2:          t                            step counter
line 3:          <rng_state as 16 hex>        one UInt64 (our LCG state)
lines 4..4+N-1:  "rows cols"                  each parameter's shape, walk order
then, walk order and row-major within each tensor, one hex Float64 per line:
  parameters (N tensors), then m (N tensors), then v (N tensors).
```

Every Float64 is stored as its raw IEEE-754 bit pattern — a 16-digit hex UInt64
via `x.to_bits[DType.uint64]()` / `bitcast[DType.float64, 1](…)` — rather than a
decimal that must be re-rounded. `%.17g` round-trips *in principle*, but bit-hex
makes exactness **structural**: the resume gate is exact equality, not a
tolerance, so the round-trip must be exact, not approximately-parsed. `load`
validates the header against the live model's `parameter_shapes()` and raises with
a named mismatch on a bad magic, an unsupported version, a parameter-count or
shape disagreement, or a truncated file — it never reads garbage.

**The resume gate** (the strongest correctness gate a trainer can have): train k
steps → checkpoint → fresh (differently-seeded) model → load → train n−k more, and
the parameters are BIT-IDENTICAL to n straight steps. Run on the overfit-batch
setup (a fixed batch, no loader state), with the lr driven off the step index so a
mis-restored step counter would pick the wrong lr and diverge. If the gate ever
fails, something is unsaved or unrestored (rng state, t, a moment list) or the
walk order drifted — it is exact-equality by design and must never be weakened to
a tolerance.

Scope cut (documented): a shuffled loader's cursor is NOT part of the checkpoint,
so resuming a shuffled-loader run mid-epoch is only approximate. The exactness gate
runs where there is no loader state to lose. (For a *fixed* corpus, `train_gpt`
reconstructs the loader epoch/position deterministically from the step index, so
even a segmented run reproduces a straight run bit-for-bit — see below.)

## The training loop and the seed-derivation scheme

`train_gpt(mut gpt, mut train_loader, mut val_loader, tc, oc, sc, mut rng,
eval_interval, eval_batches, …)`. Per step: reshuffle the loader at each epoch
boundary → `zero_grad` → for each of the B sequences, `forward_cached(training)`,
`cross_entropy_rows_backward` scaled by 1/B, `backward` (grads accumulate across
the batch) → `clip_grad_norm` → `lr_at(step, …)` → `apply_adamw` (t = step + 1).
Every `eval_interval` steps `estimate_loss` reports the dropout-free `forward`
loss on both loaders; it evaluates from the start of the current epoch order and
**restores the loader cursor**, so it never disturbs training. `overfit_batch`
feeds the overfit path through this SAME loop (a one-batch loader) — no
special-cased trainer.

**Seeds (one seed reproduces a run end to end):** the training loader is
reshuffled at each epoch with `tc.seed + epoch` (Part VI's documented per-epoch
convention), and the dropout stream is the caller-supplied `rng` — a SEPARATE
generator seeded distinctly from `tc.seed` (the example offsets the seed by the
golden-ratio constant `0x9E3779B97F4A7C15`) so the two streams never coincide.
With dropout 0 the rng is never drawn and the run is fully determined by
`tc.seed` alone.

**Resumability (added for the example, tested on the overfit batch).** `train_gpt`
takes optional `start_step` / `end_step` (run any sub-range under ONE schedule
horizon, so segmenting a run does not change the lr at any step), `init_m` /
`init_v` (carry the AdamW moments across an interruption), and returns the final
m/v in `TrainReport`. The loader epoch/within-epoch position is reconstructed from
the step index (each epoch yields `num_batches()` batches, so epoch =
`start_step // num_batches`). A test pins that a two-segment run — [0, k) then
[k, n), moments carried across — reproduces a single [0, n) run bit-for-bit. This
is what lets the example checkpoint periodically and resume through the same
trainer instead of a special path.

## The example, and its tuning history

`examples/train_gpt_shakespeare.mojo`: char-level tiny Shakespeare through
`CharTokenizer` + `BatchLoader`, a mini-GPT, a full warmup+cosine run in segments
with periodic checkpoints and eval logging, a final checkpoint, and a
load-and-resume demo. **No sampling** — generation is Part XV, and its first demo
will be this checkpoint speaking.

Tuning history (the plan suggested "≈4 layers, d_model 128, T 64, tuned on the
branch"): at scalar Float64 on CPU that config runs ~6–8 s per optimizer step
(measured: a 6-step run of d_model 128 / 4 layers / T 64 / batch 16 took ~108 s
wall including compile), so a meaningful step count is 10–15+ minutes — past the
"minutes on CPU" target. The example was tuned down to d_model 96 / 3 layers /
4 heads / T 48 / batch 12 (~346k parameters, ~1 s/step), which trains clearly in
a few minutes. This is the deliberate scale trade the plan called for; the trainer
machinery is identical at any size (the 124M path is Part XVI).

Final reported run (SEED 1337, 330-step horizon interrupted at 300, warmup 30,
peak lr 3e-3 -> min 3e-4, batch 12, dropout 0.1; **5:50 wall-clock on CPU
including compile**):

```
vocab 65 chars, train 1,003,855 tokens, val 111,539 tokens
initial:  train loss 4.180  ppl 65.4   (a uniform 65-token model)
step  50: train 2.691  val 2.745  val ppl 15.6   lr 2.97e-3
step 100: train 2.573  val 2.628  val ppl 13.8   lr 2.66e-3  [checkpoint]
step 150: train 2.502  val 2.566  val ppl 13.0   lr 2.08e-3
step 200: train 2.464  val 2.530  val ppl 12.5   lr 1.38e-3  [checkpoint]
step 250: train 2.420  val 2.487  val ppl 12.0   lr 7.6e-4
step 300: train 2.400  val 2.472  val ppl 11.8   lr 3.7e-4   [checkpoint, interrupt]
```

The load-and-resume demo restores from the step-300 checkpoint into a
differently-initialized fresh model and reports the **identical** val loss
(2.4718109811387374 — bit-for-bit, so the parameters loaded exactly), then
finishes the run `[300, 330)` from the restored moments, step counter, and rng —
under the same 330-step cosine, so the lr continues without a jump — landing at
val ppl 11.70. Loss fell smoothly the whole way. No hyperparameter thrash was
needed: every gradient and the optimizer are oracle-tested upstream, so the
end-to-end run converging on the first honest config is the expected outcome.

## Deviations from plan

- **`nn/optim.mojo` reconciled with Part XIII's inlined SGD.** The plan (D1)
  assumed `apply_sgd` might already import `sgd_step`; XIII had instead inlined
  `sgd_parameter` in `block.mojo` to respect the layering. Both `apply_sgd`
  methods now delegate to `nn.optim.sgd_update` (the layering-legal home),
  behavior-frozen; the `sgd_parameter` helper is removed. Same arithmetic, same
  walk order.
- **`train_gpt` takes two loaders + eval params, and gained resume params.** The
  plan's representative signature showed one `loader`; eval-on-train-and-val needs
  a train and a val loader, and `eval_interval`/`eval_batches` are explicit
  arguments (config.mojo untouched). `start_step`/`end_step`/`init_m`/`init_v` and
  the m/v in `TrainReport` were added to make the trainer resumable (so the
  example checkpoints and resumes through it, not a special path). All additive
  with defaults; existing call sites are unchanged.
- **`export_gradients` added** (symmetric to `export_parameters`) to drive the
  against-oracle walk-consistency check and for per-layer gradient inspection.
- **`clip_grad_norm` lives in `training/optimizer.mojo`** (grouped with the
  optimizer machinery) rather than a new module; `grad_norm`/`scale_grads` are the
  `GPT` methods it composes.
- **`grad_norm` is non-raising** (sqrt of a sum of squares cannot fail), where the
  plan's representative signature marked it `raises`.
- **Example scale reduced** from the plan's 128/4/64 to keep the CPU run in
  minutes (above).

## Mojo lessons this part

- **`mojo format` rejects a plain reassignment to a variable named `out`**
  (`out = expr`) even though the compiler accepts it — `out` is the argument
  convention keyword, and the formatter's parser treats `out = …` as a syntax
  error (`Cannot parse`). `out += …`, `out[i] = …`, and `return out` all format
  fine; only the bare reassignment trips it. Renaming the local (here `acc`) fixes
  it. A file can compile and pass tests yet fail `fmt-check` for this alone.
- **`var d: UInt64` without an initializer, assigned in later branches, also
  upset the formatter** in the same function; folding the branch logic into the
  reassignment (compute the digit inline) sidestepped it.
- **Float64 ↔ bit pattern:** `x.to_bits[DType.uint64]()` and
  `bitcast[DType.float64, 1](SIMD[DType.uint64, 1](bits))[0]` (from
  `std.memory`). `Float64.from_bits` does not exist on the pinned Mojo.
- **`String.split("\n")` yields `StringSlice`, not `String`**, and always leaves a
  trailing empty element for a newline-terminated file. `load_checkpoint` owns the
  slices as `String`s and drops the trailing empty so a truncated file runs out of
  lines cleanly (surfacing as "truncated") rather than as a blank-line parse error.
- **Default `List[T]()` arguments work** (`init_m: List[Tensor2D] =
  List[Tensor2D]()`), which is what lets the resume params stay optional.
- **A `List[Tensor2D]` field of a returned struct copies, never transfers:**
  `report.train_losses^` fails ("destroyed out of the middle"); `.copy()` is the
  sanctioned path (the AGENTS.md single-field rule, seen again).

## Review triage

Dual external review over `git diff main...part-14-training`, both read-only, both
asked to VERIFY THE OPTIMIZER SEMANTICS AND CONTRACTS by re-deriving (AdamW
decoupled decay + bias correction, the selective-decay partition, the global-norm
clip, the single-order parameter walk, the bit-exact checkpoint/resume, the
layering). Neither found a correctness blocker; the AdamW math, decoupled/selective
decay, clipping, the walk registry, and the resume gate all held under
re-derivation.

- **Claude Opus 4.8 (xhigh): 0 blocker, 1 should-fix, 5 nits, 12 confirmed-correct.**
  Full text: `docs/plans/part-14-review-opus.md` (gitignored).
- **Codex (GPT-5.5, high): 0 blocker, 2 P2, 2 P3.** Full text:
  `docs/plans/part-14-review-codex.md` (gitignored).

Triage (both reviewers), fix/reject per finding:

- **Opus S1 — FIXED.** `test_checkpoint.mojo`'s shape-mismatch test wrote
  `GPTConfig(8, 16, 8, …)  # C=16` but the constructor order is
  `(vocab, context_length, d_model, …)`, so slot 2 sets *context_length*, not
  d_model — the test passed only because a wider context reshapes `wpe`. Changed
  to `GPTConfig(8, 8, 16, …)  # d_model 16` so the first mismatch is at tensor 0
  (`wte`) as the docstring claims, and corrected the comment. A real
  mis-teaching, not just a label slip.
- **Opus N1 — FIXED.** Three `GPTConfig` comments labelled the positions
  `V, C, ctx` when the true order is `V, ctx, C` (all values were 8, so
  numerically inert). Relabelled.
- **Opus N2 / Codex-implicit — FIXED.** The example printed `perplexity(0.0)`
  (= 1.0, a *perfect* model) as the "uniform model" baseline. A uniform model
  scores loss `log V`, perplexity `V`; now prints `vocab_size` directly.
- **Codex P2a — FIXED.** `save_checkpoint` checked only `len(m)/len(v)`, not each
  moment tensor's shape; a same-length but wrong-shaped state would serialize a
  payload that no longer matches the header shapes, and load would shift values
  between tensors. Added a per-tensor shape check before writing, with a test
  (`test_save_rejects_misshaped_moment`).
- **Opus N5b — FIXED.** `load_checkpoint` read exactly `N` tensors × 3 sections
  but ignored *extra* trailing lines. Added a "file consumed exactly" check
  (`cursor == len(lines)`) so a file with extra data is rejected, with a test
  (`test_load_rejects_trailing_garbage`). Truncation (the dangerous direction) was
  already caught.
- **Codex P2b — FIXED (example restructured).** The resume demo saved the
  checkpoint under a 300-step schedule horizon but resumed under a 330-step one,
  so `lr_at(step, …, max_steps)` used a different cosine denominator on the first
  resumed step — an lr jump, not a seamless continuation. Restructured the example
  to ONE horizon (`TOTAL_STEPS = 330`): it trains to `CHECKPOINT_STOP = 300` in
  segments, then the resume finishes `[300, 330)` under the same `tc` — a genuine
  interrupt-and-continue with no lr discontinuity (the same property
  `test_segmented_resume_matches_straight_run` proves in the suite).
- **Codex P3a — FIXED.** The example printed the segment's final lr for every eval
  in a segment (a segment can hold multiple evals). Now indexes
  `report.lrs[eval_step - segment_start]` so each eval shows its own lr.
- **Codex P3b — FIXED.** A code comment in `gpt.mojo` carried the internal marker
  `(D2)`, which AGENTS.md bans (private plan references dangle in the public repo).
  Removed the marker, kept the explanation. (Pre-existing `D5` markers in the Part
  XI backward tests are out of scope for this branch and untouched.)

Rejected / left as-is, with justification:

- **Opus N3 (decay partition authored twice).** `apply_adamw` hard-codes each
  parameter's decay as a literal `0.0`/`weight_decay` argument while
  `parameter_decay_flags` hard-codes the same partition as `True`/`False`. In
  Mojo's concrete-struct world there is no clean way to derive one from the other
  without a parallel reference-yielding walk (which the layout does not support).
  The two are cross-checked by `test_apply_adamw_matches_flat_walk_two_steps` (the
  flat reference drives `adamw_update` from `parameter_decay_flags`; a drift
  diverges the run), so it is a guarded maintenance smell, not a correctness risk.
  Kept, as Opus concluded ("no change required").
- **Opus N4 (new heavy tests not in `SLOW_6554`).** The full-model suites are NOT
  #6554-slow: measured real wall-clock is ~0.85 s (`test_checkpoint`) and ~1.2 s
  (`test_gpt_train_loop`) — the multi-second numbers in `TestSuite`'s per-test
  display are the known-unreliable timer (AGENTS.md). So they do not belong in
  `SLOW_6554` and `scripts/test_all.sh` is left untouched.
- **Opus N5a (`adamw_update` trusts the beta range).** `nn/optim` is deliberately
  low-level per-tensor math; `AdamWConfig.validate()` (called by `train_gpt`)
  enforces `beta ∈ [0, 1)` at the config edge. Adding a redundant beta check to
  the inner loop buys nothing the config guard does not already give. Left; the
  docstring's raising contract (t and shapes) is accurate for this layer.
