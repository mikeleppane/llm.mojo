# Part X — Attention: build notes

Raw material for the chapter: the decisions as built, the deviations from plan,
and the review triage. Ships on branch `part-10-attention`. Deliverables: two
column ops at the tensor layer, the `transformer/` package (additive masks, the
scaled-dot-product core, GPT-2's fused-QKV multi-head attention), four test
files, and a NumPy oracle. Forward passes only — backward is a later part. No
Transformer block, no residuals, no KV cache, no cross-attention *layer*, no
Tensor4D.

---

## One core, self- and cross-attention for free

`scaled_dot_product_attention(q [T_q, D], k [T_k, D], v [T_k, D_v],
mask [T_q, T_k])` keeps q separate from k/v, each with its own length. That
single choice makes it the cross-attention core: self-attention passes q, k, v
from one sequence; a later encoder-decoder lab passes q from the decoder and k/v
from the encoder — the function neither knows nor cares. The property is
guaranteed *now*, before any consumer exists, by testing the cross-shaped case
`T_q=3 != T_k=4` (oracle Case A) and pinning the output shape as `[T_q, D_v]`,
not `[T_k, ...]`.

## The core returns weights, not just output

`AttentionResult{output, weights}` exposes the post-softmax weights so tests
prove causality *directly* — every weight strictly above the diagonal is 0 —
rather than inferring it from the output, and so the chapter can visualize what
each query attends to. Recomputing the weights in a second path would test a
copy, not the code.

## No Tensor4D — `[B, H, T, D]` is loop discipline

Every operation in attention is a 2D matmul or a row softmax. B and H are loops;
the tile is `[T, D]`. `MultiHeadAttention.forward` takes one sequence `[T, C]`
and loops over `H` heads; the batch loop belongs to the block (a later part). A
`Tensor4D` would need its own ops and teach nothing — the shape discipline lives
in the loop structure and the `# per head: [T, D]` shape comments. Deferred, not
rejected: the performance part revisits a packed layout with benchmarks in hand.

## Additive masks, and why MASKED_SCORE is finite

Masks are additive `[T_q, T_k]` tensors: `0.0` attends, `MASKED_SCORE = -1e9`
blocks, composition is plain tensor `add` (causal + padding = sum). The value is
**finite on purpose**. With `-inf`, a fully-blocked query row makes the stable
softmax compute `exp(-inf - (-inf)) = exp(NaN) = NaN` and poisons everything
downstream. With `-1e9` the row stays finite and its weights still sum to 1 —
the load-bearing guarantee. A fully-blocked query is itself padding whose output
is never read, so finite-but-wrong beats a NaN landmine that surfaces parts
later. `-1e9` also dwarfs any real score (attention logits are
O(sqrt(d_head) · unit variance)), so a *partially* blocked row drives every
masked key to ~0 weight.

**The subtlety the review sharpened (see triage below):** because softmax is
shift-invariant, adding the *same* `MASKED_SCORE` to every key of a row leaves it
equal to `softmax(unmasked scores)`. That is *uniform only when the underlying
scores tie* — not the general rule. With composed masks (one cell
`MASKED_SCORE`, another `2·MASKED_SCORE`) a fully-blocked row can even lean
toward the least-masked key. The first draft's comment (and the plan's D3
wording) called this "degrades to uniform", which is only the tied-score special
case. The source comment and the tests now say the honest thing: *finiteness* is
the guarantee; uniform is a special case. Two tests pin both readings — a tied
case (query `q=0`, genuinely uniform) and a non-tied case (distinct dots,
explicitly not uniform, still finite and summing to 1).

## Pinned order in the core

`scores = q @ k^T` → scale by `1/sqrt(D)` with `D = q.cols` (the per-head width
`d_head`, **not** `d_model`) → add mask → `softmax_rows` → `@ v`. The order is a
classic bug surface, so it is documented in the docstring and pinned by tests:

- **1/sqrt(d_head), not squared, not d_model** (oracle Case B): q has a single 1
  in column 0, so the two dot products stay fixed at 8 and 2 while D goes 2 → 4;
  only the `1/sqrt(D)` factor moves the weights. A squared or `d_model` scale
  gives different frozen numbers.
- **Scale before mask** (oracle Case C): a *small finite* mask distinguishes the
  orders. `-1e9` is too large to tell "scale then add" from "add then scale"
  apart (the plan's own risk table notes `-1e9/sqrt(64)` is still `-1.25e8`), so
  the order test uses mask entries like `-1.0, -2.0, -3.0` where the two orders
  visibly diverge.

## Fused QKV, contiguous head split — GPT-2's layout

`MultiHeadAttention` is one `Linear(C -> 3C)` (GPT-2's `c_attn`) plus one
`Linear(C -> C)` (`c_proj`). The `[T, 3C]` projection splits into Q/K/V thirds,
and each third into `H` **contiguous** `[T, D]` head slices, `D = C/H` — head `h`
owns columns `[h·D, (h+1)·D)`, not an interleaved stride. This is the checkpoint
layout weight loading will fill, and its cost `3C^2+3C + C^2+C = 4C^2+4C` is the
first structural reconciliation of the architecture's committed attention
parameter arithmetic — the test sums the layer's *real* Parameter tensors, not a
formula.

## Column ops live at the tensor layer

Head split/merge needs `slice_cols(a, start, end)` and `concat_cols(parts)`.
They are general 2D utilities, so they live in `tensor/ops.mojo` with their own
round-trip tests, not private helpers buried in attention where a stride bug
would masquerade as a model bug. Both allocate; both raise on bad ranges /
mismatched row counts / empty lists.

## Deviations from plan

- **The "fully-blocked row degrades to uniform" framing (plan D3 / §5) was
  corrected to "degrades to finite; uniform only when scores tie".** The plan's
  wording is true only for the tied-score special case under additive masking;
  the implementation and both reviewers confirmed the general behavior is
  `softmax(unmasked scores)`. Tests pin both the tied case (uniform) and a
  non-tied case (not uniform), plus finiteness in both. This is a documentation
  correction, not a behavior change — the finite `MASKED_SCORE` and its NaN
  avoidance are exactly as planned.
- **Added a full-MHA forward oracle (Case E) beyond the plan's test list** to
  pin the contiguous head split for `H>1` (see triage). The plan's
  single-head-equivalence and causal-locality tests do not distinguish
  contiguous from interleaved.
- Everything else matches the plan's signatures and test list.

## Mojo lessons this part

- **A single struct field cannot be transferred out of a live value with `^`.**
  `head_outputs.append(result.output^)` fails with "field '…' destroyed out of
  the middle of a value". `.copy()` the field instead (or move the whole
  struct). Same rule that bit the `Parameter` factories in Part IX, now hit on
  `AttentionResult.output` inside the head loop. Recorded in AGENTS.md already;
  reconfirmed here.
- **Indexing a `List[Tensor2D]` element to bind a local copies it.**
  `var part = parts[i]` fails ("cannot be implicitly copied") because `Tensor2D`
  is `Copyable` but not `ImplicitlyCopyable`. Read the scalar fields you need
  (`parts[i].cols`) and index `parts[i][r, c]` directly instead of binding the
  element, or `.copy()` if you truly need the value.

## Review triage

Full dual review, read-only, non-interactive, over
`git diff main...part-10-attention`: **Codex GPT-5.5** (high reasoning,
`danger-full-access` sandbox) and **Claude Opus 4.8** (xhigh reasoning, separate
context). Raw outputs saved under `docs/plans/` (gitignored). Both verified the
core order, `1/sqrt(d_head)`, additive masking, the Q/K/V-thirds-before-heads
split, no `causal_mask` off-by-one, the independent (non-circular) oracle, and
zero Mojo syntax drift.

Findings and disposition:

1. **[both, should-fix] No test pins the `H>1` contiguous head split.** The
   single-head-equivalence test uses `H=1` (nothing to split), and causal row-0
   locality is split-invariant, so an interleaved split or a swapped head order
   passed the whole suite green — the contiguous layout was asserted only by a
   comment. **Fixed:** added oracle Case E — a full MHA forward (`H=2, C=4`,
   causal) with hand-built qkv/proj weights the NumPy reference rebuilds exactly;
   the frozen output uses a contiguous split and differs from the interleaved
   variant (also printed in the oracle for contrast). Verified the new test
   *fails* under a reordered head split and passes when correct.
2. **[Codex, should-fix] The `masks.mojo` source comment overclaimed "uniform".**
   **Fixed:** rewrote the comment to state finiteness as the guarantee and
   uniform as the tied-score special case; added the non-tied fully-blocked-row
   test asserting finite + sums-to-1 + not-uniform.
3. **[Opus, nit] `forward` traps (modulo-by-zero) on `n_heads == 0`** when a
   caller bypasses `init_random`'s guard via `@fieldwise_init`. **Fixed:** added
   an explicit `n_heads > 0` check at the top of `forward` so both entry points
   raise a catchable error symmetrically.

No findings rejected — all three were legitimate and all three were fixed. Gates
re-run green after the fixes.
