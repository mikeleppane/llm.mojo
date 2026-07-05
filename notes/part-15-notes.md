# Part XV — Generation: build notes

The part where the model finally speaks. Its defining property: **there is no new
math below the top layer.** `GPT.forward` (ids → logits), `softmax_row_temperature`
(stable at extreme T), `argmax` (first-wins ties), and `sample_categorical`
(inverse-CDF, one draw) all already exist and are tested. Part XV is pure assembly
in `generation/` — two distribution filters, one policy struct, and the
autoregressive loop — which made **scope discipline** the quality bar: every layer
below `generation/` stayed frozen, absolutely.

## Probability space vs logit space for the filters

The two standard ways to write top-k / top-p:

1. **Logit space:** set the filtered-out logits to `-inf`, then softmax once. The
   dropped tokens get probability exactly 0 because `exp(-inf) = 0`.
2. **Probability space** (chosen here): softmax first, then zero the filtered-out
   probabilities and renormalize.

They are algebraically identical — normalizing `exp(logit)` over the kept set is
the same whether you zero before or after the exponential. The reasons for
probability space are about the *codebase*, not the math:

- **Every intermediate is a valid distribution.** `sample_categorical` already
  validates its input (non-negative, sums to 1 within 1e-6). Feeding it a filtered
  distribution turns that guard into a free integrity check on the filters — a bug
  that failed to renormalize would raise at the sampler, not silently skew draws.
- **No ±inf ever enters the codebase.** This project has kept every float finite
  since Part III; `-inf` logits would be the first exception, and `inf - inf = NaN`
  is one careless max-subtraction away.
- **Each filter tests as distribution → distribution** against a NumPy-free Python
  oracle. The goldens are ordinary probability vectors, not logits-with-holes.

The one tie rule, shared everywhere (top-k, top-p, and the oracle): **equal
probabilities → lower index survives**, matching `argmax`'s first-wins. It lives in
a single `_order_by_prob_desc` (a bottom-up merge sort ordering indices by
`(prob desc, index asc)`), which both filters call. The comparator avoids an `==`
on floats: "i before j" is `probs[i] > probs[j] or (not (probs[j] > probs[i]) and
i < j)` — if neither is strictly greater they are equal and the index decides. The
sort is O(V log V); this matters not at all at V = 11 but a great deal at V = 50257
in the BPE part, where an O(V²) scan over the vocab would be a real defect.

**top-p's disabled check must precede the cumulative sum.** `p >= 1.0` is the
disabled sentinel and is returned as the identity *before* any cumsum runs.
Deciding "disabled" via `cumsum >= 1.0` instead would let a distribution whose mass
rounds to 0.999… silently drop its tail. The explicit `if p >= 1.0: return copy`
is the honest gate. (`p > 1` is out of range and raises; `p == 1.0` is the exact
disabled value, so `SamplerConfig`'s `top_p = 1.0` preset means "off".)

## Greedy consumes zero rng draws

`temperature == 0.0` is the greedy sentinel — `sample_next` returns `argmax(logits)`
and **draws nothing**. This extends the house invariant dropout established
("disabling randomness never perturbs the seeded generator") to decoding: switching
a run to greedy cannot shift any other seeded draw stream, so a greedy `generate`
can be dropped anywhere into a seeded pipeline and leave `rng.state` bit-identical.
The sampled path draws **exactly one** uniform per emitted token (the single draw
inside `sample_categorical`). Both counts are pinned by tests bit-comparing
`rng.state` before and after — a greedy path that secretly drew, or a sampled path
that drew twice, would break the exact replay of any mixed pipeline.

Greedy is encoded as a *point in policy space*, not a separate mode. Rejected: a
separate `generate_greedy` entry point (two loops to keep in sync) and a
`greedy: Bool` field (redundant with the sentinel, and it admits the contradictory
state `greedy=True, temperature=0.7`). `softmax_row_temperature` raises on `T <= 0`,
so `0.0` has no valid sampled interpretation — an unambiguous sentinel.

## Stop tokens and the context crop

**Stop is append-then-halt.** When an *emitted* token is in `stop_tokens`, it is
appended to the output and *then* the loop halts. The output's last element records
why it stopped; a caller that doesn't want it strips one element. Dropping it
instead would make "hit a stop token" and "budget exhausted at the same length"
indistinguishable from outside. A stop id that appears only in the **prompt** never
halts anything — only emitted tokens are checked. Membership is a linear scan (stop
lists hold 0–3 entries; no set structure is warranted).

**The context crop is a sliding window: the LAST `min(len, context_length)`
tokens.** Without it, step `context_length + 1` would raise on the positional
embedding's bounds. The off-by-one here is silent quality rot, not a crash (a crop
that took `context_length + 1` *would* crash on the wpe bounds; one that cropped
from the front or dropped a token would just degrade), so it gets a dedicated
**equivalence test**: a prompt at exactly `context_length`, generate 4 more, and
each emitted token must equal an independent manual forward over the hand-cropped
window.

**`generate` binds to nothing but the `GPT` struct** — no tokenizer type, no
Shakespeare config, no checkpoint knowledge. It is ids-in / ids-out, so the BPE
part reuses it verbatim with real GPT-2 weights and `END_OF_TEXT_ID = 50256` as the
stop token. The per-step full-forward recompute (O(T²·steps)) is deliberate and
documented — that waste is the KV cache's motivation two parts on.

## The LCG-replay oracle trick (chapter material)

Sampling tests here are **exact integer equality, never statistical** — no flaky
frequency assertions. The trick: the Python oracle (`tests/oracles/
sampling_reference.py`) replays the *same* LCG the Mojo `Rng` uses (Knuth MMIX
constants, already Python-oracle-verified in Part VI's `test_rng`) and the same
inverse-CDF walk, so for a fixed seed + fixed logit row it predicts the *exact*
token ids `sample_next` must emit. Pure-Python IEEE-754 doubles match Mojo's
`Float64`, and softmax is implemented the same stable way (subtract the row max);
token ids are integers over wide CDF buckets, so ulp-level differences between the
two `exp` implementations cannot flip a draw. The first draw is even hand-worked in
a test comment: `Rng(42).uniform()` = 0.5682303266439076 (the top 53 bits of the
first LCG state, `10481999410520546993`, scaled by 2⁻⁵³ — the state `test_rng`
pins), whose threshold lands in the argmax bucket `[0.2071, 0.7701)`, giving id 1.
The oracle derives *nothing* from the Mojo code; the goldens are frozen inline.

## The four-policy demo (the payoff)

`examples/generate_shakespeare.mojo` loads the Part XIV checkpoint (parameters
only — the optimizer state the loader returns is discarded), rebuilds the char
tokenizer deterministically from the corpus, and continues a one-newline prompt
under four policies. The 346k-parameter model (trained ~330 steps to val
perplexity ~11.7) is char-plausible, not fluent — expected at this scale — but the
**safe → surprising axis is exactly visible**:

```
===  greedy (argmax)  ===
The the the the the the the the the the the the the the the the the the ...

===  temperature 0.8  ===
The meis coued mor ke nenanco pr f wino, that wan mele.
Goner, miceed ted benthat:
I t theasthigriserees t y ced mopyo lou wis therde besl teard l seincessutas ...

===  top-k 40 (T=1.0)  ===
TTheandar tod, los irancl mes no d winie th tave?
IRINCGORENRDENI HINNO:
CERe Monge t tharithiessisteding y bed kinwo itr wis therde'chth ...

===  top-p 0.9 (T=1.0)  ===
SLOLEIINI, war ise he men mau no d winigate t wan lllin isonde the
Brenofe f sthatene t thangrgo thisthes tethame, taye ito wis te, is chth s ore ...
```

Greedy collapses into the textbook `"The the the the …"` degenerate loop — the
argmax of a low-entropy model is a fixed point. The sampled policies escape it,
growing progressively more varied (and more error-prone) as the truncation loosens.
That single screen *is* the README's "trade off between safe and surprising."

## Deviations from plan

- **The capstone continuation is 6 tokens, not 8.** The plan said "~8 tokens." The
  memorize-then-speak model learns *position-dependent* next-token mappings (GPT has
  learned positional embeddings). Generating 8 tokens from a 2-token prompt crosses
  the `context_length` (8) boundary, and the sliding-window crop then shifts tokens
  to positions the overfit model was never trained at — so the 8th token legitimately
  diverges from the naive cycle. Keeping the continuation within one context window
  (prompt 2 + 6 emitted = 8) tests "memorize then speak" cleanly; the crop boundary
  and its position shift are exercised separately in `test_context_crop_equivalence`,
  where a position-agnostic manual replay is the reference. This is not a weaker test
  — it is the correct scope for each of the two properties.
- **`filter_top_p` raises on `p > 1`** (only `p == 1.0` is the disabled identity).
  The plan's prose said both "p >= 1.0 is the identity" and "raises on p > 1"; the
  consistent reading is `p == 1.0` disabled, `p > 1` out of range — which is what
  the code and `SamplerConfig.validate` both enforce.
- **A private `_renormalize` helper** was factored out of both filters (divide by
  the kept mass, raise if it is non-positive). Not in the plan's signatures, but it
  removes duplicated arithmetic and gives one honest "the input was not a valid
  distribution" error site.
- **The example generates 200 tokens, not 300** — purely to keep the uncached
  CPU demo comfortably under a couple of minutes; the recompute cost is the point,
  and the constant is documented as such.

## Mojo lessons this part

- **A temporary cannot bind to a `mut` argument** (already in AGENTS.md, hit again):
  `generate(gpt, prompt, …, Rng(0))` fails because an rvalue has no mutable storage.
  Even greedy calls, which never draw, need a named `var rng` bound first. The tests
  use a tiny `_rng()` helper and bind its result to a local before each call.
- **`return probs.copy()` for the identity paths.** `List[Float64]` is `Copyable`
  but not `ImplicitlyCopyable`, so `return probs` errors; the disabled/identity
  branches copy explicitly.

## Review triage

Dual external review over `git diff main...part-15-generation`, both read-only,
both told test_seq_tasks is excluded from all runs (Mojo #6554) so they would not
flag the exclusion as a gap, and both asked to VERIFY THE SAMPLING SEMANTICS AND
CONTRACTS by re-deriving them (top-p nucleus rule, the shared tie rule, the
pipeline order and per-stage renormalization, the greedy-zero-draw / sampled-one-
draw invariant, the context crop, the stop semantics, the O(V log V) sort, the
example config match + discarded optimizer state, and the ids-only reuse contract).

Both reviewers independently re-derived the top-p nucleus rule, the shared tie
rule, the pipeline order, the greedy-zero / sampled-one draw contract, the context
crop, the stop semantics, the O(V log V) sort, the example config parity + discarded
optimizer state, and the ids-only `GPT` binding — and confirmed every one CORRECT.
Neither found a correctness bug in the generation code.

- **Codex (GPT-5.5, high): 0 code findings.** Its only flagged item was a
  "blocker" claiming the `AGENTS.md` / `PROGRESS.md` / `notes/` edits violate the
  frozen-layer rule. **Rejected:** those are documentation, and the frozen-layer
  rule (and the review header's allowed set `{generation/, tests, examples, docs}`)
  is about the code layers below `generation/` — `tensor/`, `nn/`, `transformer/`,
  `training/`, `config.mojo`, `models/`, `lab/`. Updating the progress ledger, the
  build notes, and recording the #6554 lesson are explicit deliverables of this
  part, not scope creep. Full text: `docs/plans/part-15-review-codex.md`.
- **Opus 4.8 (xhigh): 0 blockers, 1 should-fix, 3 nits.** Full text:
  `docs/plans/part-15-review-opus.md`.

Triage of the Opus findings:

- **S1 (should-fix) — FIXED.** The `p = 0.9` golden's *comment* in
  `test_sampling_filters.mojo` said both tied 0.05 tokens (i0 and i7) are dropped,
  but the (correct) golden keeps i0. The cause is a floating-point subtlety worth
  teaching: the descending-order running sum after i6 is
  `0.30+0.20+0.15+0.10+0.08+0.07 = 0.8999999999999999`, one ULP *under* 0.9, so the
  `>= p` test does not fire there and i0 is admitted (kept sum 0.95); only i7 is
  dropped. The implementation and golden were always right — only the prose lied.
  Rewrote the comment to explain the exact-FP-comparison point. (Exactly the
  "indict your comment, never the oracle" discipline: the number was correct.)
- **N1 (nit) — FIXED.** `test_context_crop_equivalence` built its oracle window
  with the same `start = len - context_length` arithmetic `generate` uses, so a
  shared off-by-one in that formula would be mirrored on both sides and pass.
  Rewrote the oracle crop to DROP from the front until the window fits — a
  genuinely different expression of "the last context_length tokens" — so the test
  now measures the crop rather than restating it.
- **N2 (nit) — rejected (documented).** The greedy path (`argmax(logits)`) assumes
  a non-empty row, where the sampled path would raise "empty distribution" on an
  empty one — an asymmetry. Not reachable: `generate` only ever feeds a full `[V]`
  row from `forward`, and `V >= 1` always (a zero-vocab model cannot be built). A
  guard here would be untestable dead code; `argmax`'s own docstring already
  documents the non-empty precondition. Left as-is.
- **N3 (nit) — rejected (documented).** `_renormalize` re-sums the kept vector that
  `filter_top_p` already accumulated while walking its prefix, one extra O(V) sweep
  per filter. It stays O(V) overall (dominated by the O(V log V) sort), and a single
  shared renormalize is more legible than threading a running sum out of each
  filter. Correctness before speed; the reviewer agreed it should not change unless
  generation latency ever matters. Left as-is.

After the two fixes, `pixi run fmt-check` is a no-op and the full `pixi run
test-fast` suite (test_seq_tasks excluded per #6554) is green.
