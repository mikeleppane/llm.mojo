# Part XIX — The gauntlet: systematic validation and CI

Three parts in a row deferred the same debt: XVI validated the real weights
against ONE prompt, XVII's 124M gate reused it, XVIII rewrote every hot kernel and
verified against — still — that one prompt plus the doll-house suite. Part XIX
builds the machinery that ends the deferral: a multi-prompt 124M validation
gauntlet (the standing release gate) and CI that is mechanically correct instead
of one push from a compiler stall. **No model code changed** — the gauntlet found
no bug on its first run (documented below), so `src/llm` is untouched.

## The two tiers, formalized

- **Tier 1 — `pixi run test`** (hermetic, CI): the full doll-house suite, no
  weights, no network. Proves the CODE.
- **Tier 2 — `pixi run gauntlet`** (local release gate): the 124M harness against
  `checkpoints/gpt2-124m.bin`, failing with a converter-pointing error if the
  weights are absent. Proves the MODEL.

Merge doctrine (now in AGENTS.md): a part is mergeable when Tier 1 is green, the
gauntlet is green, and `fmt-check` passes.

## The #6554 exclusion flip (a behaviour change, stated loudly)

Before: `pixi run test` RAN `tests/test_seq_tasks.mojo` (the #6554 stall) by
default; `test-fast` (`SKIP_SLOW=1`) skipped it and was the real gate. The
mechanics contradicted the standing rule that "green" excludes it.

After: `scripts/test_all.sh` EXCLUDES the `SLOW_6554` list by default and prints a
loud per-file line every run; `RUN_SLOW=1` opts back in. pixi tasks:

- `test` — canonical gate, skips the slow file (loud SKIPPED line);
- `test-fast` — retained alias of `test` (muscle memory / older docs);
- `test-full` — `RUN_SLOW=1`, the toolchain-upgrade check (NOT the gate; it can
  hang). AGENTS.md records the trigger: on a Mojo version bump, run it once; if
  #6554 is fixed upstream, retire the exclusion list entirely.
- `test-full` was deliberately NOT executed in this part (running it hits the very
  stall it isolates); it exists for the next toolchain bump.

The exclusion list lives in exactly ONE place (`scripts/test_all.sh`), commented
with the bug number. The flip changes which files RUN, not any test's content.

**Flip confirmed green.** Tier 1's old canonical gate (`test-fast`) already
excluded the slow file, so the flip renames the default rather than changing the
set the gate runs. `pixi run test` under the new default: **67 test files run,
"All tests passed."**, with the loud line:

```
==> tests/test_seq_tasks.mojo  SKIPPED (Mojo #6554) — RUN_SLOW=1 (pixi run test-full) to include
...
SKIPPED 1 #6554-slow file(s): tests/test_seq_tasks.mojo
All tests passed.
```

(A first run failed spuriously — a contiguous alphabetical block of tests hit
`failed to resolve parent package body` because the editor's Mojo LSP recompiled
`build/llm.mojopkg` mid-run, the exact "don't stack concurrent cold compiles"
hazard AGENTS.md warns about. Re-run with nothing else touching Mojo: clean green.
Not a flip defect.)

## The gauntlet — three pieces

Following the Part XVI pattern (offline Python writes / committed text / native
Mojo reads):

1. **`data/gauntlet/prompts.txt`** — 15 curated prompts, one `=== id: <name> ===`
   record each with an inline rationale after the closing `===`. Format documented
   in the file header. Both parsers split on `\n` ONLY (never a unicode-aware
   `splitlines`, which would treat CJK/exotic codepoints as line breaks) and drop
   the single file-terminating empty line, so the two sides see byte-identical
   records. No body line collides with the separator convention.
2. **`scripts/gpt2_gauntlet_reference.py`** — offline golden generator, NumPy +
   stdlib only. REUSES the existing oracles rather than reimplementing them:
   `tests/oracles/gpt2_reference_encoder.py` (OpenAI's BPE) to encode, and
   `scripts/gpt2_reference_logits.py`'s `read_gpt2w` + `forward` (the f64 forward
   over OUR `.bin`) — no second BPE, no second forward. Writes
   `data/gauntlet/goldens.txt` with a `sha256=<hash> — do not hand-edit` header.
   Determinism self-check (a second pass compared byte-for-byte) runs by default.
3. **`examples/gpt2_gauntlet.mojo`** — the Tier 2 harness. Loads the model ONCE;
   per prompt: (a) `GPT2Tokenizer.encode` == golden ids EXACT; (b) full-forward
   probe logits @ 1e-6, argmax + top-5 ids EXACT; (c) mean next-token NLL via
   `GPT.loss(ids[:-1], ids[1:])` @ 1e-6 (skipped for the single-token prompt);
   (d) for a short subset, `generate` vs `generate_cached` token-for-token EXACT.
   Any failure names the prompt id, the check, and the offending value, then exits
   non-zero.

**Parity discipline:** cross-implementation checks stay at LOGIT level with a
tolerance; token-SEQUENCE exactness is only ever our-vs-our (generate vs
generate_cached). A cross-implementation greedy-text golden would flake on genuine
near-ties; probes @ 1e-6 plus exact argmax/top-5 ids give the same protection with
no flake channel.

## Prompt inventory — why each earns its slot

| id | tokens | what it exercises |
|----|--------|-------------------|
| `short-english` | 8 | the XVI continuity prompt; plain ASCII prose (also greedy subset) |
| `english-prose` | 29 | longer plain English across several sentences |
| `contractions` | 35 | BPE-adversarial: `Don't`/`'Tis`/`y'all'd've`, a rare long word (greedy subset) |
| `accented-european` | 48 | diacritics (`Café`, `Zürich`, `Kraków`) — byte-level BPE on Latin-1 |
| `cjk` | 55 | Chinese + Japanese + Korean; multi-byte codepoints, no spaces |
| `emoji` | 36 | emoji incl. a ZWJ family sequence and a skin-tone modifier |
| `code-snippet` | 43 | source code: indentation, symbols, no natural-language spacing |
| `digits-punct` | 33 | digit runs and punctuation runs |
| `url-email` | 38 | symbol-dense, space-free spans (a URL and an email) |
| `whitespace-mixed` | 18 | leading spaces, a tab, doubled internal spaces, a trailing newline |
| `newline-heavy` | 30 | multi-paragraph text with blank lines |
| `single-token` | 1 | the shortest legal input; NLL undefined (`none`), greedy subset |
| `finnish` | 65 | agglutinative non-English with long compound word forms |
| `near-context` | 1000 | a long tinyshakespeare slice, near the context window |
| `exact-1024` | 1024 | **the context-length boundary** the sliding window and cache-capacity guard |

The two long prompts are slices of `data/tinyshakespeare/input.txt` (already
committed), trimmed with the tokenizer oracle to hit exactly 1000 and 1024 tokens.
Coverage spans: plain/long English, BPE-adversarial contractions, three
non-Latin scripts + emoji, code, numbers/punctuation, space-free symbol spans,
every whitespace class, the shortest and the boundary-length inputs, and a
non-English natural language. The greedy-parity subset is the three shortest
(`short-english`, `contractions`, `single-token`).

## Goldens provenance and determinism

- `.bin` sha256 = `e6f6ccacec40b9e64e246b2d1073e3bcc52537e0ebe5fe80e886b50f6fafb1f3`
  (pinned in the goldens header). Note: part-16-notes recorded the sha256 of the
  548 MB **safetensors source** (`248dfc39…`), not the 498 MB `.bin`; that source
  hash still matches on disk, so provenance is intact — the `.bin` was regenerated
  from the same verified source and its own hash was never previously recorded.
- Determinism self-check: **two passes byte-identical**. Re-verified after the red
  test: the restored `goldens.txt` is byte-identical to a fresh `--skip-self-check`
  regeneration (`GOLDENS MATCH FRESH REGEN`).
- Cross-check: the `short-english` probes reproduce the independently-frozen Part
  XVI parity goldens to the digit (`logit[0] = -103.75580955455361`, argmax 407) —
  the gauntlet's oracle agrees with the existing single-prompt gate.

## Red-test-first evidence (the gate actually goes red)

Corrupted one golden line (`short-english` probe `0:-103.75580955455361` →
`0:-999.0`) and ran the harness. It went red, naming prompt + check + values, and
the process exited non-zero:

```
Unhandled exception caught during execution: GAUNTLET FAILED [short-english] probe[0]: got -103.75580955455058, golden -999.0 (gap 895.2441904454495)
mojo: error: execution exited with a non-zero result: 1
```

Incidental finding: the real Mojo-vs-reference gap on that probe is
`|-103.75580955455058 − -103.75580955455361| ≈ 3e-12` — comfortably inside the
`1e-6` bar, confirming the tolerance is neither too tight nor too loose.

## Full gauntlet run — GREEN

`pixi run mojo run -I build examples/gpt2_gauntlet.mojo`, wall-clock **30.9 s**
(includes the ~2 GB model load, 15 forwards, the NLL second-forwards, and the
greedy subset; 485% CPU — threaded post-XVIII):

```
PASS  short-english  T=8  argmax=407  probes=9  nll=4.003301000867618  greedy=ok
PASS  english-prose  T=29  argmax=198  probes=9  nll=4.415431523350696
PASS  contractions  T=35  argmax=198  probes=9  nll=4.3351386163135865  greedy=ok
PASS  accented-european  T=48  argmax=198  probes=9  nll=3.8205531980762877
PASS  cjk  T=55  argmax=198  probes=9  nll=2.4065670453797536
PASS  emoji  T=36  argmax=37929  probes=9  nll=3.2158113838203843
PASS  code-snippet  T=43  argmax=198  probes=9  nll=3.4146819084701834
PASS  digits-punct  T=33  argmax=405  probes=9  nll=5.140678525442255
PASS  url-email  T=38  argmax=198  probes=9  nll=3.9496312205732877
PASS  whitespace-mixed  T=18  argmax=198  probes=9  nll=8.030731788920267
PASS  newline-heavy  T=30  argmax=198  probes=9  nll=3.0227222575411252
PASS  single-token  T=1  argmax=198  probes=9  nll=   n/a  greedy=ok
PASS  finnish  T=65  argmax=198  probes=9  nll=4.7813484793188525
PASS  near-context  T=1000  argmax=198  probes=9  nll=4.2721819978339886
PASS  exact-1024  T=1024  argmax=329  probes=9  nll=4.271336177044599

GAUNTLET OK — 15/15 prompts matched the float64 reference (tokens/argmax/top5 exact, probes & nll @ 1e-6); generate vs generate_cached agreed on 3 short prompts.
```

**No bug surfaced.** The from-scratch Mojo port matches the f64 reference across
the whole input space — adversarial unicode tokenization is exact, the numerics
hold at 1e-6 including the 1024-token boundary, and the KV-cache path agrees
token-for-token with the uncached path on real text. The gauntlet paid for itself
as a standing gate, not (this time) as a bug detector — which is the outcome we
want after three parts of single-prompt validation.

## CI hardening

- `.github/workflows/ci.yml` inherits the exclusion flip (its test step now runs
  `pixi run test`, the excluding default) and gains a **Build examples** step:
  `pixi run build-examples` compile-checks every file in `examples/` and
  `benchmarks/`. Output (all 11 examples + 3 benchmarks): `All examples and
  benchmarks build.`
- `scripts/build_examples.sh` uses `mojo build --emit object` (full
  parse/typecheck/codegen, no link). Reason: this toolchain's `mojo build` link
  step omits libm, so linking any file that pulls in `exp`/`tanh` fails with an
  irrelevant `expm1` undefined-reference — the SAME failure on the existing
  `gpt2_parity_check.mojo`, so it is environmental, not our code (`mojo run`
  JIT-links fine, which is why the examples still RUN). Emitting an object runs
  the whole compiler and stops before that link. CI stays inert until a remote
  exists — explicitly not this part's problem; the deliverable is that the first
  push finds CI correct and not stalling.

## Deviations from plan

- **`--emit object` for build-examples** instead of a linked executable — forced
  by the environmental libm link failure above; the compile-check (parse +
  typecheck + codegen) is what catches bit-rot, and it runs fully.
- **`test-fast` kept as an alias** (not removed) — either was acceptable; keeping
  it avoids breaking older docs/muscle memory at zero cost.
- **NLL via `GPT.loss`** (the existing public surface) rather than a new
  reduction, per the handoff's "prefer reusing it". Costs one extra forward on the
  ≤1024-token prompts; absorbed in the 30.9 s budget.
- **`.bin` sha256 clarification** — the recorded XVI hash was the safetensors
  source, not the `.bin`; documented above and the current `.bin` hash pinned in
  the goldens header.

## Review triage

Dual external review (Codex GPT-5.5 high + Claude Opus 4.8 xhigh) over
`git diff main...part-19-gauntlet`. Findings and fix/reject decisions:

_(to be filled after the review pass.)_
