---
name: code-review-and-quality
description: Multi-axis code review for the mojo-llm-from-scratch repo before merge — your own code, another agent's, or a teammate's PR. Adds shape/numerical-fidelity, syntax-drift, and teaching-clarity checks on top of the standard correctness / readability / architecture / performance review. Use whenever you are about to merge, or when asked "is this ready?", "review this", "check this", "look this over". Reviewing AI-generated Mojo is a stronger trigger, not a weaker one — obsolete-syntax and plausible-but-wrong math are the dominant failure modes.
---

# Code Review & Quality (mojo-llm-from-scratch)

Multi-axis review for this repo. The product is **understanding you can trust**,
so a review here checks two things a generic reviewer misses: that the *math is
right and stable*, and that the *code still teaches*. The output is a
**structured Markdown report** — findings grouped by severity, each with a
`file:line` reference and a quoted snippet, then a clear verdict.

This is the *review* moment. Companion skills own the *production* rules; when a
finding is "this violates rule X", **cite the owning skill, don't restate it**:

- [mojo-coding-guidance](../mojo-coding-guidance/SKILL.md) — shapes, numerics,
  error handling, allocation, module boundaries.
- [test-driven-development](../test-driven-development/SKILL.md) — failing-test-first,
  oracle/golden/overfit, tolerance policy.
- [git-conventions](../git-conventions/SKILL.md) — commit shape, scope, no-AI-attribution.
- [improve-architecture](../improve-architecture/SKILL.md) — layering and depth.
- the global **`mojo-syntax`** skill — the authority on current Mojo syntax.

Project rules live in [AGENTS.md](../../../AGENTS.md) and override this skill.

The triage prompts below name *what to look for*; they are not the rules. File
findings against the source skill or AGENTS.md.

---

## Before reading a line

1. **Reproduce the floor.** `pixi run fmt-check` and `pixi run test-fast` (the
   canonical gate; see AGENTS.md) must be green. A red suite is finding #1 —
   stop and report it. *Exception:* for a **docs-only diff** (Markdown,
   docstrings, comments — nothing under `src/`, `tests/`, `examples/`,
   `benchmarks/`), skip the test run; `fmt-check` and a read are enough. Run the
   suite the moment the diff touches code.
2. **Read the diff with its commit messages.** Does each commit do one thing with
   an honest scope and a *why* body? Does a `refactor`/`perf` commit actually
   preserve behavior (no moved golden test, no changed logits)?
3. **Know the blast radius.** Did the change touch an Ask-first boundary
   (checkpoint format, pinned Mojo version, a public API used by examples/tests, a
   new dependency)? If so, was it raised?

---

## The axes

Walk the diff once per axis. For each, the triage prompts are starting points,
not a script.

### 1. Correctness

- Does it do what the commit says, on the stated shapes and on the edge shapes
  (empty sequence, `T=1`, single head, `batch=1`)?
- Off-by-one in indexing, loop bounds, sequence positions? Causal mask covering
  exactly the right positions?
- Integer vs float division; truncation where a float was meant?
- Error paths: does an invalid shape/config **raise a clear error**, or crash /
  silently produce garbage?

### 2. Shape & numerical fidelity (this repo's signature axis)

- **Do the shape comments match the code?** A wrong shape comment is worse than
  none — it teaches the reader a lie. Trace the dimensions by hand.
- Is softmax / log-sum-exp / layer-norm computed **stably** (max subtracted,
  epsilon present and documented)? A naive `exp` overflows on real logits.
- Is dtype used deliberately? Today model *and* reference math are both
  `Float64`; `Float32` is planned and currently appears only at the
  released-weights boundary (exact `f32 → f64` widening on load). Flag code or a
  comment that describes model math as `Float32` before that narrowing lands.
- Any float compared with `==` **for a numerical result**? (Exact equality is
  correct for the exactness contracts — RNG replay, checkpoint round-trip,
  gradient doubling, zero-preservation, integer counts — so don't flag those;
  flag a *widened* exactness test, which deletes a guarantee.) Any magic constant
  (an epsilon, a scale, a `sqrt(d_head)`) undocumented?
- For a backward pass: is there a **finite-difference gradient test**? Unverified
  gradients are the #1 quiet bug — treat a missing one as a High finding.

### 3. Readability & teaching value

- Would a reader following the guide understand this without running it? Names
  descriptive (short symbols only next to a shape comment)?
- Any clever trick that saves three lines but hides a concept? In teaching code,
  clarity beats cleverness — flag it.
- Public API has a real docstring stating shapes / mutate / allocate / raise?

### 4. Architecture

- Does every new import point **down** the dependency graph? Judge against the
  **authoritative full graph** in
  [AGENTS.md](../../../AGENTS.md#dependency-layering--one-direction-only) — it
  includes `config → transformer/training`, the `data → models → training`
  branch, and the quarantined `lab` leaf, not just the
  `utils → tensor → nn → transformer → {training, generation}` spine. An "up"
  import or a cycle is a High finding — cite
  [improve-architecture](../improve-architecture/SKILL.md).
- Is new code in the right home (`src/llm` vs `examples` vs `tests` vs
  `benchmarks`)? Core logic duplicated into an example, or an experiment landed in
  `src`?
- Does `__init__.mojo` re-export the intended surface and hold no top-level code?

### 5. Mojo currency & safety

- **Any obsolete syntax?** `fn`, `let`, `alias`, `@parameter`, `inout`/`owned`/
  `borrowed`, non-`std.` imports, stdlib `Tensor[T]`. Cite the `mojo-syntax`
  skill. This is the most common defect in generated Mojo — check it explicitly.
- `raises` present iff the function can raise?
- `UnsafePointer` owners: explicit origin, allocated with `alloc`, freed in
  `__del__`? Any leak or double-free path?
- Correct `.copy()` / `^` transfer for non-`ImplicitlyCopyable` types?

### 6. Performance (only after correctness)

- Any accidental quadratic where linear was meant; reallocation inside a hot
  loop; a copy where a `Span`/reference would do?
- Is an "optimization" actually measured? A `perf` commit needs a benchmark under
  `benchmarks/`; an untested optimization that muddies the code is a net loss —
  flag it and ask for the number.

---

## The syntax-drift gate (guide code)

This repo backs a written guide, so review the *docs* too when the diff touches
them: every ```mojo``` block a reader will copy must pass the syntax contract and
should have a runnable counterpart in `examples/` or `tests/`. Pseudocode must be
fenced as ```text```, not ```mojo```. A guide chapter with unverified `mojo`
blocks is not mergeable — see AGENTS.md *Publishing*.

---

## Reviewing AI-generated code

False confidence is the dominant failure mode. Generated Mojo tends to: emit
last-year's syntax; write shape comments that don't match the code; produce a
softmax that looks right but skips the max-subtraction; add a backward pass with
no gradient test; and write commit messages with an AI trailer this repo forbids.
Check each explicitly — polish is not correctness.

---

## Severity and output

Group findings by severity; each carries `file:line` and a quoted snippet.

| Severity | Meaning |
|---|---|
| **Critical** | wrong results, data loss, crash, or a leaked/garbage checkpoint |
| **High** | unverified gradient, wrong shape comment, unstable numerics, up-graph import, obsolete syntax that will break |
| **Medium** | missing test for new behavior, undocumented magic constant, readability that hurts the guide |
| **Low** | naming, a redundant copy, a docstring gap |
| **Nit** | subjective; label it as such |

End with a **verdict**: *approve*, *approve with nits*, or *request changes*, and
the one or two findings that gate the merge. Prefer few high-confidence findings
over a long list; a review the author trusts is one they act on.
