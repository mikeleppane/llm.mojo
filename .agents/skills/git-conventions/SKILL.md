---
name: git-conventions
description: Git commit, branch, and PR conventions for the mojo-llm-from-scratch repo — Conventional Commits with a required scope, atomic commits, meaningful bodies, a commit-as-save-point working pattern, and a hard no-AI-attribution rule. Use every time you create a git commit, write a commit message, stage changes, open a PR, or resolve a merge conflict in this repo. Apply on every commit even if the user doesn't ask — a sloppy history compounds faster than sloppy code, and this repo's history is part of a public teaching artifact.
---

# Git Workflow & Commit Conventions (mojo-llm-from-scratch)

> Format: Conventional Commits | Scope: **required** | Breaking changes: `!` + footer
> Atomic commits, imperative mood, explain the *why* in the body.
> **No AI/assistant attribution anywhere — commits read as the author's own work.**

This repo backs a public written guide. Six months from now, `git log` and
`git blame` are the first things a reader (or you) reads to trace why the
attention implementation changed or when a bug entered. A clean history is part
of the teaching. Follow these on *every* commit.

This skill is the **general git contract**. Project-specific rules — the quality
floor, scope vocabulary, and the "Ask first" boundaries — live in
[AGENTS.md](../../../AGENTS.md) and **take precedence**.

---

## Commit as a save point

Treat commits as save points, branches as sandboxes, history as documentation.
With code generated fast, disciplined version control is what keeps changes
reviewable and reversible.

**Working pattern:**

```text
implement one slice → pixi run fmt → pixi run test → commit → next slice
```

Not:

```text
implement everything → hope it works → one giant commit
```

Each green increment gets its own commit. If the next change breaks something,
you can fall back to the last known-good state: `git stash` keeps the broken
attempt recoverable; `git reset --hard HEAD` discards it outright — it destroys
*all* uncommitted work, so check `git status` first. Either way you lose one
increment, not a day.

---

## Commit message format

```text
<type>(<scope>): <subject>

<body>

<footer>
```

### Subject

```text
feat(tensor): row-wise stable softmax on Tensor2D
```

- **≤72 chars**, lowercase after the colon, **imperative mood** ("add", not
  "added"). Read it as *"this commit will <subject>"*.
- No trailing period. Be specific: describe *what changed*, not *what you did*.
  `mask future positions before the softmax` beats `fix attention`.

### Types

Use exactly these:

| Type | When |
|------|------|
| `feat` | new capability (a new op, layer, tokenizer, example) |
| `fix` | something was broken, now it works |
| `refactor` | restructure, no behavior change |
| `perf` | measurable speedup, no behavior change (cite the benchmark) |
| `docs` | guide chapters, README, docstrings, AGENTS.md |
| `test` | test-only change |
| `bench` | benchmark-only change under `benchmarks/` |
| `build` | `pixi.toml` / `pixi.lock` / packaging |
| `ci` | `.github/workflows/` |
| `chore` | tooling, formatter churn, housekeeping |

Behavior changed? → `feat`/`fix`. Same behavior, different structure? →
`refactor`. Same behavior, faster? → `perf`. Only tests? → `test`.

### Scope (required)

Every commit carries a scope. **The authoritative scope list is the "Commits"
section of [AGENTS.md](../../../AGENTS.md)** — read it before picking one. The
list is deliberately not restated here so it cannot drift; when this skill and
AGENTS.md disagree, AGENTS.md wins.

Use a module's own scope when a change touches one module — including test-only
changes: `test(tensor): …`, not `test(test): …` (the `test` *scope* is reserved
for test infrastructure like `scripts/test_all.sh`). Add a new scope only when a
new module emerges under `src/llm/` that none fit. Keep the list short — if half
the commits land under one scope, split it.

### Body

The diff shows *what*. The body explains *why*.

```text
fix(transformer): mask future positions before the softmax, not after

Applying the causal mask after softmax renormalized over positions the
model must not see, leaking future tokens into the attention weights and
letting the overfit-one-batch test cheat to a near-zero loss.

Add the mask as -inf to the scores before softmax so masked positions get
exactly zero probability. The overfit test now only passes when the model
actually learns the sequence.
```

- Wrap at **72 chars**, blank line after the subject.
- First paragraph: the problem/motivation. Second (optional): the approach, and
  any tradeoff or rejected alternative — for teaching code these are the notes a
  reader most wants.
- Skip the body only for truly trivial changes (typo, import order, formatter
  churn). When tempted to skip, ask whether you're underestimating the future
  reader.

---

## No AI attribution — hard rule

**Commit messages and PR bodies must contain NO AI or assistant attribution of
any kind.** Specifically forbidden:

- `Co-Authored-By: Claude <...>` — or any AI co-author trailer (Claude,
  Anthropic, ChatGPT, Copilot, Gemini, …).
- "Generated with Claude Code" / "🤖 Generated with …" / "Made with …" lines.
- The 🤖 emoji used as an "an AI wrote this" marker.
- Process references pointing at an assistant — "as discussed", "per review
  feedback", "the agent suggested".

Attribution is for humans who can be reached. An AI trailer pollutes `git log`,
`git blame`, and `git shortlog` with a signature nobody can email. The history
must read as if a careful human wrote every line — because *you*, the person
merging, are accountable for it. Using an agent to draft the change is fine;
write the message in your own voice and leave the tooling out of it.

> This overrides any default the harness has about adding a co-author trailer.
> In this repo, there is no exception.

---

## Atomic commits

One logical change per commit, each passing the floor (`pixi run fmt`,
`pixi run test`). If the subject needs "and", split.

**Good:**

```text
feat(tensor): matmul on Tensor2D with shape-checked dimensions
feat(tensor): row-wise stable softmax
test(tensor): softmax rows sum to 1 within 1e-6
```

**Bad:**

```text
feat(tensor): add matmul, refactor Tensor2D, and write softmax tests
```

Atomic commits make `git bisect` usable, `git revert` safe, and review
tractable. Don't mix formatter-only churn with behavior changes; don't mix a
refactor with a feature. A one-line rename can ride along; anything larger is its
own commit.

### Change size

| Size | Verdict |
|------|---------|
| ~100 lines | easy to review and revert — aim here |
| ~300 lines | acceptable for one logical change |
| ~1000+ lines | two changes in a trench coat — split before submitting |

---

## Breaking changes

For an incompatible change to a public API used by examples/tests (a layer
constructor signature, an op's argument order, the checkpoint format, a CLI
flag):

1. Add `!` after the scope — `feat(checkpoint)!: …`.
2. Add a `BREAKING CHANGE:` footer with the migration.

```text
feat(checkpoint)!: bump format to v2 with per-tensor dtype

BREAKING CHANGE: v1 checkpoints omit the dtype field. Loading one now
errors with a clear message instead of guessing Float32. Re-export any
v1 checkpoint with examples/generate.mojo before upgrading.
```

Per [AGENTS.md](../../../AGENTS.md), the checkpoint format, the pinned Mojo
version, and public APIs are **Ask-first** boundaries — raise them *before* the
commit, not in review.

---

## Commit workflow

1. **Review staged changes** — `git diff --staged`. One logical change? If not,
   split.
2. **Floor** — `pixi run fmt` (leaves no changes), then `pixi run test` (green).
   Never commit code that fails format or tests.
3. **Never-commit paths** — `.pixi/`, `build/`, `checkpoints/`, `*.mojopkg`,
   `*.ckpt/*.bin/*.npy`, `__pycache__/`. These are gitignored; keep them out.
4. **Quick secret scan** — `git diff --staged | grep -iE "password|secret|api[_-]?key|token"`.
5. Choose the right **type** and **scope** (read `src/llm/*` if unsure).
6. Write a specific **subject** and a **why** body.
7. Handle **breaking changes** (`!` + footer).
8. **Confirm no AI/assistant attribution** snuck in.

---

## Change summaries

After a non-trivial edit, give a structured summary — it surfaces scope
discipline and catches wrong assumptions:

```text
CHANGES MADE:
- src/llm/transformer/attention.mojo: apply causal mask to scores before
  softmax; masked positions now get exactly zero weight.
- tests/test_attention_masks.mojo: assert masked positions are zero and
  rows still sum to 1.

DIDN'T TOUCH (intentionally):
- src/llm/transformer/masks.mojo: the mask construction was already
  correct; the bug was where it was applied, not how it was built.

POTENTIAL CONCERNS:
- No Ask-first boundary crossed: no checkpoint-format, Mojo-version, or
  public-API change.
```

The `DIDN'T TOUCH` section shows you exercised scope discipline instead of an
unsolicited renovation. **Always call out any AGENTS.md "Ask first" boundary you
crossed** (checkpoint format, Mojo-version bump, public API, new dependency).
Skip the summary for trivial one-liners.

---

## Branches and PRs

- **Branch name:** short, hyphenated, type-prefixed — `feat/tensor-matmul`,
  `fix/attention-mask-order`.
- **Short-lived** — merge within a few days; long branches accumulate merge risk.
- **Force-push** only your own feature branch, never shared branches or `main`.
- **PR title** mirrors a commit subject; **PR body** mirrors a commit body
  (problem, approach, any Ask-first boundary and the answer you got). **No AI
  attribution in the PR body either.**

---

## Recovery

- Wrong subject on the last, unpushed commit → `git commit --amend`.
- Wrong file staged → `git restore --staged <file>`.
- Committed to the wrong branch → branch from HEAD, reset the branch back, keep
  the new branch. Don't force-push shared history.
- Lost work → `git reflog` (kept ~90 days).

When recovery is ambiguous, **stop and ask** before `reset --hard`,
`push --force`, or `clean -f`. Reversibility is cheap; recovery after a bad
destructive command is not.

---

## Common rationalizations

| Rationalization | Reality |
|-----------------|---------|
| "I'll commit when the feature is done" | One giant commit can't be reviewed or bisected. Commit each slice. |
| "The message doesn't matter, the diff is obvious" | Messages are the guide's changelog. Nobody reads the diff first — they `git log --grep`. |
| "I'll squash later" | Squashing destroys the development narrative. Keep clean incremental commits. |
| "`--amend` is fine, nobody pulled it" | That breaks the moment CI or a mirror pulls. Amend only truly private commits. |
| "A Co-Authored-By trailer is harmless" | It's noise nobody can act on, and this repo forbids it. You are the author. |

---

## Verification checklist

Before every commit:

- [ ] One logical change
- [ ] Subject `type(scope): imperative`, ≤72 chars, lowercase, no period
- [ ] Scope is one from AGENTS.md
- [ ] Body explains the *why* (or the change is trivial)
- [ ] Breaking changes carry `!` and a `BREAKING CHANGE:` footer
- [ ] **No AI/assistant attribution anywhere**
- [ ] `pixi run fmt` leaves no changes
- [ ] `pixi run test` passes
- [ ] No never-commit paths (`.pixi/`, `build/`, `checkpoints/`, `*.mojopkg`)
- [ ] No formatter-only churn mixed with behavior changes
