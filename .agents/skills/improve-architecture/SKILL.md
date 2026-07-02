---
name: improve-architecture
description: Explore the mojo-llm-from-scratch codebase, surface architectural friction, and propose module-deepening refactors as actionable plan documents under docs/plans/. Use when asked to review architecture, "make this cleaner", "reduce coupling", "this file is doing too much", "the modules are just thrown there", or to evaluate whether a module pulls its weight. Tuned to this repo's one-directional layering (utils → tensor → nn → transformer → {training, generation}), its per-concern package split, and its shape-documented public surfaces. Produces a durable refactor plan, not edits — execution is a separate, approved step.
---

# Improve Architecture (mojo-llm-from-scratch)

Explore the codebase organically, surface architectural friction, and propose
**module-deepening refactors as durable plan documents** under `docs/plans/`.
This is teaching code: a boundary that hides the math badly is a boundary that
makes the guide harder to follow. Architecture work here serves clarity and
testability, not tidiness for its own sake.

A **deep module** (Ousterhout, *A Philosophy of Software Design*) has "a small
interface hiding a large implementation." Deep modules are more testable, more
navigable for humans and agents, and let you test at the boundary instead of
poking internals. This skill finds *shallow* modules — interface nearly as
complex as implementation — and the plainer case this repo will hit: files that
are individually fine but **piled into a package with no clear boundary**, where
the friction is navigational.

Project rules live in [AGENTS.md](../../../AGENTS.md) and override this skill. For
the per-edit coding contract, cite
[mojo-coding-guidance](../mojo-coding-guidance/SKILL.md); this skill is about
*structure*, not line-level style.

**This skill produces a plan, not edits.** Execution is a separate, approved
step — refactors touch many files and each is its own atomic commit.

---

## The invariant to protect: one-directional layering

The dependency graph flows one way. Lower layers never import higher ones:

```text
utils  →  tensor  →  nn  →  transformer  →  { training, generation }
config + vocab  →  nn, transformer
tokenizer  →  data  →  training
```

The first thing to check in any architecture pass: **are all imports pointing
down?** An "up" import (`tensor` importing `nn`, `utils` importing anything) or a
cycle is the highest-priority finding — cycles make code impossible to test in
isolation and impossible to explain in order. The fix is almost always to move
the shared thing *down* to the layer that both callers can reach, or to invert
the dependency so the lower layer exposes a hook the higher layer fills.

---

## Friction to look for

Walk the tree and the imports. Common findings, roughly by value:

1. **Up-graph imports / cycles** — as above. Always a plan item.
2. **A file owning two responsibilities.** `transformer.mojo` holding attention
   *and* masks *and* positional encoding. Splitting usually reveals a hidden
   dependency you can then make explicit (masks don't depend on attention;
   attention depends on masks).
3. **A shallow module.** A wrapper whose interface is as wide as its body — e.g.
   a "layer" that just forwards three calls and exposes all three. Either deepen
   it (hide the sequencing) or inline it.
4. **A leaky public surface.** `__init__.mojo` re-exporting internal helpers, or
   callers reaching past the package (`from llm.tensor.ops import _row_max`)
   because the clean name isn't exported. Fix the surface, not the callers.
5. **Duplicated core logic in `examples/`.** An example reimplementing a training
   step instead of calling the library — the library is missing an entry point.
6. **A struct exposing raw fields** that callers mutate directly, so the invariant
   (a tensor's shape matching its buffer length) can't be enforced. Deepen behind
   methods.
7. **Flat pile with no boundary.** Twelve files in one package with no
   sub-grouping and no re-exported surface — the friction is navigational; the
   fix is a package split + `__init__.mojo`, not a rewrite.

---

## What "deeper" looks like here

- **Interface stays small, implementation grows.** `softmax_rows(scores) ->
  probs` is deep: one call hides the max-subtraction, the exp, and the
  normalization. Good.
- **Test at the boundary.** If testing a module forces you to construct its
  internals, the boundary is in the wrong place. A deep module is tested through
  its public function on small inputs (see
  [test-driven-development](../test-driven-development/SKILL.md)).
- **The `__init__.mojo` is the contract.** Re-export the names callers should use
  (`from .ops import matmul, softmax_rows`) so files can move inside the package
  without breaking `from llm.tensor import matmul`.
- **Shapes are the interface too.** A public tensor function whose shape contract
  is unclear is shallow *for a reader* even if the code is fine. Deepening
  sometimes just means stating the shapes and hiding the index arithmetic.

---

## The exploration → plan flow

1. **Map before judging.** List the packages under `src/llm/`, read each
   `__init__.mojo`, and sketch the actual import graph (grep the `from llm.`
   lines). Compare it to the intended layering above.
2. **Collect friction**, not fixes yet. Note each smell with a `file:line` and
   one sentence on why it costs the reader or the tests.
3. **Cluster into refactors.** Group related smells into a handful of named
   refactors, each independently shippable. A good refactor has a clear before →
   after and a way to prove behavior is unchanged (a golden or overfit test that
   passes on both sides).
4. **Write the plan** to `docs/plans/<short-name>.md`:
   - **Problem** — the friction, with evidence (`file:line`, the import that
     points up, the responsibility that's doubled).
   - **Proposed structure** — the target layout / interface, and why it's deeper.
   - **Migration** — the ordered atomic commits (`refactor(<scope>): …`), each
     green on its own.
   - **Risk & proof** — what could break, and the test that guards it. Call out
     any AGENTS.md Ask-first boundary the refactor would cross (a public API
     rename → `!` commit; a checkpoint-format touch → confirm first).
   - **Explicitly out of scope** — what you are *not* changing, so the plan
     stays reviewable.

---

## Guardrails

- **Don't rename for taste.** A rename churns `git blame` and the guide's
  cross-references. Rename only when the current name actively misleads.
- **No premature abstraction.** Two similar functions are cheaper to read than
  one clever generic. Extract a shared helper on the *third* occurrence, not the
  second, and only if the abstraction has a name a reader recognizes.
- **No GPU/SIMD restructuring dressed up as architecture.** Performance work is a
  separate track behind a benchmark (AGENTS.md); don't fold it into a "cleanup".
- **Keep the plan small.** A plan proposing to move ten files at once is a plan
  nobody will execute safely. Prefer several small plans over one grand redesign.
- **Match the size of the fix to the size of the problem.** A navigational pile
  needs a package split, not a redesign; a cycle needs one dependency inverted,
  not a new layer.
