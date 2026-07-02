---
name: test-driven-development
description: Test-driven development for the mojo-llm-from-scratch repo. Use whenever you write or change observable behavior — a new tensor op, a layer, a bug fix, a refactor of anything a test can see. Write the failing test first; reproduce a bug with a test before fixing it. Apply on every behavioral change, not only when the user asks for tests. Covers this repo's TestSuite mechanics, the oracle / golden / overfit-one-batch patterns, finite-difference gradient checks, and the Float32/Float64 tolerance policy. Defers to mojo-coding-guidance for how the code under test is written and to the global mojo-syntax skill for syntax.
---

# Test-Driven Development (mojo-llm-from-scratch)

Write the failing test before the code. For a bug, reproduce it with a test
*before* fixing it. Tests are proof — "seems right" is not done. In a
from-scratch numerics project the bugs are quiet: a softmax that sums to 1.01, a
gradient with a flipped sign, a mask applied one step too late. Every test here
locks a correctness invariant, not a coverage number.

This skill covers the *process* and the *shape* of a good test here. It does not
restate how the code under test must be written — cite
[mojo-coding-guidance](../mojo-coding-guidance/SKILL.md) for that, and the global
`mojo-syntax` skill for test syntax. Project rules live in
[AGENTS.md](../../../AGENTS.md) and override this skill.

---

## The TestSuite mechanics

Tests are ordinary `def ... raises` functions discovered by `TestSuite` and run
with `mojo run` (the `mojo test` subcommand was removed). Every test file ends
with the same runner:

```mojo
from std.testing import assert_equal, assert_true, assert_almost_equal, assert_raises, TestSuite
from llm.tensor import Tensor2D, softmax_rows


def test_softmax_rows_sum_to_one() raises:
    var probs = softmax_rows(some_scores())
    for r in range(probs.rows):
        assert_almost_equal(row_sum(probs, r), 1.0, atol=1e-6)


def test_bad_shape_raises() raises:
    with assert_raises():
        matmul(a_2x3, b_2x2)  # inner dims disagree


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

Run everything (smoke test first, then the suite), or one file while iterating:

```bash
pixi run test
pixi run mojo run -I src tests/test_softmax.mojo
```

One file per unit under test, named `tests/test_<thing>.mojo`. `scripts/test_all.sh`
picks up new `test_*.mojo` files automatically — no list to maintain.

---

## The test pyramid

Many cheap unit tests, some component tests, a few expensive integration tests.

| Layer | Examples |
|---|---|
| **unit** (many) | tensor indexing/bounds, matmul, softmax rows, mask construction, config validation, RNG determinism |
| **component** (some) | attention forward shapes, a transformer block forward, embedding + norm |
| **integration** (few) | overfit-one-batch, greedy-generation smoke, checkpoint save/load round-trip |

Categories worth a dedicated file: `smoke`, `config` (invalid configs raise),
`tokenizer` (encode/decode round-trip), `tensor`, `softmax`/`cross_entropy`
(math), `random` (determinism), `attention_masks`, `generation` (deterministic
greedy), `checkpoint` (save/load round-trip).

---

## Three patterns worth naming

### Oracle tests — compare against a trusted reference

Calculus, `Float64` hand-math, or a tiny NumPy computation is the oracle. The
**finite-difference gradient check** is the canonical one: the analytic gradient
must match `(f(x+h) - f(x-h)) / 2h`.

```mojo
def test_grad_matches_finite_difference() raises:
    var h = 1e-4
    var analytic = grad_wrt_x(f, x)
    var numeric = (f(x + h) - f(x - h)) / (2.0 * h)
    assert_almost_equal(analytic, numeric, atol=1e-4)  # Float32 tolerance
```

If the analytic gradient disagrees with the oracle, the backward pass is wrong —
this catches the single most common ML bug before it poisons training.

### Golden tests — freeze a known-good output

For a fixed tiny input and a fixed seed, freeze the logits (or generated tokens)
and assert future runs match within tolerance. Golden tests catch accidental
behavior changes during refactors — a "no behavior change" `refactor` commit that
moves a golden test is lying. Regenerate a golden deliberately, in its own
commit, with the reason in the body.

### Overfit-one-batch — the highest-value integration test

A correct model with a correct training loop drives the loss on *one* small batch
to near zero. If it can't, something is broken — usually the loss, the gradient,
or the optimizer, in that order of suspicion. Make this a **test**, not a manual
check:

```mojo
def test_overfits_single_batch() raises:
    var model = tiny_model(seed=0)
    var batch = one_small_batch()
    var loss = train_steps(model, batch, steps=200)
    assert_true(loss < 0.05)  # near-zero: the model memorized one batch
```

When it fails, it is the most informative failure in the repo — trust it over
your reading of the code.

---

## Tolerance policy

**Never compare floats with `==`.** Size the tolerance to the precision:

| What | Tolerance |
|---|---|
| `Float64` reference / hand math | `1e-9` – `1e-12` |
| `Float32` model math | `1e-4` – `1e-5` |
| finite-difference gradient check | `~1e-4` (limited by `h`, not by `Float32`) |

Pass `atol=` (and `rtol=` when comparing large magnitudes) explicitly so the
tolerance is visible and reviewable. **Loosening a tolerance to make a red test
green is an Ask-first action** (AGENTS.md) — a widened tolerance usually hides a
real regression, not floating-point noise.

---

## Determinism

- Thread a **seed** through anything random (init, dropout, sampling). A test
  that can't reproduce its input can't localize a failure.
- Greedy generation is deterministic — assert the exact token sequence.
- Keep tests independent: no shared mutable global state between tests; each test
  builds its own inputs.

---

## Process

1. **Write the failing test first.** For a bug, reproduce it (Prove-It): the test
   fails *for the reason you claim*, then your fix makes it pass. A bug fix
   without a test that would have caught it is not done.
2. **Smallest input that shows the behavior.** A 2×3 matrix, a 4-token sequence,
   a 2-layer tiny model. Small inputs make failures readable and tests fast.
3. **One invariant per test**, named for what it locks
   (`test_softmax_rows_sum_to_one`, not `test_softmax`).
4. **Run the floor** — `pixi run fmt`, `pixi run test` — before declaring done.

---

## Checklist

- [ ] Test written *before* the code (or before the fix, for a bug)
- [ ] It fails without the change, passes with it
- [ ] Smallest input that demonstrates the behavior
- [ ] Floats compared with `assert_almost_equal` + an explicit tolerance sized to the dtype
- [ ] New math has an oracle (hand-calc, finite differences, or NumPy)
- [ ] Randomness is seeded; greedy paths asserted exactly
- [ ] A refactor commit does not move a golden test
- [ ] `pixi run test` green; the new file follows `tests/test_<thing>.mojo`
