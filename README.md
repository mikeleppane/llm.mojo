# Build a Large Language Model from Scratch in Mojo

A from-scratch, educational implementation of a small decoder-only Transformer
language model, written in [Mojo](https://mojolang.org/docs/manual/) — tokenizer,
tensors, attention, training loop, and generation, with **no ML framework
underneath**. The focus is correctness, explainability, testing, and (later)
performance engineering. It is the companion codebase to a written guide.

## What this is

- A readable, tested path from a bigram toy to a working tiny GPT.
- Every tensor operation carries its shapes; every claim of correctness has a test.
- Pure Mojo — the point is to understand each piece by building it.

## What this is not

- Not a production framework, not a PyTorch clone, not a race for benchmarks.
- Not GPU-first: correctness on CPU comes first; optimization comes later, behind
  a benchmark.

## Status

Early scaffolding. The toolchain, test harness, and CI are in place; the library
under `src/llm/` is being built chapter by chapter.

## Install

Everything runs through [pixi](https://pixi.sh):

```bash
pixi install            # set up the environment from pixi.lock
pixi run mojo-version   # print the pinned Mojo version
```

## First commands

```bash
pixi run hello          # examples/hello.mojo
pixi run test           # run the test suite (smoke test first)
pixi run fmt            # format all Mojo sources
```

Run a single test or example with the `llm` package on the import path:

```bash
pixi run mojo run -I src tests/test_smoke.mojo
```

## Repo map

```text
src/llm/      the library — one package per concern (tensor, nn, transformer, ...)
examples/     runnable demonstrations
tests/        correctness checks (TestSuite)
benchmarks/   performance measurement (after correctness)
docs/         the written guide chapters
scripts/      dev scripts (test_all.sh)
```

Contributor and agent conventions live in [AGENTS.md](AGENTS.md); deeper,
task-specific guidance is under [`.agents/skills/`](.agents/skills/).

## Learning path

Follow the guide chapters under `docs/` in order: setup → tokenization → tensors
→ attention → transformer → training → generation → optimization. Each chapter's
runnable code lands in `examples/` and `tests/`.

## Development

The floor before any change is considered done:

```bash
pixi run fmt            # format (rewrites in place)
pixi run test           # smoke test, then the whole suite
```

CI runs `pixi run fmt-check` (fails on a formatting diff) and `pixi run test` on
every push and pull request — see [.github/workflows/ci.yml](.github/workflows/ci.yml).

## References

- [Mojo manual](https://mojolang.org/docs/manual/)

## License

TBD.
