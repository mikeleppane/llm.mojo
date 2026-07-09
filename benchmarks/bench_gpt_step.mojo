# GPT forward+backward step timing.
#
# One training step assembles a cache for every layer on the forward pass and
# consumes it on the backward pass. This harness times that whole round trip so
# a change to how the caches are assembled (copied vs moved) shows up as a step
# time, not a guess.
#
# This is outside the test gate — it measures, it does not assert. Run it
# (ideally a release build) and record your own numbers:
#     pixi run mojo precompile src/llm -o build/llm.mojopkg
#     pixi run mojo run -I build benchmarks/bench_gpt_step.mojo
#
# The timer uses perf_counter_ns; the median comes from the unit-tested helper
# in llm.utils.timing.
#
# Config: a scaled-down GPT-2 (same proportions — 4x MLP, C/H head width, tied
# head — small enough to run in seconds): V=256, T=64, C=128, L=6, H=4,
# dropout=0. dropout=0 keeps the run deterministic; the cached forward still
# builds every layer's cache exactly as training does, which is the assembly the
# step time reflects.

from std.time import perf_counter_ns
from std.collections import List

from llm.config import GPTConfig
from llm.tensor.ops import cross_entropy_rows_backward
from llm.transformer.gpt import GPT
from llm.utils.random import Rng
from llm.utils.timing import median_ns


def bench_gpt_step(steps: Int, warmup: Int) raises:
    var cfg = GPTConfig(256, 64, 128, 6, 4, 0.0)  # V, T, C, L, H, dropout
    var init_rng = Rng(0)
    var gpt = GPT.init_random(cfg, init_rng)

    var t = cfg.context_length
    var ids = List[Int]()
    var targets = List[Int]()
    for i in range(t):
        ids.append(i % cfg.vocab_size)
        targets.append((i + 1) % cfg.vocab_size)

    var drop_rng = Rng(1)
    for _ in range(warmup):
        var fwd = gpt.forward_cached(ids, True, drop_rng)
        var d_logits = cross_entropy_rows_backward(fwd.logits, targets)
        gpt.zero_grad()
        gpt.backward(fwd.cache, d_logits)

    var samples = List[Int]()
    for _ in range(steps):
        var start = perf_counter_ns()
        var fwd = gpt.forward_cached(ids, True, drop_rng)
        var d_logits = cross_entropy_rows_backward(fwd.logits, targets)
        gpt.zero_grad()
        gpt.backward(fwd.cache, d_logits)
        var end = perf_counter_ns()
        samples.append(Int(end - start))

    var med = median_ns(samples)
    print(
        "gpt fwd+bwd step  [V=256 T=64 C=128 L=6 H=4]  steps",
        steps,
        " median_ns",
        med,
        " median_ms",
        Float64(med) / 1.0e6,
    )


def main() raises:
    bench_gpt_step(steps=21, warmup=5)
