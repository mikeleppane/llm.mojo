# THE MONEY DEMO: the same real GPT-2 124M, now with a KV cache — seconds per
# token instead of minutes, and BIT-IDENTICAL text.
#
# Part XVI played the instrument honestly but slowly: every generated token
# re-ran the FULL forward over the whole growing context, ~minutes per token on
# CPU. That was the opening argument for this part. Here the KV cache answers it:
# each layer's keys and values are cached, and per step only the ONE new token
# flows through the network. Per-token cost drops from O(T·C²·L) — a full forward
# over T positions — to O(C²·L + T·C·L): one position's linear layers plus
# attention reads over the cache.
#
# The demo makes the before/after concrete on the SAME machine and weights:
#   1. Time ONE uncached token — a single full gpt.forward at the FINAL sequence
#      length (prompt + budget). The uncached path re-runs that whole forward per
#      token and it grows as the sequence grows, so the final-length forward is the
#      honest peak "before" number the cache is measured against.
#   2. Generate the continuation with the cache, printing the per-token wall-clock
#      and the measured speedup.
# The greedy run reproduces Part XVI's continuation CHARACTER-FOR-CHARACTER — the
# proof that caching changed the ALGORITHM, not the arithmetic, on all
# 124,439,808 real parameters. Then a nucleus run for variety.
#
# What is STILL not done: the arithmetic is scalar float64 — no SIMD, no
# threading, no blocked matmul. That is the next part's job. This part changed
# recompute-everything into cache-and-reuse; the part after changes how fast each
# remaining flop runs. The two compose.
#
# Run (after downloading the weights and running the converter — see
# scripts/convert_gpt2_weights.py):
#   pixi run mojo run -I build examples/gpt2_generate_cached.mojo

from std.time import perf_counter_ns

from llm.generation.generate import generate_cached
from llm.generation.sampler import SamplerConfig
from llm.tokenizer.gpt2 import GPT2Tokenizer, END_OF_TEXT_ID
from llm.transformer.gpt2_weights import load_gpt2
from llm.utils.random import Rng

comptime WEIGHTS_PATH = "checkpoints/gpt2-124m.bin"
comptime VOCAB_PATH = "data/gpt2/vocab.json"
comptime MERGES_PATH = "data/gpt2/merges.txt"
comptime PROMPT = "Hello, I'm a language model,"
comptime MAX_NEW_TOKENS = 25  # same budget as the uncached "before" example
comptime SEED = 1337


def _require_file(path: String) raises:
    # Cheap existence probe (open, do not read the 498 MB payload) with a message
    # pointing at the converter. NO fallback to random weights.
    try:
        var probe = open(path, "r")
        probe.close()
    except:
        raise Error(
            "missing '"
            + path
            + "' — download the HF gpt2 model.safetensors and run"
            " scripts/convert_gpt2_weights.py to produce it (see that script's"
            " header). No random-weight fallback: this demo needs the real"
            " weights."
        )


def main() raises:
    _require_file(WEIGHTS_PATH)
    _require_file(VOCAB_PATH)
    _require_file(MERGES_PATH)

    print("=" * 70)
    print(
        "GPT-2 124M — real weights, from-scratch Mojo forward pass, KV-CACHED"
    )
    print(
        "Per-token cost: O(T*C^2) full forward  ->  O(C^2) one pass + O(T*C)."
    )
    print("Same weights, same text as the uncached run — seconds, not minutes.")
    print("Still scalar float64: the ARITHMETIC speed-up comes next.")
    print("=" * 70)

    print("loading weights (~2 GB resident: f64 values + zero grads)...")
    var gpt = load_gpt2(WEIGHTS_PATH)
    var tok = GPT2Tokenizer.from_files(String(VOCAB_PATH), String(MERGES_PATH))
    var prompt = tok.encode(String(PROMPT))
    print('prompt: "' + PROMPT + '"  (' + String(len(prompt)) + " tokens)")
    print()

    # --- the honest "before": one uncached token at the FINAL context length ---
    # The uncached path re-runs the FULL forward for every token, and that forward
    # grows as the sequence grows — so its per-token cost peaks at the final length
    # (prompt + budget). Timing one full gpt.forward at that length is the honest
    # worst-case "before" the cache is measured against (the token values do not
    # affect the matmul sizes, so a cyclically-extended prompt suffices).
    var final_len = len(prompt) + MAX_NEW_TOKENS
    var timing_ctx = List[Int]()
    for i in range(final_len):
        timing_ctx.append(prompt[i % len(prompt)])
    print(
        "--- one uncached token (a full gpt.forward at the final length "
        + String(final_len)
        + ") ---"
    )
    var b0 = perf_counter_ns()
    var probe_logits = gpt.forward(timing_ctx)
    var b1 = perf_counter_ns()
    var uncached_ns = b1 - b0
    # Touch the result so the forward can't be optimized away.
    print("logits shape [", probe_logits.rows, ",", probe_logits.cols, "]")
    var uncached_s = Float64(uncached_ns) / 1.0e9
    print("[ one uncached token:", uncached_s, "s ]")
    print()

    var stop = List[Int]()
    stop.append(END_OF_TEXT_ID)

    # --- greedy, KV-cached (the 124M parity gate) -----------------------------
    print("--- greedy (temperature 0), KV-cached ---")
    var t0 = perf_counter_ns()
    var rng_greedy = Rng(SEED)
    var greedy = generate_cached(
        gpt, prompt, MAX_NEW_TOKENS, SamplerConfig.greedy(), stop, rng_greedy
    )
    var t1 = perf_counter_ns()
    var full_greedy = prompt.copy()
    for i in range(len(greedy)):
        full_greedy.append(greedy[i])
    print(tok.decode(full_greedy))
    var greedy_ns = t1 - t0
    var greedy_per_token = Float64(greedy_ns) / 1.0e9 / Float64(len(greedy))
    print(
        "[",
        len(greedy),
        "tokens in",
        Float64(greedy_ns) / 1.0e9,
        "s ->",
        greedy_per_token,
        "s/token ]",
    )
    print(
        "[ speedup vs one uncached token:", uncached_s / greedy_per_token, "x ]"
    )
    print()

    # --- nucleus sampling (top-p 0.9, temperature 1.0), KV-cached -------------
    print("--- nucleus (top-p 0.9, temperature 1.0), KV-cached ---")
    var sampler = SamplerConfig(1.0, 0, 0.9)
    var t2 = perf_counter_ns()
    var rng_sample = Rng(SEED)
    var sampled = generate_cached(
        gpt, prompt, MAX_NEW_TOKENS, sampler, stop, rng_sample
    )
    var t3 = perf_counter_ns()
    var full_sampled = prompt.copy()
    for i in range(len(sampled)):
        full_sampled.append(sampled[i])
    print(tok.decode(full_sampled))
    var sampled_ns = t3 - t2
    var sampled_per_token = Float64(sampled_ns) / 1.0e9 / Float64(len(sampled))
    print(
        "[",
        len(sampled),
        "tokens in",
        Float64(sampled_ns) / 1.0e9,
        "s ->",
        sampled_per_token,
        "s/token ]",
    )
    print(
        "[ speedup vs one uncached token:",
        uncached_s / sampled_per_token,
        "x ]",
    )
