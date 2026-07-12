"""Generate coherent English from a from-scratch Mojo forward over real GPT-2 124M.

Loads the converted weights and the GPT-2 tokenizer, encodes a classic prompt,
and calls generate() once greedily (argmax, deterministic) and once with nucleus
sampling (top-p 0.9, temperature 1.0).

This model is coherent, not fast: there is no KV cache, so every new token
re-runs the full forward over the entire growing context (~10^10 float64 flops
per token in a naive scalar matmul), which is minutes per token on CPU. If the
wall-clock is too long, shrink MAX_NEW_TOKENS, never the model.

Run (after downloading the weights and running the converter, see
scripts/convert_gpt2_weights.py):
    pixi run mojo run -I build examples/gpt2_generate.mojo
"""

from std.time import perf_counter_ns

from llm.generation.generate import generate
from llm.generation.sampler import SamplerConfig
from llm.tokenizer.gpt2 import GPT2Tokenizer, END_OF_TEXT_ID
from llm.transformer.gpt2_weights import load_gpt2
from llm.utils.random import Rng

comptime WEIGHTS_PATH = "checkpoints/gpt2-124m.bin"
comptime VOCAB_PATH = "data/gpt2/vocab.json"
comptime MERGES_PATH = "data/gpt2/merges.txt"
comptime PROMPT = "Hello, I'm a language model,"
comptime MAX_NEW_TOKENS = 25  # modest budget — every token is a full forward
comptime SEED = 1337


def _require_file(path: String) raises:
    """Raise a converter-pointing error unless `path` exists.

    Cheap existence probe (open, do not read the 498 MB payload); there is no
    fallback to random weights.

    Args:
        path: File that must exist.

    Raises:
        Error: If the file cannot be opened.
    """
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
    """Generate greedy and nucleus continuations, printing per-token timings."""
    _require_file(WEIGHTS_PATH)
    _require_file(VOCAB_PATH)
    _require_file(MERGES_PATH)

    print("=" * 70)
    print("GPT-2 124M — real weights, from-scratch Mojo forward pass")
    print("NO KV cache: every token re-runs the full forward (~minutes/token).")
    print("This model is COHERENT, not FAST — the speed work comes next.")
    print("=" * 70)

    print("loading weights (~2 GB resident: f64 values + zero grads)...")
    var gpt = load_gpt2(WEIGHTS_PATH)
    var tok = GPT2Tokenizer.from_files(String(VOCAB_PATH), String(MERGES_PATH))
    var prompt = tok.encode(String(PROMPT))
    print('prompt: "' + PROMPT + '"  (' + String(len(prompt)) + " tokens)")
    print()

    var stop = List[Int]()
    stop.append(END_OF_TEXT_ID)

    # --- greedy (deterministic argmax; draws no rng) --------------------------
    print("--- greedy (temperature 0) ---")
    var t0 = perf_counter_ns()
    var rng_greedy = Rng(SEED)
    var greedy = generate(
        gpt, prompt, MAX_NEW_TOKENS, SamplerConfig.greedy(), stop, rng_greedy
    )
    var t1 = perf_counter_ns()
    var full_greedy = prompt.copy()
    for i in range(len(greedy)):
        full_greedy.append(greedy[i])
    print(tok.decode(full_greedy))
    var greedy_ns = t1 - t0
    print(
        "[",
        len(greedy),
        "tokens in",
        Float64(greedy_ns) / 1.0e9,
        "s ->",
        Float64(greedy_ns) / 1.0e9 / Float64(len(greedy)),
        "s/token ]",
    )
    print()

    # --- nucleus sampling (top-p 0.9, temperature 1.0) ------------------------
    print("--- nucleus (top-p 0.9, temperature 1.0) ---")
    var sampler = SamplerConfig(1.0, 0, 0.9)
    var t2 = perf_counter_ns()
    var rng_sample = Rng(SEED)
    var sampled = generate(
        gpt, prompt, MAX_NEW_TOKENS, sampler, stop, rng_sample
    )
    var t3 = perf_counter_ns()
    var full_sampled = prompt.copy()
    for i in range(len(sampled)):
        full_sampled.append(sampled[i])
    print(tok.decode(full_sampled))
    var sampled_ns = t3 - t2
    print(
        "[",
        len(sampled),
        "tokens in",
        Float64(sampled_ns) / 1.0e9,
        "s ->",
        Float64(sampled_ns) / 1.0e9 / Float64(len(sampled)),
        "s/token ]",
    )
