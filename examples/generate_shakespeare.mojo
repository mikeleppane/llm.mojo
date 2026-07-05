# Make the trained Shakespeare checkpoint SPEAK — the payoff of the generation
# chapter, and the first time this project's model produces text end to end.
#
# It loads the checkpoint that examples/train_gpt_shakespeare.mojo saved, rebuilds
# the same character tokenizer deterministically from the corpus, and prints the
# continuation of a single one-newline prompt ("start of a line of dialogue")
# under FOUR decoding policies — greedy, temperature 0.8, top-k 40, top-p 0.9 —
# so the "safe vs surprising" trade-off is visible in one screen. Seeded, so the
# text is reproducible.
#
# Run (needs the Part XIV checkpoint on disk):
#     pixi run mojo run -I src examples/train_gpt_shakespeare.mojo   # writes it
#     pixi run mojo run -I src examples/generate_shakespeare.mojo    # this file
#
# The model shape constants below are DUPLICATED from train_gpt_shakespeare.mojo
# on purpose — examples don't import one another, and the checkpoint header's
# shape validation turns any drift between the two into a named load error rather
# than silent garbage. If you retune the trainer, retune here too.

from llm.config import GPTConfig
from llm.data.corpus import load_text
from llm.generation.generate import generate
from llm.generation.sampler import SamplerConfig
from llm.tokenizer.char import CharTokenizer
from llm.training.checkpoint import load_checkpoint
from llm.transformer.gpt import GPT
from llm.utils.random import Rng

# Model shape — MUST match examples/train_gpt_shakespeare.mojo (see file header).
comptime D_MODEL = 96
comptime N_LAYERS = 3
comptime N_HEADS = 4
comptime CONTEXT = 48
# Dropout is irrelevant to generation (the inference forward never applies it) and
# is not part of the checkpoint's shape validation; 0.0 states the intent.
comptime DROPOUT = 0.0

comptime CHECKPOINT_PATH = String("checkpoints/gpt_shakespeare.ckpt")
comptime CORPUS_PATH = String("data/tinyshakespeare/input.txt")

comptime NEW_TOKENS = 200  # one screen of generated characters (uncached, so
# every step recomputes the full forward — the recompute cost is the KV cache's
# motivation; a larger count is honest but slow on CPU)
comptime SEED: UInt64 = 20260705  # fixed, so the quoted output is reproducible


def _speak(
    label: String,
    gpt: GPT,
    tokenizer: CharTokenizer,
    prompt: List[Int],
    cfg: SamplerConfig,
) raises:
    # Generate NEW_TOKENS from `prompt` under `cfg` and print the decoded text
    # (prompt + continuation). Each policy gets its OWN fresh generator seeded the
    # same way, so the four samples are directly comparable and reproducible.
    var rng = Rng(SEED)
    var generated = generate(gpt, prompt, NEW_TOKENS, cfg, List[Int](), rng)
    var full = prompt.copy()
    for i in range(len(generated)):
        full.append(generated[i])
    print("\n=== ", label, " ===")
    print(tokenizer.decode(full))


def main() raises:
    # --- Rebuild the tokenizer and the model shape ------------------------
    var text = load_text(CORPUS_PATH)
    var tokenizer = CharTokenizer.from_text(
        text
    )  # deterministic from the corpus
    var vocab_size = tokenizer.vocab_size()

    var cfg = GPTConfig(
        vocab_size, CONTEXT, D_MODEL, N_LAYERS, N_HEADS, DROPOUT
    )
    # A throwaway init: every parameter is overwritten by load_checkpoint below.
    var init_rng = Rng(0)
    var gpt = GPT.init_random(cfg, init_rng)

    # --- Load the trained parameters --------------------------------------
    # load_checkpoint also returns the optimizer state (m, v, step, rng state);
    # generation needs PARAMETERS ONLY, so that state is deliberately discarded.
    # A missing file becomes a pointing error — never a silent fall back to the
    # random init, which would present gibberish as if it were the trained model.
    try:
        _ = load_checkpoint(CHECKPOINT_PATH, gpt)
    except e:
        raise Error(
            "generate_shakespeare: could not load "
            + CHECKPOINT_PATH
            + " ("
            + String(e)
            + "). Run examples/train_gpt_shakespeare.mojo first to train and"
            " save it."
        )

    print(
        "loaded",
        CHECKPOINT_PATH,
        "| vocab",
        vocab_size,
        "| model",
        cfg,
    )

    # --- Speak, four ways -------------------------------------------------
    # One prompt: a single newline, the start of a line of dialogue. The four
    # policies trace the safe -> surprising axis: greedy is deterministic and
    # repetitive; temperature 0.8 loosens it; top-k 40 and top-p 0.9 sample from a
    # truncated tail, the usual sweet spot between coherence and variety.
    var prompt = tokenizer.encode("\n")

    _speak("greedy (argmax)", gpt, tokenizer, prompt, SamplerConfig.greedy())
    _speak(
        "temperature 0.8", gpt, tokenizer, prompt, SamplerConfig(0.8, 0, 1.0)
    )
    _speak(
        "top-k 40 (T=1.0)", gpt, tokenizer, prompt, SamplerConfig(1.0, 40, 1.0)
    )
    _speak(
        "top-p 0.9 (T=1.0)", gpt, tokenizer, prompt, SamplerConfig(1.0, 0, 0.9)
    )
