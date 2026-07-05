# Train a small GPT on character-level tiny Shakespeare — the first real
# training run of the model this project has been building.
#
# It puts the whole Part XIV surface to work end to end: a char tokenizer and
# batch loader over a train/val split, a mini-GPT, and train_gpt (AdamW with a
# warmup+cosine schedule and global-norm gradient clipping) run in segments so it
# can checkpoint periodically and log train/val loss and perplexity. It saves a
# final checkpoint, then demonstrates load-and-resume: a freshly built model
# loads the checkpoint (parameters, optimizer moments, step counter, rng state)
# and continues training with no discontinuity.
#
# Run (needs a checkpoints/ directory, which is gitignored):
#     mkdir -p checkpoints
#     pixi run mojo run -I src examples/train_gpt_shakespeare.mojo
#
# There is deliberately NO text generation here — sampling from a trained model
# is Part XV, and its first demo will be *this* checkpoint speaking. The arc of
# this chapter ends on the saved checkpoint file.

from llm.config import GPTConfig, TrainingConfig
from llm.data.corpus import load_text
from llm.data.dataset import train_val_split
from llm.data.loader import BatchLoader
from llm.tensor.tensor2d import Tensor2D
from llm.tokenizer.char import CharTokenizer
from llm.training.checkpoint import load_checkpoint, save_checkpoint
from llm.training.gpt_trainer import estimate_loss, train_gpt
from llm.training.loss import perplexity
from llm.training.optimizer import AdamWConfig
from llm.training.schedule import ScheduleConfig
from llm.transformer.gpt import GPT
from llm.utils.random import Rng

# Model shape — a mini-GPT (GPT-2's layout at toy scale).
comptime D_MODEL = 96
comptime N_LAYERS = 3
comptime N_HEADS = 4
comptime CONTEXT = 48  # T: sequence length the model trains on
comptime DROPOUT = 0.1

# Run knobs.
comptime BATCH = 12
comptime PEAK_LR = 3e-3
comptime MIN_LR = 3e-4
comptime WARMUP = 30
comptime MAX_STEPS = 300
comptime EVAL_INTERVAL = 50
comptime EVAL_BATCHES = 6
comptime CHECKPOINT_INTERVAL = 100  # save every this many steps
comptime RESUME_STEPS = 30  # extra steps the resume demo trains
comptime SEED: UInt64 = 1337

# The dropout rng is a stream distinct from the loader's `seed + epoch` family so
# the two never coincide; offsetting the seed by a fixed constant separates them.
comptime DROPOUT_STREAM_OFFSET: UInt64 = 0x9E3779B97F4A7C15

comptime CHECKPOINT_PATH = String("checkpoints/gpt_shakespeare.ckpt")


def _report_perplexity(label: String, gpt: GPT, mut loader: BatchLoader) raises:
    var loss = estimate_loss(gpt, loader, EVAL_BATCHES)
    print(
        "  ",
        label,
        "loss",
        loss,
        " perplexity",
        perplexity(loss),
    )


def main() raises:
    # --- Data -------------------------------------------------------------
    var text = load_text("data/tinyshakespeare/input.txt")
    var tokenizer = CharTokenizer.from_text(text)
    var vocab_size = tokenizer.vocab_size()
    var ids = tokenizer.encode(text)
    var split = train_val_split(ids, 0.1)
    print(
        "corpus:",
        text.byte_length(),
        "bytes, vocab:",
        vocab_size,
        "chars, train tokens:",
        split.train.size(),
        "val tokens:",
        split.val.size(),
    )

    var train_loader = BatchLoader(
        split.train.copy(), BATCH, seq_len=CONTEXT, stride=CONTEXT
    )
    var val_loader = BatchLoader(
        split.val.copy(), BATCH, seq_len=CONTEXT, stride=CONTEXT
    )

    # --- Model and run configs -------------------------------------------
    var cfg = GPTConfig(
        vocab_size, CONTEXT, D_MODEL, N_LAYERS, N_HEADS, DROPOUT
    )
    var init_rng = Rng(SEED)
    var gpt = GPT.init_random(cfg, init_rng)
    print(
        "model:",
        cfg,
        "parameters:",
        gpt.parameter_count_actual(),
    )

    var tc = TrainingConfig(BATCH, PEAK_LR, MAX_STEPS, SEED)
    var oc = AdamWConfig.gpt2_defaults()
    var sc = ScheduleConfig(WARMUP, MIN_LR)
    var dropout_rng = Rng(SEED + DROPOUT_STREAM_OFFSET)

    print(
        "\ninitial loss (a uniform model would be ~", perplexity(0.0), "x V):"
    )
    _report_perplexity("train", gpt, train_loader)
    _report_perplexity("val  ", gpt, val_loader)

    # --- Train in segments, checkpointing between -------------------------
    # Each segment runs under the SAME schedule horizon (tc.max_steps), carrying
    # the AdamW moments across via init_m/init_v, so the segmented run is
    # identical to one uninterrupted run — just with checkpoints in the gaps.
    print("\ntraining", MAX_STEPS, "steps (warmup", WARMUP, "-> cosine):")
    var m = List[Tensor2D]()  # empty -> train_gpt starts fresh zeros
    var v = List[Tensor2D]()
    var done = 0
    while done < MAX_STEPS:
        var seg_end = done + CHECKPOINT_INTERVAL
        if seg_end > MAX_STEPS:
            seg_end = MAX_STEPS
        var report = train_gpt(
            gpt,
            train_loader,
            val_loader,
            tc,
            oc,
            sc,
            dropout_rng,
            EVAL_INTERVAL,
            EVAL_BATCHES,
            done,
            m,
            v,
            seg_end,
        )
        for e in range(len(report.eval_steps)):
            print(
                "  step",
                report.eval_steps[e] + 1,
                "| train",
                report.eval_train_losses[e],
                "| val",
                report.eval_val_losses[e],
                "| val ppl",
                perplexity(report.eval_val_losses[e]),
                "| lr",
                report.lrs[len(report.lrs) - 1],
            )
        m = report.m.copy()
        v = report.v.copy()
        done = seg_end
        save_checkpoint(CHECKPOINT_PATH, gpt, m, v, done, dropout_rng.state)
        print("  [checkpoint saved at step", done, "->", CHECKPOINT_PATH, "]")

    print("\nfinal loss after", MAX_STEPS, "steps:")
    _report_perplexity("train", gpt, train_loader)
    _report_perplexity("val  ", gpt, val_loader)

    # --- Load-and-resume demo --------------------------------------------
    # A fresh model (deliberately different init) loads the checkpoint and
    # continues, restoring parameters, optimizer moments, step counter, and the
    # dropout rng state — so training picks up exactly where it left off.
    print("\n--- load-and-resume demo ---")
    var resume_init = Rng(SEED + 999)  # a different init, fully overwritten
    var resumed = GPT.init_random(cfg, resume_init)
    var state = load_checkpoint(CHECKPOINT_PATH, resumed)
    print("restored from checkpoint at step", state.t)
    print("resumed model matches the saved model (identical eval loss):")
    _report_perplexity("val  ", resumed, val_loader)

    var tc2 = TrainingConfig(BATCH, PEAK_LR, MAX_STEPS + RESUME_STEPS, SEED)
    var resume_rng = Rng(0)
    resume_rng.state = state.rng_state  # continue the exact dropout stream
    var report2 = train_gpt(
        resumed,
        train_loader,
        val_loader,
        tc2,
        oc,
        sc,
        resume_rng,
        EVAL_INTERVAL,
        EVAL_BATCHES,
        state.t,
        state.m.copy(),
        state.v.copy(),
        MAX_STEPS + RESUME_STEPS,
    )
    print("trained", RESUME_STEPS, "more steps from the checkpoint:")
    _report_perplexity("train", resumed, train_loader)
    _report_perplexity("val  ", resumed, val_loader)
    _ = report2

    print(
        "\nDone. The saved checkpoint",
        CHECKPOINT_PATH,
        "is the artifact Part XV will make speak — generation comes next.",
    )
