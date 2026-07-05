# Train a small GPT on character-level tiny Shakespeare — the first real
# training run of the model this project has been building.
#
# It puts the whole Part XIV surface to work end to end: a char tokenizer and
# batch loader over a train/val split, a mini-GPT, and train_gpt (AdamW with a
# warmup+cosine schedule and global-norm gradient clipping) run in segments so it
# can checkpoint periodically and log train/val loss and perplexity. It interrupts
# the run at a checkpoint, then demonstrates load-and-resume: a freshly built
# model loads that checkpoint (parameters, optimizer moments, step counter, rng
# state) and FINISHES the run under the same schedule — no lr discontinuity, a
# genuine continuation rather than a fresh run.
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

# Run knobs. The schedule horizon is the FULL run TOTAL_STEPS; we interrupt at
# CHECKPOINT_STOP to demonstrate load-and-resume, so both phases share one cosine
# (the resume continues the same schedule, no lr jump). CHECKPOINT_STOP is also
# where the last periodic checkpoint lands.
comptime BATCH = 12
comptime PEAK_LR = 3e-3
comptime MIN_LR = 3e-4
comptime WARMUP = 30
comptime TOTAL_STEPS = 330  # the full run and the schedule horizon
comptime CHECKPOINT_STOP = 300  # interrupt here; resume finishes to TOTAL_STEPS
comptime EVAL_INTERVAL = 50
comptime EVAL_BATCHES = 6
comptime CHECKPOINT_INTERVAL = 100  # save every this many steps
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

    # The schedule horizon is the FULL run (TOTAL_STEPS), so the cosine is one
    # continuous curve across both the pre-checkpoint and the resumed phase.
    var tc = TrainingConfig(BATCH, PEAK_LR, TOTAL_STEPS, SEED)
    var oc = AdamWConfig.gpt2_defaults()
    var sc = ScheduleConfig(WARMUP, MIN_LR)
    var dropout_rng = Rng(SEED + DROPOUT_STREAM_OFFSET)

    print(
        "\ninitial loss (a uniform model scores loss log V; perplexity V =",
        vocab_size,
        "):",
    )
    _report_perplexity("train", gpt, train_loader)
    _report_perplexity("val  ", gpt, val_loader)

    # --- Train up to the checkpoint, saving periodically ------------------
    # Each segment runs under the SAME schedule horizon (tc.max_steps =
    # TOTAL_STEPS), carrying the AdamW moments across via init_m/init_v, so the
    # segmented run is identical to one uninterrupted run — just with checkpoints
    # in the gaps. We stop at CHECKPOINT_STOP to hand off to the resume demo.
    print(
        "\ntraining to step", CHECKPOINT_STOP, "(warmup", WARMUP, "-> cosine):"
    )
    var m = List[Tensor2D]()  # empty -> train_gpt starts fresh zeros
    var v = List[Tensor2D]()
    var done = 0
    while done < CHECKPOINT_STOP:
        var seg_end = done + CHECKPOINT_INTERVAL
        if seg_end > CHECKPOINT_STOP:
            seg_end = CHECKPOINT_STOP
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
            # report.lrs is indexed from this segment's start (`done`), so the lr
            # for eval at absolute step S is lrs[S - done].
            var lr_idx = report.eval_steps[e] - done
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
                report.lrs[lr_idx],
            )
        m = report.m.copy()
        v = report.v.copy()
        done = seg_end
        save_checkpoint(CHECKPOINT_PATH, gpt, m, v, done, dropout_rng.state)
        print("  [checkpoint saved at step", done, "->", CHECKPOINT_PATH, "]")

    print("\nloss at the checkpoint (step", CHECKPOINT_STOP, "):")
    _report_perplexity("train", gpt, train_loader)
    _report_perplexity("val  ", gpt, val_loader)

    # --- Load-and-resume demo --------------------------------------------
    # A fresh model (deliberately different init) loads the checkpoint and
    # finishes the run, restoring parameters, optimizer moments, step counter, and
    # the dropout rng state — so training picks up exactly where it left off, under
    # the same schedule (no lr discontinuity).
    print("\n--- load-and-resume demo ---")
    var resume_init = Rng(SEED + 999)  # a different init, fully overwritten
    var resumed = GPT.init_random(cfg, resume_init)
    var state = load_checkpoint(CHECKPOINT_PATH, resumed)
    print("restored from checkpoint at step", state.t)
    print("resumed model matches the saved model (identical eval loss):")
    _report_perplexity("val  ", resumed, val_loader)

    var resume_rng = Rng(0)
    resume_rng.state = state.rng_state  # continue the exact dropout stream
    var report2 = train_gpt(
        resumed,
        train_loader,
        val_loader,
        tc,  # same horizon -> the schedule continues seamlessly
        oc,
        sc,
        resume_rng,
        EVAL_INTERVAL,
        EVAL_BATCHES,
        state.t,
        state.m.copy(),
        state.v.copy(),
        TOTAL_STEPS,
    )
    print(
        "resumed to step",
        TOTAL_STEPS,
        "(",
        TOTAL_STEPS - CHECKPOINT_STOP,
        "more steps):",
    )
    _report_perplexity("train", resumed, train_loader)
    _report_perplexity("val  ", resumed, val_loader)
    _ = report2

    print(
        "\nDone. The saved checkpoint",
        CHECKPOINT_PATH,
        "is the artifact Part XV will make speak — generation comes next.",
    )
