# The GPT training loop — AdamW + warmup/cosine schedule + global-norm clipping
# over Part VI's BatchLoader, with periodic dropout-free evaluation.
#
# This is where all of Part XIV's pieces meet: each step fetches a batch, runs
# the cached (dropout) forward and the loss backward per sequence with the
# gradient accumulated across the batch, clips the whole-model gradient norm,
# looks up the scheduled learning rate, and takes one AdamW step. The optimizer
# state (the m/v moment lists) is trainer-owned here — allocated once from the
# model's parameter_shapes and threaded through every step. overfit-one-batch
# runs through THIS loop unchanged (a one-batch loader), so the capstone tests
# the real trainer, not a special case.
#
# Seed derivation (so one seed reproduces a run end to end): the training loader
# is reshuffled at the top of each epoch with `tc.seed + epoch` (Part VI's
# documented per-epoch convention), and the dropout stream is the caller-supplied
# `rng` — a SEPARATE generator the caller seeds distinctly from tc.seed (the
# example offsets it) so the two streams do not coincide. With dropout 0 the rng
# is never drawn and the run is fully determined by tc.seed alone.

from llm.config import TrainingConfig
from llm.data.loader import BatchLoader
from llm.tensor.ops import cross_entropy_rows, cross_entropy_rows_backward
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.training.optimizer import AdamWConfig, clip_grad_norm
from llm.training.schedule import ScheduleConfig, lr_at
from llm.transformer.gpt import GPT
from llm.utils.random import Rng


@fieldwise_init
struct TrainReport(Copyable, Movable):
    # The histories a run produces, for tests and the example to assert on and
    # plot, plus the FINAL optimizer state so a caller can checkpoint a resumable
    # run. The per-step lists have one entry per optimizer step run this call; the
    # eval lists have one entry per evaluation point.
    var train_losses: List[Float64]  # batch loss at each step
    var grad_norms: List[Float64]  # pre-clip global grad norm at each step
    var lrs: List[Float64]  # scheduled lr at each step
    var eval_steps: List[Int]  # step indices where evaluation ran
    var eval_train_losses: List[
        Float64
    ]  # dropout-free train loss at eval points
    var eval_val_losses: List[Float64]  # dropout-free val loss at eval points
    var m: List[Tensor2D]  # final AdamW first moments (walk order)
    var v: List[Tensor2D]  # final AdamW second moments (walk order)


def _row(flat: List[Int], b: Int, t: Int) -> List[Int]:
    # Extract row b (length t) from a flat row-major [B, T] array. Allocates the
    # row; cannot raise (caller guarantees the bounds).
    var out = List[Int]()
    for i in range(t):
        out.append(flat[b * t + i])
    return out^


def estimate_loss(
    gpt: GPT, mut loader: BatchLoader, num_batches: Int
) raises -> Float64:
    # Mean dropout-free cross-entropy over the first `num_batches` batches of the
    # loader's current epoch order — the plain `forward` (no dropout, no rng), so
    # this is the model's true predictive loss, not the training loss. Evaluates
    # from the START of the current order and RESTORES the loader's cursor, so it
    # never disturbs a training loop that shares the loader. Reads gpt; advances
    # then rewinds the loader cursor; allocates scratch; raises if there are no
    # batches to average (an empty loader).
    var saved_cursor = loader.cursor
    loader.cursor = 0
    var nb = num_batches
    if nb > loader.num_batches():
        nb = loader.num_batches()
    var total = 0.0
    var count = 0
    for _ in range(nb):
        var batch = loader.next_batch()
        for b in range(batch.batch_size):
            var ids = _row(batch.inputs, b, batch.seq_len)
            var targets = _row(batch.targets, b, batch.seq_len)
            var logits = gpt.forward(ids)  # dropout-free inference path
            total += cross_entropy_rows(logits, targets)
            count += 1
    loader.cursor = saved_cursor
    if count == 0:
        raise Error("estimate_loss: loader yielded no batches to average")
    return total / Float64(count)


def train_gpt(
    mut gpt: GPT,
    mut train_loader: BatchLoader,
    mut val_loader: BatchLoader,
    tc: TrainingConfig,
    oc: AdamWConfig,
    sc: ScheduleConfig,
    mut rng: Rng,
    eval_interval: Int,
    eval_batches: Int,
    start_step: Int = 0,
    init_m: List[Tensor2D] = List[Tensor2D](),
    init_v: List[Tensor2D] = List[Tensor2D](),
    end_step: Int = -1,
) raises -> TrainReport:
    # Train `gpt` for steps [start_step, tc.max_steps) over `train_loader`,
    # evaluating on both loaders every `eval_interval` steps (and on the final
    # step). Returns the loss/lr/grad-norm histories AND the final AdamW moments
    # (so a caller can checkpoint a resumable run).
    #
    # Per step: reshuffle the loader at each epoch boundary (tc.seed + epoch) ->
    # zero_grad -> for each of the B sequences, cached (dropout) forward, the
    # cross-entropy backward scaled by 1/B, and backward (grads accumulate across
    # the batch) -> clip the global grad norm to oc.grad_clip -> look up
    # lr_at(step, ...) -> one AdamW step (t = step + 1, so bias correction starts
    # at 1).
    #
    # Resuming: start_step > 0 continues the schedule and the AdamW step counter
    # from where a prior run stopped; pass init_m/init_v (from a checkpoint or a
    # prior TrainReport) to carry the optimizer moments across the interruption.
    # Empty init_m/init_v (the default) start fresh zeros. The train loader's
    # epoch and within-epoch position are reconstructed from start_step, so a
    # fixed corpus resumes at the same batch it would have reached uninterrupted.
    # `end_step` (default tc.max_steps) stops the loop early WITHOUT changing the
    # schedule horizon — so training in segments [0, k), [k, n) with one save
    # between reproduces the single [0, n) run exactly (the lr at each step still
    # references tc.max_steps).
    #
    # Mutates gpt (parameters and grads), both loaders (cursor/order), and rng
    # (dropout draws when cfg.dropout > 0); allocates the state and the report;
    # raises on an invalid config, a mis-sized init state, or a loader that cannot
    # yield a batch.
    tc.validate()
    oc.validate()
    sc.validate(tc.max_steps, tc.learning_rate)
    if eval_interval <= 0:
        raise Error("train_gpt: eval_interval must be positive")
    if eval_batches <= 0:
        raise Error("train_gpt: eval_batches must be positive")
    if start_step < 0 or start_step >= tc.max_steps:
        raise Error("train_gpt: start_step must be in [0, max_steps)")
    var last_step = tc.max_steps if end_step < 0 else end_step
    if last_step <= start_step or last_step > tc.max_steps:
        raise Error("train_gpt: end_step must be in (start_step, max_steps]")

    # Trainer-owned AdamW state: continue from init_m/init_v if given, else fresh
    # zeros. A non-empty init state must have one tensor per parameter.
    var shapes = gpt.parameter_shapes()
    var m = List[Tensor2D]()
    var v = List[Tensor2D]()
    if len(init_m) == 0 and len(init_v) == 0:
        for k in range(len(shapes)):
            m.append(zeros_2d(shapes[k].rows, shapes[k].cols))
            v.append(zeros_2d(shapes[k].rows, shapes[k].cols))
    elif len(init_m) == len(shapes) and len(init_v) == len(shapes):
        m = init_m.copy()
        v = init_v.copy()
    else:
        raise Error(
            "train_gpt: init_m/init_v must be empty or have one tensor per"
            " parameter"
        )

    var train_losses = List[Float64]()
    var grad_norms = List[Float64]()
    var lrs = List[Float64]()
    var eval_steps = List[Int]()
    var eval_train_losses = List[Float64]()
    var eval_val_losses = List[Float64]()

    # Reconstruct the loader position from the global step: each epoch yields
    # num_batches() batches, so the epoch is start_step // num_batches and the
    # within-epoch offset is the remainder (start_step = 0 gives epoch 0, cursor
    # 0 — the fresh-run path).
    var per_epoch = train_loader.num_batches()
    var epoch = start_step // per_epoch
    train_loader.start_epoch(tc.seed + UInt64(epoch))
    train_loader.cursor = (start_step % per_epoch) * train_loader.batch_size

    for step in range(start_step, last_step):
        if not train_loader.has_next():
            epoch += 1
            train_loader.start_epoch(tc.seed + UInt64(epoch))
        var batch = train_loader.next_batch()
        var inv_b = 1.0 / Float64(batch.batch_size)

        gpt.zero_grad()
        var batch_loss = 0.0
        for b in range(batch.batch_size):
            var ids = _row(batch.inputs, b, batch.seq_len)
            var targets = _row(batch.targets, b, batch.seq_len)
            var fwd = gpt.forward_cached(ids, True, rng)
            batch_loss += cross_entropy_rows(fwd.logits, targets) * inv_b
            # Scale the per-sequence gradient by 1/B so the accumulated batch
            # gradient is the MEAN over sequences (matching the reported loss).
            var d_logits = cross_entropy_rows_backward(fwd.logits, targets)
            for i in range(d_logits.rows):
                for j in range(d_logits.cols):
                    d_logits[i, j] = d_logits[i, j] * inv_b
            gpt.backward(fwd.cache, d_logits)

        var norm = clip_grad_norm(gpt, oc.grad_clip)
        var lr = lr_at(
            step, tc.learning_rate, sc.warmup_steps, tc.max_steps, sc.min_lr
        )
        gpt.apply_adamw(
            m, v, step + 1, lr, oc.beta1, oc.beta2, oc.eps, oc.weight_decay
        )

        train_losses.append(batch_loss)
        grad_norms.append(norm)
        lrs.append(lr)

        if (step + 1) % eval_interval == 0 or step == last_step - 1:
            eval_steps.append(step)
            eval_train_losses.append(
                estimate_loss(gpt, train_loader, eval_batches)
            )
            eval_val_losses.append(estimate_loss(gpt, val_loader, eval_batches))

    return TrainReport(
        train_losses^,
        grad_norms^,
        lrs^,
        eval_steps^,
        eval_train_losses^,
        eval_val_losses^,
        m^,
        v^,
    )
