"""The GPT training loop: AdamW + warmup/cosine schedule + global-norm clipping.

Runs over a BatchLoader with periodic dropout-free evaluation. Each step fetches
a batch, runs the cached (dropout) forward and the loss backward per sequence
with the gradient accumulated across the batch, clips the whole-model gradient
norm, looks up the scheduled learning rate, and takes one AdamW step. The
optimizer state (the m/v moment lists) is trainer-owned here: allocated once from
the model's parameter_shapes and threaded through every step. An overfit-one-batch
run goes through this loop unchanged (a one-batch loader).

Seed derivation, so one seed reproduces a run end to end: the training loader is
reshuffled at the top of each epoch with `tc.seed + epoch`, and the dropout
stream is the caller-supplied `rng`, a separate generator the caller seeds
distinctly from tc.seed so the two streams do not coincide. With dropout 0 the
rng is never drawn and the run is fully determined by tc.seed alone.
"""

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
    """The histories a run produces, plus its final optimizer state.

    The per-step lists have one entry per optimizer step run this call; the eval
    lists have one entry per evaluation point. The final m/v moments let a caller
    checkpoint a resumable run.
    """

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
    """Extract row b (length t) from a flat row-major [B, T] array.

    Args:
        flat: The flattened [B, T] array.
        b: The row index.
        t: The row length (T).

    Returns:
        The extracted row of length t. Allocates it; the caller guarantees the
        bounds, so this cannot raise.
    """
    var out = List[Int]()
    for i in range(t):
        out.append(flat[b * t + i])
    return out^


def estimate_loss(
    gpt: GPT, mut loader: BatchLoader, num_batches: Int
) raises -> Float64:
    """Compute the mean dropout-free cross-entropy over the first batches.

    Uses the plain `forward` (no dropout, no rng), so this is the model's true
    predictive loss, not the training loss. Evaluates from the start of the
    current epoch order and restores the loader's cursor, so it never disturbs a
    training loop that shares the loader.

    Args:
        gpt: The model to evaluate (read only).
        loader: The batch loader; its cursor is advanced then rewound.
        num_batches: How many batches to average (clamped to the loader's count).

    Returns:
        The mean cross-entropy. Allocates scratch.

    Raises:
        Error: If there are no batches to average (an empty loader).
    """
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
    """Train `gpt` for steps [start_step, last_step) over `train_loader`.

    Evaluates on both loaders every `eval_interval` steps and on the final step.
    Per step: reshuffle the loader at each epoch boundary (tc.seed + epoch),
    zero_grad, then for each of the B sequences do a cached (dropout) forward, the
    cross-entropy backward scaled by 1/B, and backward (grads accumulate across
    the batch); clip the global grad norm to oc.grad_clip; look up the scheduled
    lr; take one AdamW step (t = step + 1, so bias correction starts at 1).

    Resuming: start_step > 0 continues the schedule and the AdamW step counter
    from where a prior run stopped; pass init_m/init_v (from a checkpoint or a
    prior TrainReport) to carry the optimizer moments across the interruption.
    Empty init_m/init_v start fresh zeros. The loader's epoch and within-epoch
    position are reconstructed from start_step, so a fixed corpus resumes at the
    same batch it would have reached uninterrupted. `end_step` stops the loop
    early without changing the schedule horizon, so training in segments [0, k),
    [k, n) with one save between reproduces the single [0, n) run exactly (each
    step's lr still references tc.max_steps).

    Args:
        gpt: The model to train (parameters and grads mutated).
        train_loader: Training batch loader (cursor/order mutated).
        val_loader: Validation batch loader (cursor mutated by evaluation).
        tc: Training config (batch_size, peak learning_rate, max_steps, seed).
        oc: AdamW hyperparameters.
        sc: Warmup/cosine schedule config.
        rng: The dropout stream (drawn only when cfg.dropout > 0).
        eval_interval: Evaluate every this many steps; must be positive.
        eval_batches: How many batches each evaluation averages; must be positive.
        start_step: First step to run, in [0, max_steps); continues a prior run.
        init_m: Initial AdamW first moments, empty or one tensor per parameter.
        init_v: Initial AdamW second moments, empty or one tensor per parameter.
        end_step: Last step (exclusive); -1 means tc.max_steps.

    Returns:
        A TrainReport with the loss/lr/grad-norm histories and the final AdamW
        moments. Allocates the state and the report.

    Raises:
        Error: On an invalid config, an out-of-range start_step or end_step, a
            mis-sized init state, or a loader that cannot yield a batch.
    """
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
