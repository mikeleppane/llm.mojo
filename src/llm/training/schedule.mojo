"""Learning-rate schedule: linear warmup then cosine decay.

Raise the learning rate linearly from 0 to the peak over `warmup_steps` (a cold
start on a randomly-initialized model would otherwise take a destructive first
step), then decay it along a cosine from the peak down to `min_lr` at
`max_steps`, and hold `min_lr` after. Pure arithmetic on the step index, so it is
hand-computable at any step:

    step in [0, warmup_steps):   lr = peak * step / warmup_steps      (from 0)
    step == warmup_steps:        lr = peak                            (warmup end)
    step in (warmup, max_steps): lr = min_lr + (peak-min_lr)*cosine   (decaying)
    step >= max_steps:           lr = min_lr                          (clamped)

with cosine = 0.5*(1 + cos(pi * progress)) and progress = (step - warmup) /
(max_steps - warmup) in [0, 1). progress = 0 gives cosine = 1 (lr = peak, so the
warmup boundary is continuous) and progress -> 1 gives cosine -> 0 (lr -> min_lr).
"""

from std.math import cos, pi


def lr_at(
    step: Int,
    peak_lr: Float64,
    warmup_steps: Int,
    max_steps: Int,
    min_lr: Float64,
) raises -> Float64:
    """Compute the scheduled learning rate at `step` (0-based).

    See the module docstring for the piecewise definition. Allocates nothing.

    Args:
        step: The 0-based optimizer step index.
        peak_lr: The peak learning rate reached at the end of warmup.
        warmup_steps: Length of the linear warmup ramp.
        max_steps: Step at which the cosine reaches min_lr.
        min_lr: The cosine floor, held after max_steps.

    Returns:
        The learning rate for this step.

    Raises:
        Error: If step < 0, warmup_steps < 0, or max_steps <= warmup_steps (the
            cosine denominator max_steps - warmup_steps must be positive, and
            warmup must finish before the run ends).
    """
    if step < 0:
        raise Error("lr_at: step must be >= 0, got " + String(step))
    if warmup_steps < 0:
        raise Error(
            "lr_at: warmup_steps must be >= 0, got " + String(warmup_steps)
        )
    if max_steps <= warmup_steps:
        raise Error(
            "lr_at: max_steps ("
            + String(max_steps)
            + ") must exceed warmup_steps ("
            + String(warmup_steps)
            + ")"
        )

    # Linear warmup from 0. Guarded by warmup_steps > 0 here (step < warmup_steps
    # with warmup_steps == 0 is step < 0, impossible after the step >= 0 check),
    # so the division never hits zero.
    if step < warmup_steps:
        return peak_lr * Float64(step) / Float64(warmup_steps)

    # Past the end: hold the floor.
    if step >= max_steps:
        return min_lr

    # Cosine decay over [warmup_steps, max_steps]. At step == warmup_steps the
    # progress is 0 (cosine 1, lr = peak), so this branch meets the warmup ramp.
    var progress = Float64(step - warmup_steps) / Float64(
        max_steps - warmup_steps
    )
    var cosine = 0.5 * (1.0 + cos(pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


@fieldwise_init
struct ScheduleConfig(Copyable, Movable):
    """The two schedule knobs not already in TrainingConfig.

    Holds how long the linear warmup lasts and the floor the cosine decays to.
    The peak lr and the step budget live in TrainingConfig (learning_rate is the
    peak, max_steps the horizon), so validate() takes them to cross-check.
    """

    var warmup_steps: Int  # linear-warmup length; 0 disables warmup
    var min_lr: Float64  # the cosine floor, held after max_steps

    @staticmethod
    def gpt2_defaults() -> ScheduleConfig:
        """Build a conventional warmup + small floor preset.

        warmup_steps is a placeholder for tiny runs; real runs set it from the
        step budget (a common choice is a few percent of max_steps). min_lr = 0
        lets the cosine reach zero and is the neutral default a caller overrides.

        Returns:
            A ScheduleConfig with warmup 0 and floor 0.0.
        """
        return ScheduleConfig(0, 0.0)

    def validate(self, max_steps: Int, peak_lr: Float64) raises:
        """Validate the fields against the run's max_steps and peak lr.

        Args:
            max_steps: The run's step budget.
            peak_lr: The run's peak learning rate.

        Raises:
            Error: On the first invalid field, naming it: warmup must be
                non-negative and finish before the run ends, and the floor must
                sit in [0, peak].
        """
        if self.warmup_steps < 0:
            raise Error("ScheduleConfig: warmup_steps must be >= 0")
        if self.warmup_steps >= max_steps:
            raise Error(
                "ScheduleConfig: warmup_steps ("
                + String(self.warmup_steps)
                + ") must be < max_steps ("
                + String(max_steps)
                + ")"
            )
        if self.min_lr < 0.0:
            raise Error("ScheduleConfig: min_lr must be >= 0")
        if self.min_lr > peak_lr:
            raise Error(
                "ScheduleConfig: min_lr ("
                + String(self.min_lr)
                + ") must be <= peak lr ("
                + String(peak_lr)
                + ")"
            )
