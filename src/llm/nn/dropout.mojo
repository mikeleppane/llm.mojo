"""Dropout — inverted dropout with train/eval as an argument, not stored state.

In training each element is kept with probability 1 - p and, if kept, scaled by
1/(1-p); dropped elements become 0. The scaling keeps each element's expected
value unchanged, so evaluation is a plain identity. Eval mode (and p == 0) draw
no random numbers, so disabling dropout never perturbs the seeded generator.
Stateless free functions, not a struct.
"""

from llm.tensor.tensor2d import Tensor2D, ones_2d, zeros_2d
from llm.utils.random import Rng


@fieldwise_init
struct DropoutResult(Copyable, Movable):
    """Output of dropout_cached plus what backward needs: the mask and scale.

    `mask[i, j]` is 1.0 for a kept element and 0.0 for a dropped one — the same
    mask the forward used. `inv_keep` is the scale the forward applied (1/(1-p)
    in training, 1.0 in eval / p == 0), cached so backward reproduces the exact
    forward map rather than recomputing it from p.
    """

    var output: Tensor2D  # [N, C]
    var mask: Tensor2D  # [N, C], entries in {0, 1}
    var inv_keep: Float64  # applied scale: 1/(1-p) training, 1.0 eval / p == 0


def dropout(
    x: Tensor2D, p: Float64, training: Bool, mut rng: Rng
) raises -> Tensor2D:
    """Apply inverted dropout.

    Args:
        x: Input, shape [N, C].
        p: Drop probability in [0, 1).
        training: If false (or p == 0), returns an identity copy without drawing.
        rng: Random generator, advanced one draw per element in the training branch.

    Returns:
        Output, shape [N, C]. Allocates; reads x only.

    Raises:
        Error: If p is outside [0, 1) (a NaN p raises too; p = 1 would divide by
            zero in the scale).
    """
    # "Not in range" rather than "p < 0 or p >= 1" so a NaN p raises too: every
    # comparison with NaN is false, so the negation fires.
    if not (p >= 0.0 and p < 1.0):
        raise Error("dropout: p must be in [0, 1), got " + String(p))
    # Eval mode, or p == 0, returns an identity copy WITHOUT drawing — disabling
    # dropout must not perturb the seeded generator. Since the guard above has
    # already excluded p < 0, `p <= 0.0` here means exactly p == 0, and avoids a
    # float `==` comparison.
    if not training or p <= 0.0:
        return x.copy()

    var keep_prob = 1.0 - p
    var inv_keep = 1.0 / keep_prob
    var out = zeros_2d(x.rows, x.cols)
    for r in range(x.rows):
        for c in range(x.cols):
            # Keep iff a fresh uniform in [0, 1) lands below keep_prob; the mask
            # is Bernoulli(keep_prob). Survivors scale by 1/keep_prob; dropped
            # elements keep the zero already in `out`. One draw per element.
            if rng.uniform() < keep_prob:
                out[r, c] = x[r, c] * inv_keep
    return out^


def dropout_cached(
    x: Tensor2D, p: Float64, training: Bool, mut rng: Rng
) raises -> DropoutResult:
    """Apply inverted dropout, also returning the kept-mask so backward can reuse it.

    Same semantics as dropout, drawing in the same fixed row-major order. Eval
    mode (or p == 0) returns an identity copy and an all-ones mask without drawing.

    Args:
        x: Input, shape [N, C].
        p: Drop probability in [0, 1).
        training: If false (or p == 0), returns an identity result without drawing.
        rng: Random generator, advanced one draw per element in the training branch.

    Returns:
        A DropoutResult with output [N, C], mask [N, C], and applied scale.
        Allocates the output and mask; reads x only.

    Raises:
        Error: If p is outside [0, 1) (rejects NaN too, matching dropout).
    """
    if not (p >= 0.0 and p < 1.0):
        raise Error("dropout_cached: p must be in [0, 1), got " + String(p))
    if not training or p <= 0.0:
        # Identity: all-ones mask, unit scale, no rng drawn.
        return DropoutResult(x.copy(), ones_2d(x.rows, x.cols), 1.0)

    var keep_prob = 1.0 - p
    var inv_keep = 1.0 / keep_prob
    var output = zeros_2d(x.rows, x.cols)
    var mask = zeros_2d(x.rows, x.cols)
    for r in range(x.rows):
        for c in range(x.cols):
            # Same draw and order as dropout: keep iff a fresh uniform lands below
            # keep_prob. Record the 0/1 kept indicator alongside the scaled value.
            if rng.uniform() < keep_prob:
                mask[r, c] = 1.0
                output[r, c] = x[r, c] * inv_keep
    return DropoutResult(output^, mask^, inv_keep)


def dropout_backward(
    mask: Tensor2D, inv_keep: Float64, d_out: Tensor2D
) raises -> Tensor2D:
    """VJP of the mask-fixed dropout forward: `(mask * inv_keep) * d_out`.

    The forward is a diagonal linear map `output = (mask * inv_keep) * x`, so its
    transpose is the same diagonal. Both `mask` and `inv_keep` must come straight
    from the paired forward's DropoutResult, not recomputed from p, so forward and
    backward stay exact inverses in every mode.

    Args:
        mask: The forward's kept-mask, shape [N, C], entries in {0, 1}.
        inv_keep: The scale the forward applied.
        d_out: Upstream gradient, shape [N, C].

    Returns:
        Gradient dL/dx, shape [N, C]. Allocates; reads its args.

    Raises:
        Error: If mask and d_out shapes do not match.
    """
    if mask.rows != d_out.rows or mask.cols != d_out.cols:
        raise Error("dropout_backward shape mismatch")
    var out = zeros_2d(mask.rows, mask.cols)
    for r in range(mask.rows):
        for c in range(mask.cols):
            out[r, c] = d_out[r, c] * mask[r, c] * inv_keep
    return out^
