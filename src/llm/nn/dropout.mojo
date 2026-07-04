# Dropout — inverted dropout with train/eval as an argument, not stored state.
#
# In training, each element is independently kept with probability 1 - p and, if
# kept, scaled by 1/(1-p); dropped elements become 0. The scaling ("inverted"
# dropout) keeps the expected value of each element unchanged, so evaluation needs
# no compensating rescale — it is a plain identity.
#
# The mode is a *call-site argument*, not a field, so train vs eval is explicit
# wherever dropout runs. Two rng invariants matter enough to be tested: eval mode
# (and p == 0) consume no draws, so disabling dropout never perturbs the seeded
# generator that downstream tests depend on; and training draws exactly one
# uniform per element in a fixed order, so a seed replays the same mask.
#
# Stateless and parameter-free, so it is a free function, not a struct.

from llm.tensor.tensor2d import Tensor2D, ones_2d, zeros_2d
from llm.utils.random import Rng


@fieldwise_init
struct DropoutResult(Copyable, Movable):
    # dropout_cached's output plus the kept-mask backward needs. mask[i, j] is 1.0
    # for a kept element and 0.0 for a dropped one — the SAME mask the forward
    # applied, so backward and forward scale exactly the same elements. In eval
    # mode (or p == 0) nothing is dropped, so the mask is all ones.
    var output: Tensor2D  # [N, C]
    var mask: Tensor2D  # [N, C], entries in {0, 1}


def dropout(
    x: Tensor2D, p: Float64, training: Bool, mut rng: Rng
) raises -> Tensor2D:
    # [N, C] -> [N, C]. Reads x; allocates the result. Mutates rng only in the
    # training/p>0 branch (one uniform draw per element). Raises if p is outside
    # [0, 1) — p = 1 would drop everything and divide by zero in the scale.
    #
    # The guard is written as "not in range" rather than "p < 0 or p >= 1" so a
    # NaN p raises too: every comparison with NaN is false, so `p >= 0.0 and
    # p < 1.0` is false for NaN and the negation fires. A NaN slipping through
    # would zero the whole output while still consuming rng draws.
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
    # Same semantics as dropout, additionally returning the kept-mask so backward
    # can reuse it. In training with p > 0, each element is kept with probability
    # 1 - p (one uniform draw per element, in the same fixed row-major order
    # dropout uses) and scaled by 1/(1-p); dropped elements are 0. Eval mode (or
    # p == 0) returns an identity copy and an all-ones mask WITHOUT drawing —
    # disabling dropout must not perturb the seeded generator. Reads x; allocates
    # the output and mask; mutates rng only in the training/p>0 branch; raises if
    # p is outside [0, 1) (the not-in-range guard rejects NaN too, matching
    # dropout).
    if not (p >= 0.0 and p < 1.0):
        raise Error("dropout_cached: p must be in [0, 1), got " + String(p))
    if not training or p <= 0.0:
        return DropoutResult(x.copy(), ones_2d(x.rows, x.cols))

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
    return DropoutResult(output^, mask^)


def dropout_backward(
    mask: Tensor2D, p: Float64, d_out: Tensor2D
) raises -> Tensor2D:
    # VJP of the mask-fixed dropout forward. Given the mask, the forward scales
    # each kept element by 1/(1-p) and zeros the rest:
    #     output = (mask * inv_keep) ⊙ x,   inv_keep = 1/(1-p).
    # That is a diagonal linear map, and the transpose of a diagonal is itself:
    #     dL/dx = (mask * inv_keep) ⊙ d_out.
    # Reusing the SAME mask the forward drew is the whole point — a fresh draw
    # would scale a different set of elements than the forward kept, so the
    # gradient would not match the function actually computed. With p == 0 (or an
    # all-ones eval mask) inv_keep = 1 and this is the identity. This backward
    # pairs with dropout_cached's training-mode output. Shapes [N, C] and [N, C]
    # -> [N, C]. Reads its args; allocates the result; raises if p is outside
    # [0, 1) or the shapes mismatch.
    if not (p >= 0.0 and p < 1.0):
        raise Error("dropout_backward: p must be in [0, 1), got " + String(p))
    if mask.rows != d_out.rows or mask.cols != d_out.cols:
        raise Error("dropout_backward shape mismatch")
    var inv_keep = 1.0 / (1.0 - p)
    var out = zeros_2d(mask.rows, mask.cols)
    for r in range(mask.rows):
        for c in range(mask.cols):
            out[r, c] = d_out[r, c] * mask[r, c] * inv_keep
    return out^
