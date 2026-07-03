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

from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.utils.random import Rng


def dropout(
    x: Tensor2D, p: Float64, training: Bool, mut rng: Rng
) raises -> Tensor2D:
    # [N, C] -> [N, C]. Reads x; allocates the result. Mutates rng only in the
    # training/p>0 branch (one uniform draw per element). Raises if p is outside
    # [0, 1) — p = 1 would drop everything and divide by zero in the scale.
    #
    # Eval mode, or p exactly 0, returns an identity copy WITHOUT drawing: p == 0
    # is the caller's explicit "no dropout" sentinel, so the exact comparison is
    # deliberate, and short-circuiting is what guarantees the generator is left
    # untouched.
    if p < 0.0 or p >= 1.0:
        raise Error("dropout: p must be in [0, 1), got " + String(p))
    if not training or p == 0.0:
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
