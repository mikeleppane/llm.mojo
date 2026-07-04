# MLP — the Transformer's position-wise feed-forward block.
#
#     forward(x) = down(gelu(up(x)))
#
# Two Linears with a GELU between them: `up` projects C -> hidden, `down` projects
# hidden -> C, so the block maps [N, C] -> [N, C] with a wider hidden bottleneck
# in the middle. The hidden width is an explicit constructor argument, never
# hardcoded — GPT-2 uses hidden = 4C, but that ratio is the model's decision to
# pass in, not this block's to assume.

from llm.nn.gelu import gelu_rows, gelu_rows_backward
from llm.nn.linear import Linear, LinearCache
from llm.tensor.tensor2d import Tensor2D
from llm.utils.random import Rng


@fieldwise_init
struct MLPCache(Copyable, Movable):
    # What MLP.backward needs, one field per stage of down(gelu(up(x))): the up
    # Linear's cache (its input x), the pre-activation hidden (the up output,
    # which gelu_rows_backward differentiates), and the down Linear's cache (its
    # input, the activated hidden). Valid only for the forward call that produced
    # it.
    var up_cache: LinearCache  # holds x        [N, C]
    var hidden: Tensor2D  # pre-activation      [N, hidden]
    var down_cache: LinearCache  # holds gelu(hidden) [N, hidden]


@fieldwise_init
struct MLPForward(Copyable, Movable):
    # forward_cached's output plus the cache its backward consumes.
    var output: Tensor2D  # [N, C]
    var cache: MLPCache


@fieldwise_init
struct MLP(Copyable, Movable):
    var up: Linear  # C -> hidden
    var down: Linear  # hidden -> C

    @staticmethod
    def init_random(mut rng: Rng, d_model: Int, d_hidden: Int) raises -> MLP:
        # An MLP with both Linears drawn from GPT-2's normal(0, 0.02) scheme and
        # zero biases. Mutates rng (advances its state, up-projection weights
        # first); allocates both layers; deterministic given the generator's
        # state. Raises (via Linear.init_random) on non-positive dimensions.
        var up = Linear.init_random(rng, d_model, d_hidden)  # C -> hidden
        var down = Linear.init_random(rng, d_hidden, d_model)  # hidden -> C
        return MLP(up^, down^)

    def forward(self, x: Tensor2D) raises -> Tensor2D:
        # [N, C] -> [N, C] as down(gelu(up(x))). Reads self only; allocates the
        # intermediate and result; raises on a feature-count mismatch (surfaced
        # by the up-projection Linear).
        var hidden = self.up.forward(x)  # [N, C] -> [N, hidden]
        var activated = gelu_rows(hidden)  # elementwise GELU
        return self.down.forward(activated)  # [N, hidden] -> [N, C]

    def forward_cached(self, x: Tensor2D) raises -> MLPForward:
        # [N, C] -> MLPForward: the same output as forward, plus the cache
        # backward needs (the two Linear caches and the pre-activation hidden).
        # Reads self; allocates the intermediates, the output, and the cache;
        # raises on a feature-count mismatch (via the up Linear). The cache is
        # valid only for this call.
        var up_fwd = self.up.forward_cached(x)  # output = hidden [N, hidden]
        var activated = gelu_rows(up_fwd.output)  # [N, hidden]
        var down_fwd = self.down.forward_cached(activated)  # output [N, C]
        var cache = MLPCache(
            up_fwd.cache.copy(),  # LinearCache(x)
            up_fwd.output.copy(),  # pre-activation hidden
            down_fwd.cache.copy(),  # LinearCache(gelu(hidden))
        )
        return MLPForward(down_fwd.output.copy(), cache^)

    def backward(mut self, cache: MLPCache, d_out: Tensor2D) raises -> Tensor2D:
        # Reverse of down(gelu(up(x))): compose the three stage backwards in
        # reverse order, threading the gradient right to left.
        #   d_activated = down.backward(down_cache, d_out)         [N, hidden]
        #   d_hidden    = gelu_rows_backward(hidden, d_activated)  [N, hidden]
        #   d_x         = up.backward(up_cache, d_hidden)          [N, C]
        # down and up each accumulate their own parameter grads (+=) inside their
        # backward. Mutates self.up and self.down parameter grads; allocates and
        # returns d_x; raises on a shape mismatch (surfaced by the sub-backwards).
        var d_activated = self.down.backward(cache.down_cache, d_out)
        var d_hidden = gelu_rows_backward(cache.hidden, d_activated)
        return self.up.backward(cache.up_cache, d_hidden)
