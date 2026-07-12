"""MLP — the Transformer's position-wise feed-forward block.

    forward(x) = down(gelu(up(x)))

Two Linears with a GELU between them: `up` projects C -> hidden, `down` projects
hidden -> C, mapping [N, C] -> [N, C] through a wider hidden bottleneck. The
hidden width is an explicit constructor argument (GPT-2 uses 4C), never hardcoded.
"""

from llm.nn.gelu import gelu_rows, gelu_rows_backward
from llm.nn.linear import Linear, LinearCache
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.utils.random import Rng


@fieldwise_init
struct MLPCache(Copyable, Movable):
    """What MLP.backward needs, one field per stage of down(gelu(up(x))).

    The up Linear's cache (its input x), the pre-activation hidden (the up output,
    which gelu_rows_backward differentiates), and the down Linear's cache (its
    input, the activated hidden). Valid only for the forward call that produced it.
    """

    var up_cache: LinearCache  # holds x        [N, C]
    var hidden: Tensor2D  # pre-activation      [N, hidden]
    var down_cache: LinearCache  # holds gelu(hidden) [N, hidden]


@fieldwise_init
struct MLPForward(Copyable, Movable):
    """Output of forward_cached plus the cache its backward consumes."""

    var output: Tensor2D  # [N, C]
    var cache: MLPCache

    def split(deinit self, mut cache_slot: MLPCache) -> Tensor2D:
        """Consume this forward: move the cache into `cache_slot`, return the output.

        Lets an assembly site take both pieces by move instead of copying each
        out of a live struct (a field cannot be transferred with `^`).

        Args:
            cache_slot: Receives the cache by move.

        Returns:
            The output tensor, shape [N, C], by move.
        """
        cache_slot = self.cache^
        return self.output^


@fieldwise_init
struct MLP(Copyable, Movable):
    """Position-wise feed-forward block: down(gelu(up(x)))."""

    var up: Linear  # C -> hidden
    var down: Linear  # hidden -> C

    @staticmethod
    def init_random(mut rng: Rng, d_model: Int, d_hidden: Int) raises -> MLP:
        """Build an MLP with both Linears from GPT-2's normal(0, 0.02), zero biases.

        Args:
            rng: Random generator, advanced (up-projection weights first).
            d_model: Feature count C.
            d_hidden: Hidden width.

        Returns:
            An MLP with up (C -> hidden) and down (hidden -> C). Allocates both;
            deterministic given rng's state.

        Raises:
            Error: On non-positive dimensions (via Linear.init_random).
        """
        var up = Linear.init_random(rng, d_model, d_hidden)  # C -> hidden
        var down = Linear.init_random(rng, d_hidden, d_model)  # hidden -> C
        return MLP(up^, down^)

    def forward(self, x: Tensor2D) raises -> Tensor2D:
        """Compute down(gelu(up(x))).

        Args:
            x: Input, shape [N, C].

        Returns:
            Output, shape [N, C]. Allocates the intermediate and result; reads
            self only.

        Raises:
            Error: On a feature-count mismatch (surfaced by the up Linear).
        """
        var hidden = self.up.forward(x)  # [N, C] -> [N, hidden]
        var activated = gelu_rows(hidden)  # elementwise GELU
        return self.down.forward(activated)  # [N, hidden] -> [N, C]

    def forward_cached(self, var x: Tensor2D) raises -> MLPForward:
        """Compute forward(x), additionally caching what backward needs.

        The cache holds the two Linear caches and the pre-activation hidden. x is
        taken by value and threaded through by move — each stage's forward is
        split into (output, cache) and both pieces move on, so no [N, *]
        activation is copied on the way.

        Args:
            x: Input, shape [N, C]; moved through the stages.

        Returns:
            An MLPForward with output [N, C] and the cache. Allocates the
            intermediates, output, and cache; cache valid only for this call.

        Raises:
            Error: On a feature-count mismatch (via the up Linear).
        """
        var up_fwd = self.up.forward_cached(x^)  # x moves into the up cache
        var up_cache = LinearCache(
            zeros_2d(0, 0)
        )  # placeholder, replaced by the move
        var hidden_pre = up_fwd^.split(
            up_cache
        )  # cache -> up_cache; returns [N, hidden]
        var activated = gelu_rows(hidden_pre)  # [N, hidden], elementwise GELU
        var down_fwd = self.down.forward_cached(
            activated^
        )  # activated moves in
        var down_cache = LinearCache(
            zeros_2d(0, 0)
        )  # placeholder, replaced by the move
        var output = down_fwd^.split(
            down_cache
        )  # cache -> down_cache; returns [N, C]
        return MLPForward(
            output^, MLPCache(up_cache^, hidden_pre^, down_cache^)
        )

    def backward(mut self, cache: MLPCache, d_out: Tensor2D) raises -> Tensor2D:
        """Backprop through down(gelu(up(x))), threading the gradient right to left.

            d_activated = down.backward(down_cache, d_out)         [N, hidden]
            d_hidden    = gelu_rows_backward(hidden, d_activated)  [N, hidden]
            d_x         = up.backward(up_cache, d_hidden)          [N, C]

        down and up each accumulate their own parameter grads (+=) inside their
        backward.

        Args:
            cache: The two Linear caches and the pre-activation hidden.
            d_out: Upstream gradient, shape [N, C].

        Returns:
            Gradient d_x, shape [N, C]. Allocates. Mutates self.up and self.down
            parameter grads.

        Raises:
            Error: On a shape mismatch (surfaced by the sub-backwards).
        """
        var d_activated = self.down.backward(cache.down_cache, d_out)
        var d_hidden = gelu_rows_backward(cache.hidden, d_activated)
        return self.up.backward(cache.up_cache, d_hidden)
