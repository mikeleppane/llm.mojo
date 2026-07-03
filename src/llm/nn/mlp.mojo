# MLP — the Transformer's position-wise feed-forward block.
#
#     forward(x) = down(gelu(up(x)))
#
# Two Linears with a GELU between them: `up` projects C -> hidden, `down` projects
# hidden -> C, so the block maps [N, C] -> [N, C] with a wider hidden bottleneck
# in the middle. The hidden width is an explicit constructor argument, never
# hardcoded — GPT-2 uses hidden = 4C, but that ratio is the model's decision to
# pass in, not this block's to assume.

from llm.nn.gelu import gelu_rows
from llm.nn.linear import Linear
from llm.tensor.tensor2d import Tensor2D
from llm.utils.random import Rng


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
