"""Integration test: a tiny network trains by hand-written backprop.

Compose the real layers Embedding -> LayerNorm -> MLP -> cross_entropy_rows into a
minimal classifier (vocab V = width C, so the MLP output is the logits). Two
things prove the chain-rule wiring end to end: the gradient of the loss with
respect to the embedding table (the furthest-back parameter) matches a central
finite difference of the whole forward; and twenty full-batch SGD steps on a
fixed input drive the loss strictly down.

Finite difference: L = cross_entropy_rows(forward(x)); central diff h = 1e-5;
mixed tolerance |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|.
"""

from std.testing import assert_true, TestSuite

from llm.nn.embedding import Embedding
from llm.nn.layernorm import LayerNorm
from llm.nn.mlp import MLP
from llm.nn.parameter import Parameter
from llm.tensor.ops import cross_entropy_rows, cross_entropy_rows_backward
from llm.tensor.tensor2d import Tensor2D
from llm.training.optimizer import sgd_step
from llm.utils.random import Rng


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    """Assert |analytic - numeric| <= 1e-7 + 1e-5 * |numeric| (mixed tolerance).
    """
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


# The network is built inline in each test (not via a shared factory) because a
# tuple of these move-only structs is awkward to return. Both tests replay the
# same seeded rng sequence — emb draws its table first, then mlp its weights — so
# they start from identical parameters. V = C = 5 (so the MLP output width is the
# vocab), hidden = 8, N = 4.


def sample_ids() raises -> List[Int]:
    return [1, 3, 0, 4]


def sample_targets() raises -> List[Int]:
    return [2, 0, 4, 1]


def forward_loss(
    table: Tensor2D,
    ln: LayerNorm,
    mlp: MLP,
    ids: List[Int],
    targets: List[Int],
) raises -> Float64:
    """Full forward from a raw embedding table to the scalar loss. Used for the
    finite difference (perturbing the table) and to read the loss during training.
    """
    var emb = Embedding(Parameter(table.copy()))
    var embedded = emb.forward(ids)  # [N, C]
    var normed = ln.forward(embedded)  # [N, C]
    var logits = mlp.forward(normed)  # [N, V]
    return cross_entropy_rows(logits, targets)


def test_embedding_table_grad_matches_finite_difference() raises:
    """The embedding-table gradient (threaded through every backward) matches a
    central finite difference of the whole forward."""
    var rng = Rng(31)
    var emb = Embedding.init_random(rng, 5, 5)  # table [V=5, C=5]
    var ln = LayerNorm.init_default(5)  # [1, 5]
    var mlp = MLP.init_random(rng, 5, 8)  # C=5 -> hidden=8 -> C=5
    var ids = sample_ids()
    var targets = sample_targets()

    # Analytic: one full forward_cached / backward pass down to the table.
    emb.table.zero_grad()
    var emb_fwd = emb.forward_cached(ids.copy())  # [N, C]
    var ln_fwd = ln.forward_cached(emb_fwd.output.copy())  # [N, C]
    var mlp_fwd = mlp.forward_cached(ln_fwd.output.copy())  # [N, V]
    var d_logits = cross_entropy_rows_backward(mlp_fwd.output, targets)
    var d_normed = mlp.backward(mlp_fwd.cache, d_logits)  # [N, C]
    var d_embedded = ln.backward(ln_fwd.cache, d_normed)  # [N, C]
    emb.backward(emb_fwd.cache, d_embedded)  # scatters into emb.table.grad

    # Numeric: perturb each table entry, rerun the whole forward.
    var table = emb.table.value.copy()
    var h = 1e-5
    for v in range(table.rows):
        for j in range(table.cols):
            var t_plus = table.copy()
            t_plus[v, j] = t_plus[v, j] + h
            var t_minus = table.copy()
            t_minus[v, j] = t_minus[v, j] - h
            var numeric = (
                forward_loss(t_plus, ln, mlp, ids, targets)
                - forward_loss(t_minus, ln, mlp, ids, targets)
            ) / (2.0 * h)
            assert_grad_close(emb.table.grad[v, j], numeric)


def test_twenty_sgd_steps_strictly_decrease_loss() raises:
    """Twenty full-batch SGD steps on a fixed input drive the loss strictly down.
    """
    var rng = Rng(31)
    var emb = Embedding.init_random(rng, 5, 5)  # table [V=5, C=5]
    var ln = LayerNorm.init_default(5)  # [1, 5]
    var mlp = MLP.init_random(rng, 5, 8)  # C=5 -> hidden=8 -> C=5
    var ids = sample_ids()
    var targets = sample_targets()
    var lr = 0.3

    var prev = 1e18
    for step in range(20):
        # Loss on the current parameters, before this step's update.
        var loss = forward_loss(emb.table.value, ln, mlp, ids, targets)
        if step > 0:
            assert_true(
                loss < prev,
                String("loss did not strictly decrease at step ")
                + String(step)
                + ": "
                + String(loss)
                + " !< "
                + String(prev),
            )
        prev = loss

        # Fresh gradients, one full backward, then an SGD update per parameter.
        emb.table.zero_grad()
        ln.weight.zero_grad()
        ln.bias.zero_grad()
        mlp.up.weight.zero_grad()
        mlp.up.bias.zero_grad()
        mlp.down.weight.zero_grad()
        mlp.down.bias.zero_grad()

        var emb_fwd = emb.forward_cached(ids.copy())
        var ln_fwd = ln.forward_cached(emb_fwd.output.copy())
        var mlp_fwd = mlp.forward_cached(ln_fwd.output.copy())
        var d_logits = cross_entropy_rows_backward(mlp_fwd.output, targets)
        var d_normed = mlp.backward(mlp_fwd.cache, d_logits)
        var d_embedded = ln.backward(ln_fwd.cache, d_normed)
        emb.backward(emb_fwd.cache, d_embedded)

        sgd_step(emb.table.value, emb.table.grad, lr)
        sgd_step(ln.weight.value, ln.weight.grad, lr)
        sgd_step(ln.bias.value, ln.bias.grad, lr)
        sgd_step(mlp.up.weight.value, mlp.up.weight.grad, lr)
        sgd_step(mlp.up.bias.value, mlp.up.bias.grad, lr)
        sgd_step(mlp.down.weight.value, mlp.down.weight.grad, lr)
        sgd_step(mlp.down.bias.value, mlp.down.bias.grad, lr)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
