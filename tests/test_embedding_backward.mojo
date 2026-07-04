# Tests for Embedding backward — the scatter-add of the upstream gradient.
#
# The forward gathers table rows by id; the backward scatters d_out back to those
# rows with +=. Three properties matter: touched rows match a finite difference;
# a repeated id ACCUMULATES (both occurrences sum into the one row — the classic
# scatter bug is to overwrite); and rows no id selected stay exactly zero.
#
# Finite-difference convention (D5, shared across this part's backward tests):
#   L = sum(cotangent ⊙ output); central diff h = 1e-5; tolerance
#   |analytic - numeric| <= 1e-7 + 1e-5 * |numeric|.

from std.testing import assert_almost_equal, assert_true, TestSuite

from llm.nn.embedding import Embedding, EmbeddingCache
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, from_rows


def assert_grad_close(analytic: Float64, numeric: Float64) raises:
    # D5 mixed tolerance |a - n| <= 1e-7 + 1e-5 * |n|.
    assert_true(
        abs(analytic - numeric) <= 1e-7 + 1e-5 * abs(numeric),
        String("grad mismatch: analytic=")
        + String(analytic)
        + " numeric="
        + String(numeric),
    )


def make_embedding() raises -> Embedding:
    # Table [V=5, C=3], asymmetric.
    var t = from_rows(
        [
            [0.1, -0.2, 0.3],
            [1.0, 0.5, -1.0],
            [-0.4, 0.9, 0.2],
            [0.7, -0.8, 1.1],
            [0.3, 0.0, -0.6],
        ]
    )
    return Embedding(Parameter(t^))


def sample_ids() raises -> List[Int]:
    # id 1 appears twice (positions 0 and 2); ids 2 and 4 never appear.
    return [1, 3, 1, 0]


def cotangent() raises -> Tensor2D:
    # Fixed asymmetric d_out [N=4, C=3].
    return from_rows(
        [[0.7, -0.2, 1.3], [0.1, 0.9, -1.1], [-0.6, 0.3, 0.2], [0.4, -0.5, 0.8]]
    )


def projected(emb: Embedding, ids: List[Int], cot: Tensor2D) raises -> Float64:
    var y = emb.forward(ids)
    var total = 0.0
    for i in range(y.rows):
        for j in range(y.cols):
            total += cot[i, j] * y[i, j]
    return total


def test_touched_rows_match_finite_difference() raises:
    var emb = make_embedding()
    var ids = sample_ids()
    var cot = cotangent()
    emb.table.zero_grad()
    var fwd = emb.forward_cached(ids)
    emb.backward(fwd.cache, cot)

    var h = 1e-5
    for v in range(emb.table.value.rows):
        for j in range(emb.table.value.cols):
            var t_plus = emb.table.value.copy()
            t_plus[v, j] = t_plus[v, j] + h
            var emb_plus = Embedding(Parameter(t_plus^))
            var t_minus = emb.table.value.copy()
            t_minus[v, j] = t_minus[v, j] - h
            var emb_minus = Embedding(Parameter(t_minus^))
            var numeric = (
                projected(emb_plus, ids, cot) - projected(emb_minus, ids, cot)
            ) / (2.0 * h)
            assert_grad_close(emb.table.grad[v, j], numeric)


def test_repeated_id_accumulates() raises:
    # id 1 is at positions 0 and 2, so its row gradient is the SUM of those two
    # cotangent rows — not just the last one written.
    var emb = make_embedding()
    var ids = sample_ids()
    var cot = cotangent()
    emb.table.zero_grad()
    var fwd = emb.forward_cached(ids)
    emb.backward(fwd.cache, cot)
    for j in range(emb.table.value.cols):
        assert_almost_equal(
            emb.table.grad[1, j], cot[0, j] + cot[2, j], atol=1e-12
        )
    # A single-occurrence id (3 at position 1, 0 at position 3) gets exactly its
    # one cotangent row.
    for j in range(emb.table.value.cols):
        assert_almost_equal(emb.table.grad[3, j], cot[1, j], atol=1e-12)
        assert_almost_equal(emb.table.grad[0, j], cot[3, j], atol=1e-12)


def test_untouched_rows_stay_zero() raises:
    # ids used are {0, 1, 3}; rows 2 and 4 are never gathered, so their gradient
    # must remain exactly zero (not merely small).
    var emb = make_embedding()
    var ids = sample_ids()
    var cot = cotangent()
    emb.table.zero_grad()
    var fwd = emb.forward_cached(ids)
    emb.backward(fwd.cache, cot)
    for j in range(emb.table.value.cols):
        assert_true(emb.table.grad[2, j] == 0.0)
        assert_true(emb.table.grad[4, j] == 0.0)


def test_backward_accumulates_across_calls() raises:
    # Two backward calls without zero_grad() between them exactly double the row
    # gradients — the same accumulation contract the other layers pin.
    var emb = make_embedding()
    var ids = sample_ids()
    var cot = cotangent()
    emb.table.zero_grad()
    var fwd = emb.forward_cached(ids)
    emb.backward(fwd.cache, cot)
    var once = emb.table.grad.copy()
    emb.backward(fwd.cache, cot)
    for v in range(emb.table.value.rows):
        for j in range(emb.table.value.cols):
            assert_true(emb.table.grad[v, j] == 2.0 * once[v, j])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
