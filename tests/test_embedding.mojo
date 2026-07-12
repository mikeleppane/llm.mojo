"""Tests for Embedding, the id-indexed lookup table used for token and positional embeddings, including the out-of-range guard at both bounds."""

from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    TestSuite,
)

from llm.nn.embedding import Embedding
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import from_rows
from llm.utils.random import Rng


def make_embedding() raises -> Embedding:
    """Build a hand-crafted [V=4, C=3] table with distinct, recognizable rows.
    """
    var table = from_rows(
        [
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
        ]
    )
    return Embedding(Parameter(table^))


def test_gather_returns_table_rows() raises:
    """Gathering a list of ids returns the corresponding table rows in order."""
    var emb = make_embedding()
    var out = emb.forward([2, 0, 3])  # [N=3, C=3]
    assert_equal(out.rows, 3)
    assert_equal(out.cols, 3)
    # Row 0 of the output is table row 2, row 1 is table row 0, row 2 is table 3.
    var expected_ids = [2, 0, 3]
    for i in range(3):
        var t = Float64(expected_ids[i])
        assert_almost_equal(out[i, 0], t + 0.0, atol=1e-15)
        assert_almost_equal(out[i, 1], t + 0.1, atol=1e-15)
        assert_almost_equal(out[i, 2], t + 0.2, atol=1e-15)


def test_repeated_ids_gather_same_row() raises:
    """A repeated id gathers the identical row each time."""
    var emb = make_embedding()
    var out = emb.forward([1, 1])
    for j in range(3):
        assert_almost_equal(out[0, j], out[1, j], atol=1e-15)


def test_positional_use_returns_leading_rows() raises:
    """Positional use (ids = 0..T-1) returns the first T table rows in order."""
    var emb = make_embedding()
    var out = emb.forward([0, 1, 2])
    for i in range(3):
        var t = Float64(i)
        assert_almost_equal(out[i, 0], t + 0.0, atol=1e-15)
        assert_almost_equal(out[i, 1], t + 0.1, atol=1e-15)


def test_negative_id_raises() raises:
    """A negative id raises rather than reading out of bounds."""
    var emb = make_embedding()
    with assert_raises(contains="out of range"):
        _ = emb.forward([0, -1, 2])


def test_id_at_or_past_vocab_raises() raises:
    """An id equal to the vocab size raises rather than reading out of bounds.
    """
    var emb = make_embedding()  # V = 4, so 4 is the first invalid id
    with assert_raises(contains="out of range"):
        _ = emb.forward([4])


def test_init_random_shape_and_determinism() raises:
    """Random init yields the requested [V, C] shape and is seed-deterministic.
    """
    var rng_a = Rng(99)
    var rng_b = Rng(99)
    var ea = Embedding.init_random(rng_a, 5, 6)  # [V=5, C=6]
    var eb = Embedding.init_random(rng_b, 5, 6)
    assert_equal(ea.table.value.rows, 5)
    assert_equal(ea.table.value.cols, 6)
    for r in range(5):
        for c in range(6):
            assert_almost_equal(
                ea.table.value[r, c], eb.table.value[r, c], atol=1e-15
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
