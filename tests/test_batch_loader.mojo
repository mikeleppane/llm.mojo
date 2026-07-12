"""Tests for BatchLoader, the heart of the dataset pipeline.

Every test uses a synthetic corpus ids = [0, 1, ..., N-1], so token id == position
and every expected value is computable by eye: a window starting at s has
input_at(b, t) == s + t and target_at(b, t) == s + 1 + t. The suite pins [B, T]
shapes, the shift-by-one link between inputs and targets, same-seed-identical
batches, epoch coverage, remainder dropping, and the minimum-size boundary.
"""

from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.testing import TestSuite

from llm.data import (
    TokenDataset,
    TokenBatch,
    BatchLoader,
    overfit_batch,
    load_text,
    train_val_split,
)
from llm.tokenizer import CharTokenizer


def _range_dataset(n: Int) -> TokenDataset:
    var ids: List[Int] = []
    for i in range(n):
        ids.append(i)
    return TokenDataset(ids^)


def _batches_equal(a: TokenBatch, b: TokenBatch) -> Bool:
    if a.batch_size != b.batch_size or a.seq_len != b.seq_len:
        return False
    if len(a.inputs) != len(b.inputs) or len(a.targets) != len(b.targets):
        return False
    for i in range(len(a.inputs)):
        if a.inputs[i] != b.inputs[i] or a.targets[i] != b.targets[i]:
            return False
    return True


def test_batch_shapes() raises:
    """Every batch at B=4, T=8 reports [4, 8] and holds 32 flat elements per array.
    """
    var loader = BatchLoader(_range_dataset(100), 4, 8, stride=8)
    var seen_any = False
    while loader.has_next():
        var batch = loader.next_batch()
        seen_any = True
        assert_equal(batch.batch_size, 4)
        assert_equal(batch.seq_len, 8)
        assert_equal(len(batch.inputs), 32)
        assert_equal(len(batch.targets), 32)
    assert_true(seen_any)


def _check_shift_by_one(mut loader: BatchLoader, seq_len: Int) raises:
    """Assert every batch's targets are its inputs shifted one position, and the
    last target is the token immediately after the window (target_at(b, T-1) == s + T).
    """
    var seen_any = False
    while loader.has_next():
        var batch = loader.next_batch()
        seen_any = True
        for b in range(batch.batch_size):
            for t in range(seq_len - 1):
                assert_equal(batch.target_at(b, t), batch.input_at(b, t + 1))
            var start = batch.input_at(b, 0)
            assert_equal(batch.target_at(b, seq_len - 1), start + seq_len)
    assert_true(seen_any)


def test_shift_by_one_property() raises:
    """The shift-by-one contract holds for overlapping (stride 1) and
    non-overlapping (stride == T) windows."""
    var overlapping = BatchLoader(_range_dataset(50), 4, 8, stride=1)
    _check_shift_by_one(overlapping, 8)
    var tiling = BatchLoader(_range_dataset(50), 4, 8, stride=8)
    _check_shift_by_one(tiling, 8)


def test_same_seed_identical_batches() raises:
    """Two loaders seeded alike produce the same batch sequence across an epoch; a
    different seed produces a different order."""
    var a = BatchLoader(_range_dataset(80), 3, 5, stride=5)
    var b = BatchLoader(_range_dataset(80), 3, 5, stride=5)
    a.start_epoch(7)
    b.start_epoch(7)
    var count = 0
    while a.has_next():
        assert_true(b.has_next())
        assert_true(_batches_equal(a.next_batch(), b.next_batch()))
        count += 1
    assert_false(b.has_next())
    assert_true(count > 0)

    # A different seed changes the order: at least one batch differs.
    var c = BatchLoader(_range_dataset(80), 3, 5, stride=5)
    var d = BatchLoader(_range_dataset(80), 3, 5, stride=5)
    c.start_epoch(7)
    d.start_epoch(999)
    var any_diff = False
    while c.has_next() and d.has_next():
        if not _batches_equal(c.next_batch(), d.next_batch()):
            any_diff = True
    assert_true(any_diff)


def test_start_epoch_depends_only_on_seed() raises:
    """Re-seeding reproduces the same order regardless of prior epochs: after a full
    epoch, start_epoch with the same seed yields the same first batch."""
    var loader = BatchLoader(_range_dataset(80), 3, 5, stride=5)
    loader.start_epoch(5)
    var first = loader.next_batch()
    while loader.has_next():
        _ = loader.next_batch()  # exhaust the epoch (order is now consumed)
    loader.start_epoch(5)  # same seed again
    var again = loader.next_batch()
    assert_true(_batches_equal(first, again))


def test_epoch_covers_all_windows_once() raises:
    """One epoch covers every window exactly once (N=46, B=3, T=5, stride=5 -> 9
    windows, 3 batches): sorted window starts equal 0,5,...,40 with no repeats.
    """
    var loader = BatchLoader(_range_dataset(46), 3, 5, stride=5)
    assert_equal(loader.num_windows(), 9)
    assert_equal(loader.num_batches(), 3)
    loader.start_epoch(3)
    var starts: List[Int] = []
    while loader.has_next():
        var batch = loader.next_batch()
        for b in range(batch.batch_size):
            starts.append(batch.input_at(b, 0))  # id == position == start
    assert_equal(len(starts), 9)
    sort(starts)
    for k in range(9):
        assert_equal(starts[k], k * 5)


def test_num_batches_drops_remainder() raises:
    """A remainder is dropped (N=51, B=4, T=5 -> 10 windows, 2 batches): exactly 2
    batches and 8 distinct windows appear, no duplicate or over-run past the tail.
    """
    var loader = BatchLoader(_range_dataset(51), 4, 5, stride=5)
    assert_equal(loader.num_windows(), 10)
    assert_equal(loader.num_batches(), 2)
    loader.start_epoch(0)
    var starts: List[Int] = []
    var count = 0
    while loader.has_next():
        var batch = loader.next_batch()
        for b in range(batch.batch_size):
            starts.append(batch.input_at(b, 0))  # id == position == start
        count += 1
    assert_equal(count, 2)
    assert_false(loader.has_next())
    # 8 windows seen, all distinct, all drawn from the valid start set {0,5,..,45}.
    assert_equal(len(starts), 8)
    sort(starts)
    for k in range(len(starts)):
        assert_true(starts[k] % 5 == 0 and starts[k] <= 45)
        if k > 0:
            assert_true(starts[k] != starts[k - 1])  # no duplicates


def test_window_bounds() raises:
    """The minimum viable corpus (B-1)*stride + T + 1 == 7 tokens constructs and
    iterates cleanly; one token fewer raises."""
    var loader = BatchLoader(_range_dataset(7), 2, 3, stride=3)
    assert_equal(loader.num_windows(), 2)
    assert_equal(loader.num_batches(), 1)
    var batch = loader.next_batch()
    # Last window starts at 3; its last target is ids[3 + 3] == 6, the final id.
    assert_equal(batch.target_at(1, 2), 6)
    # One token smaller yields only one window — too few to fill a batch of 2.
    with assert_raises():
        _ = BatchLoader(_range_dataset(6), 2, 3, stride=3)


def test_construct_invalid_args_raise() raises:
    """Degenerate batch/seq/stride arguments are rejected at construction."""
    with assert_raises():
        _ = BatchLoader(_range_dataset(50), 0, 8, stride=8)  # B < 1
    with assert_raises():
        _ = BatchLoader(_range_dataset(50), 4, 0, stride=8)  # T < 1
    with assert_raises():
        _ = BatchLoader(_range_dataset(50), 4, 8, stride=0)  # stride < 1


def test_overfit_batch_too_small_raises() raises:
    """A dataset too small for one non-overlapping [B, T+1] batch raises (B=2, T=3
    needs 7 tokens; 6 is one too few)."""
    with assert_raises():
        _ = overfit_batch(_range_dataset(6), 2, 3)


def test_overfit_batch_is_fixed() raises:
    """The overfit batch is the first B non-overlapping windows and is bit-stable
    across calls."""
    var first = overfit_batch(_range_dataset(20), 2, 3)
    var second = overfit_batch(_range_dataset(20), 2, 3)
    assert_true(_batches_equal(first, second))
    # First window starts at 0, second at T == 3 (non-overlapping inputs).
    assert_equal(first.input_at(0, 0), 0)
    assert_equal(first.input_at(1, 0), 3)
    assert_equal(first.target_at(0, 0), 1)


def test_end_to_end_tinyshakespeare() raises:
    """The whole pipeline on the real corpus: load, build a char vocab, encode,
    split 90/10, window it; a decoded input row is the actual corpus text."""
    var text = load_text("data/tinyshakespeare/input.txt")
    var tok = CharTokenizer.from_text(text)
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)

    var loader = BatchLoader(split.train.copy(), 4, 16, stride=16)
    # No start_epoch: natural order, so the first batch's first row is the very
    # start of the training ids, which equals the start of the corpus.
    var batch = loader.next_batch()
    assert_equal(batch.batch_size, 4)
    assert_equal(batch.seq_len, 16)

    # Shift-by-one holds on real ids too.
    for b in range(4):
        for t in range(15):
            assert_equal(batch.target_at(b, t), batch.input_at(b, t + 1))

    # Decode the first input row and compare it to the first 16 codepoints of the
    # corpus. Both must read "First Citizen:\nB".
    var row0: List[Int] = []
    for t in range(16):
        row0.append(batch.input_at(0, t))
    var decoded = tok.decode(row0)

    var expected = String("")
    var taken = 0
    for cp in text.codepoint_slices():
        if taken >= 16:
            break
        expected += String(cp)
        taken += 1
    assert_equal(decoded, expected)
    assert_equal(decoded, String("First Citizen:\nB"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
