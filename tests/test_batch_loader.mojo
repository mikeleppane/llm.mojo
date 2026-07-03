# Tests for BatchLoader — the heart of the dataset pipeline.
#
# Every test uses a synthetic corpus ids = [0, 1, 2, ..., N-1], so token id ==
# position and every expected value is computable by eye. That turns otherwise
# opaque windowing into arithmetic: a window starting at s has input_at(b, t) ==
# s + t and target_at(b, t) == s + 1 + t. The suite pins the three roadmap
# criteria — [B, T] shapes, the shift-by-one link between inputs and targets, and
# same-seed-identical-batches — plus epoch coverage, remainder dropping, and the
# exact minimum-size boundary.

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
    # B=4, T=8: every produced batch reports [4, 8] and holds 32 flat elements
    # per array. (Roadmap criterion 1.)
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
    # For every batch, targets are inputs shifted one position; the last target
    # is the corpus token immediately after the window. Because id == position,
    # input_at(b, 0) is the window's start s, so target_at(b, T-1) must be s + T.
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
    # The shift-by-one contract holds under overlapping windows (stride 1) and
    # non-overlapping ones (stride == T). (Roadmap criterion 2.)
    var overlapping = BatchLoader(_range_dataset(50), 4, 8, stride=1)
    _check_shift_by_one(overlapping, 8)
    var tiling = BatchLoader(_range_dataset(50), 4, 8, stride=8)
    _check_shift_by_one(tiling, 8)


def test_same_seed_identical_batches() raises:
    # Two loaders over identical data, both seeded 7, produce the same batch
    # sequence element-for-element across a full epoch; a different seed produces
    # a different order. (Roadmap criterion 3.)
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
    # start_epoch(seed) must reproduce the same order regardless of what the
    # loader did before — otherwise the order would depend on the whole history
    # of prior epochs, not just the seed, and "same seed -> same batches" would
    # quietly fail mid-training. Run a full epoch, then re-seed with the same
    # value: the first batch must match the first batch of the first epoch.
    var loader = BatchLoader(_range_dataset(80), 3, 5, stride=5)
    loader.start_epoch(5)
    var first = loader.next_batch()
    while loader.has_next():
        _ = loader.next_batch()  # exhaust the epoch (order is now consumed)
    loader.start_epoch(5)  # same seed again
    var again = loader.next_batch()
    assert_true(_batches_equal(first, again))


def test_epoch_covers_all_windows_once() raises:
    # N=46, B=3, T=5, stride=5 -> exactly 9 windows (starts 0,5,...,40), 3 full
    # batches, nothing dropped. Collect the start of every window seen in one
    # epoch; sorted, it must equal the full expected start list with no repeats.
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
    # N=51, B=4, T=5, stride=5 -> 10 windows; 10 // 4 == 2 batches, and the two
    # leftover windows are dropped. The iterator yields exactly 2 batches.
    var loader = BatchLoader(_range_dataset(51), 4, 5, stride=5)
    assert_equal(loader.num_windows(), 10)
    assert_equal(loader.num_batches(), 2)
    loader.start_epoch(0)
    var count = 0
    while loader.has_next():
        _ = loader.next_batch()
        count += 1
    assert_equal(count, 2)
    assert_false(loader.has_next())


def test_window_bounds() raises:
    # B=2, T=3, stride=3: the minimum viable corpus is (B-1)*stride + T + 1 == 7
    # tokens (exactly 2 windows, 1 batch). It constructs and iterates cleanly.
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
    # Degenerate batch/seq/stride arguments are rejected at construction.
    with assert_raises():
        _ = BatchLoader(_range_dataset(50), 0, 8, stride=8)  # B < 1
    with assert_raises():
        _ = BatchLoader(_range_dataset(50), 4, 0, stride=8)  # T < 1
    with assert_raises():
        _ = BatchLoader(_range_dataset(50), 4, 8, stride=0)  # stride < 1


def test_overfit_batch_is_fixed() raises:
    # overfit_batch returns the first B consecutive non-overlapping windows and
    # is bit-stable across calls — the fixed batch later parts drive to ~0 loss.
    var first = overfit_batch(_range_dataset(20), 2, 3)
    var second = overfit_batch(_range_dataset(20), 2, 3)
    assert_true(_batches_equal(first, second))
    # First window starts at 0, second at T == 3 (non-overlapping inputs).
    assert_equal(first.input_at(0, 0), 0)
    assert_equal(first.input_at(1, 0), 3)
    assert_equal(first.target_at(0, 0), 1)


def test_end_to_end_tinyshakespeare() raises:
    # The whole pipeline on the real corpus: load text, build a char vocab,
    # encode, split 90/10, and window it. Proves the pieces compose and that a
    # decoded input row is the actual corpus text — not just shape-correct noise.
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
