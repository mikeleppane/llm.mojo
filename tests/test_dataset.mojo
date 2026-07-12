"""Tests for corpus loading and the train/val split: split arithmetic, clean partition, and a missing-corpus error message a reader can act on."""

from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from llm.data import load_text, TokenDataset, TrainValSplit, train_val_split


def _range_ids(n: Int) -> List[Int]:
    var ids: List[Int] = []
    for i in range(n):
        ids.append(i)
    return ids^


def test_split_sizes() raises:
    """100 ids, 10% held out -> train 90, val 10, and the sizes tile the whole.
    """
    var split = train_val_split(_range_ids(100), 0.1)
    assert_equal(split.train.size(), 90)
    assert_equal(split.val.size(), 10)
    assert_equal(split.train.size() + split.val.size(), 100)


def test_split_is_partition() raises:
    """Train is exactly the prefix ids[0:90], val the suffix ids[90:100]: order preserved, no overlap, no gap.
    """
    var split = train_val_split(_range_ids(100), 0.1)
    for i in range(90):
        assert_equal(split.train.ids[i], i)
    for i in range(10):
        assert_equal(split.val.ids[i], 90 + i)


def test_split_deterministic() raises:
    """No hidden RNG: two calls with the same arguments give identical splits.
    """
    var a = train_val_split(_range_ids(37), 0.25)
    var b = train_val_split(_range_ids(37), 0.25)
    assert_equal(a.train.size(), b.train.size())
    assert_equal(a.val.size(), b.val.size())
    for i in range(a.train.size()):
        assert_equal(a.train.ids[i], b.train.ids[i])
    for i in range(a.val.size()):
        assert_equal(a.val.ids[i], b.val.ids[i])


def test_split_invalid_fraction_raises() raises:
    """Reject fractions outside (0, 1), and any fraction that would empty a side.
    """
    # Fractions outside the open interval (0, 1) are rejected...
    with assert_raises():
        _ = train_val_split(_range_ids(100), 0.0)
    with assert_raises():
        _ = train_val_split(_range_ids(100), 1.0)
    with assert_raises():
        _ = train_val_split(_range_ids(100), -0.1)
    # ...and so is a fraction that, while in (0, 1), would empty a side: with 3
    # ids and 10% held out the val count floors to 0.
    with assert_raises():
        _ = train_val_split(_range_ids(3), 0.1)


def test_split_empty_ids_raises() raises:
    """An empty corpus can't be split: the empty-side guard fires instead of returning two empty datasets.
    """
    var empty: List[Int] = []
    with assert_raises():
        _ = train_val_split(empty, 0.1)


def test_corpus_missing_file_raises() raises:
    """A missing corpus fails with a message that names the download script, telling the reader how to fix it.
    """
    var raised = False
    try:
        _ = load_text("this/path/does/not/exist_xyz.txt")
    except e:
        raised = True
        assert_true("download_tinyshakespeare" in String(e))
    assert_true(raised)


def test_corpus_loads_real_file() raises:
    """The committed corpus loads, is the expected size, and begins with the canonical opening line.
    """
    var text = load_text("data/tinyshakespeare/input.txt")
    assert_true(text.byte_length() > 1_000_000)
    assert_true(text.startswith("First Citizen"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
