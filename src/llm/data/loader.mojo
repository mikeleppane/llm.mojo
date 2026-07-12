"""Sliding-window batching over a token dataset.

A window of length T + 1 starting at position s covers ids[s .. s + T]; its
first T tokens are the model input and its last T tokens are the target (the
input shifted left by one), so the network at position t is trained to predict
token t + 1. An epoch is one pass over all window starts (0, stride, 2*stride,
...), optionally shuffled by a seeded RNG and consumed B at a time; the same
seed replays the exact same batch sequence, so a run is reproducible. Windows
that don't fill a final batch of B are dropped, so num_batches() ==
num_windows() // B. Pass stride == seq_len for non-overlapping windows, or a
smaller stride to oversample with overlap.
"""

from llm.utils import Rng

from .batch import TokenBatch
from .dataset import TokenDataset


def _window_starts(size: Int, seq_len: Int, stride: Int) -> List[Int]:
    """Enumerate window starts in natural order.

    Every s with s + seq_len + 1 <= size (so both ids[s : s+T] and
    ids[s+1 : s+T+1] stay in bounds), stepping by stride from 0.

    Args:
        size: Number of tokens in the dataset.
        seq_len: T, the window's input/target length.
        stride: Step between consecutive window starts.

    Returns:
        The window starts. Allocates.
    """
    var starts: List[Int] = []
    var max_start = size - seq_len - 1
    var s = 0
    while s <= max_start:
        starts.append(s)
        s += stride
    return starts^


struct BatchLoader(Movable):
    """Windows a token dataset into reproducible, shuffleable training batches.
    """

    var dataset: TokenDataset  # the token sequence being windowed
    var batch_size: Int  # B: windows per batch
    var seq_len: Int  # T: input/target length per window
    var stride: Int  # step between consecutive window starts
    var order: List[Int]  # this epoch's window starts, in consumption order
    var cursor: Int  # index into `order` of the next unread window

    def __init__(
        out self,
        var dataset: TokenDataset,
        batch_size: Int,
        seq_len: Int,
        stride: Int,
    ) raises:
        """Build a loader over `dataset`.

        The window starts are enumerated in natural order so a fresh loader is
        immediately iterable; call start_epoch to shuffle them.

        Args:
            dataset: The token sequence to window (ownership transferred).
            batch_size: B, windows per batch, must be >= 1.
            seq_len: T, input/target length per window, must be >= 1.
            stride: Step between consecutive window starts, must be >= 1.

        Raises:
            Error: On a degenerate argument (< 1), or a dataset too small to
                fill even one batch of B windows.
        """
        if batch_size < 1:
            raise Error(
                "BatchLoader: batch_size must be >= 1, got "
                + String(batch_size)
            )
        if seq_len < 1:
            raise Error(
                "BatchLoader: seq_len must be >= 1, got " + String(seq_len)
            )
        if stride < 1:
            raise Error(
                "BatchLoader: stride must be >= 1, got " + String(stride)
            )

        self.dataset = dataset^
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride = stride
        self.cursor = 0
        # Window starts in natural order; start_epoch reshuffles a fresh copy of
        # this canonical enumeration so its result depends only on the seed.
        self.order = _window_starts(self.dataset.size(), seq_len, stride)

        if len(self.order) < batch_size:
            raise Error(
                "BatchLoader: dataset of "
                + String(self.dataset.size())
                + " tokens yields only "
                + String(len(self.order))
                + " windows (seq_len="
                + String(seq_len)
                + ", stride="
                + String(stride)
                + "), too few to fill a batch of "
                + String(batch_size)
            )

    def num_windows(self) -> Int:
        """Return the total window starts in one epoch."""
        return len(self.order)

    def num_batches(self) -> Int:
        """Return the full batches per epoch; remainder windows are dropped."""
        return self.num_windows() // self.batch_size

    def start_epoch(mut self, seed: UInt64):
        """Begin a fresh epoch: rebuild the window starts, shuffle, and rewind.

        Rebuilding the canonical (natural-order) starts before shuffling makes
        the result depend only on the seed; shuffling the previous epoch's
        already-permuted order in place would make each epoch a function of the
        whole prior history. Callers wanting per-epoch variation pass
        seed + epoch. Mutates self.

        Args:
            seed: RNG seed for the shuffle.
        """
        self.order = _window_starts(
            self.dataset.size(), self.seq_len, self.stride
        )
        var rng = Rng(seed)
        rng.shuffle(self.order)
        self.cursor = 0

    def has_next(self) -> Bool:
        """Return True while at least one full batch of B windows remains."""
        return self.cursor + self.batch_size <= len(self.order)

    def next_batch(mut self) raises -> TokenBatch:
        """Assemble the next B windows into a TokenBatch.

        For each window start s, inputs = ids[s : s+T] and
        targets = ids[s+1 : s+T+1]. Advances the cursor by B.

        Returns:
            The next batch. Allocates its two flat arrays.

        Raises:
            Error: If no full batch remains (guard with has_next).
        """
        if not self.has_next():
            raise Error(
                "BatchLoader.next_batch: no full batch remains (cursor="
                + String(self.cursor)
                + ", windows="
                + String(len(self.order))
                + ")"
            )
        var inputs: List[Int] = []
        var targets: List[Int] = []
        for row in range(self.batch_size):
            var start = self.order[self.cursor + row]
            for t in range(self.seq_len):
                inputs.append(self.dataset.ids[start + t])
                targets.append(self.dataset.ids[start + 1 + t])
        self.cursor += self.batch_size
        return TokenBatch(inputs^, targets^, self.batch_size, self.seq_len)


def overfit_batch(
    dataset: TokenDataset, batch_size: Int, seq_len: Int
) raises -> TokenBatch:
    """Return the first B non-overlapping windows as one fixed batch.

    No shuffling: this tiny, constant batch is driven to near-zero loss to prove
    the model and training loop are wired correctly, so calling it twice must
    return identical batches and it never touches the RNG. Built by windowing
    with stride == seq_len and taking the first batch.

    Args:
        dataset: The token sequence to window.
        batch_size: B, windows in the batch.
        seq_len: T, input/target length per window.

    Returns:
        The first batch. Allocates.

    Raises:
        Error: If the loader cannot be built or filled (see BatchLoader).
    """
    var loader = BatchLoader(
        dataset.copy(), batch_size, seq_len, stride=seq_len
    )
    return loader.next_batch()
