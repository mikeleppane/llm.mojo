# Sliding-window batching over a token dataset.
#
# This is where the corpus becomes training batches. A *window* of length T + 1
# starting at position s covers ids[s .. s + T]; its first T tokens are the model
# input and its last T tokens are the target (the input shifted left by one), so
# the network at position t is always trained to predict token t + 1. That
# shift-by-one is built into the window here, not bolted on later.
#
# An *epoch* is one pass over all window starts. The starts are enumerated once
# (0, stride, 2*stride, ...), optionally shuffled by a seeded RNG, and consumed B
# at a time. Because the order is a permutation of a fixed list, the same seed
# replays the exact same batch sequence — a training run is reproducible and a
# test can assert on it. Windows that don't fill a final batch of B are dropped
# (standard practice; it keeps every batch the same shape). Consequently
# num_batches() == num_windows() // B.
#
# `stride` is explicit: pass stride == seq_len for non-overlapping windows (each
# token used once as an input), or a smaller stride to oversample with overlap.
# There is no magic default — the caller states the step it wants.

from llm.utils import Rng

from .batch import TokenBatch
from .dataset import TokenDataset


struct BatchLoader(Movable):
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
        # Build a loader over `dataset`. Raises on degenerate arguments
        # (batch_size, seq_len, or stride < 1) or a dataset too small to fill even
        # one batch of B windows. The window starts are enumerated in natural
        # order so a fresh loader is immediately iterable; call start_epoch to
        # shuffle them.
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
        self.order = []
        self.cursor = 0

        # Enumerate window starts: every s with s + seq_len + 1 <= size, i.e.
        # s <= size - seq_len - 1, stepping by stride from 0.
        var max_start = self.dataset.size() - seq_len - 1
        var s = 0
        while s <= max_start:
            self.order.append(s)
            s += stride

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
        # Total window starts in one epoch.
        return len(self.order)

    def num_batches(self) -> Int:
        # Full batches per epoch; the remainder windows are dropped.
        return self.num_windows() // self.batch_size

    def start_epoch(mut self, seed: UInt64):
        # Begin a fresh epoch: shuffle the window starts with Rng(seed) and rewind
        # the cursor. The same seed always yields the same order, so batches are
        # reproducible. Callers who want per-epoch variation pass seed + epoch.
        var rng = Rng(seed)
        rng.shuffle(self.order)
        self.cursor = 0

    def has_next(self) -> Bool:
        # True while at least one full batch of B windows remains unread.
        return self.cursor + self.batch_size <= len(self.order)

    def next_batch(mut self) raises -> TokenBatch:
        # Assemble the next B windows into a TokenBatch. For each window start s,
        # inputs = ids[s : s+T] and targets = ids[s+1 : s+T+1]. Raises if called
        # without a full batch remaining (guard with has_next). Advances the
        # cursor by B; allocates the batch's two flat arrays.
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
    # Return the first B consecutive non-overlapping windows as one fixed batch,
    # with no shuffling. This is the tiny, constant batch that later parts drive
    # to near-zero loss to prove the model and training loop are wired correctly;
    # calling it twice must return identical batches, so it must never touch the
    # RNG. Built by windowing with stride == seq_len and taking the first batch.
    var loader = BatchLoader(
        dataset.copy(), batch_size, seq_len, stride=seq_len
    )
    return loader.next_batch()
