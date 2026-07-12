"""Token datasets and the train/validation split.

A `TokenDataset` is a thin, owning wrapper around a flat sequence of token ids
(the output of any tokenizer's `encode`), giving later layers a named thing with
a `size()` rather than a bare `List[Int]`. Tiny Shakespeare is one continuous
document, so the split is a single cut — train prefix, val suffix. Cutting on
the raw id sequence before any windowing guarantees no training window can ever
contain a validation token.
"""


struct TokenDataset(Copyable, Movable):
    """An owning wrapper around a flat, in-order token id sequence."""

    var ids: List[Int]  # the token id sequence, in order

    def __init__(out self, var ids: List[Int]):
        """Take ownership of `ids` (transfer with `^` at the call site)."""
        self.ids = ids^

    def size(self) -> Int:
        """Return the number of tokens in the dataset."""
        return len(self.ids)


struct TrainValSplit(Copyable, Movable):
    """The result of `train_val_split`: named train and val datasets.

    A struct rather than a tuple so the fields read at the call site
    (`split.train`, `split.val`).
    """

    var train: TokenDataset
    var val: TokenDataset

    def __init__(out self, var train: TokenDataset, var val: TokenDataset):
        self.train = train^
        self.val = val^


def train_val_split(
    ids: List[Int], val_fraction: Float64
) raises -> TrainValSplit:
    """Split a contiguous id sequence into a train prefix and val suffix.

    The validation count is `floor(len(ids) * val_fraction)` and the training
    set is everything before it. Computing the val count directly (rather than
    `floor(len * (1 - val_fraction))`) sidesteps a floating-point trap:
    `1.0 - 0.1` is `0.8999999999999999`, so the naive form would slice 100 ids
    as 89/11 instead of the intended 90/10.

    Args:
        ids: The contiguous token id sequence.
        val_fraction: Fraction of tokens held out for validation, in (0, 1).

    Returns:
        The train/val split. Allocates two new lists; does not mutate `ids`.

    Raises:
        Error: If val_fraction is not strictly inside (0, 1), or if the split
            would leave either side empty.
    """
    if not (val_fraction > 0.0 and val_fraction < 1.0):
        raise Error(
            "train_val_split: val_fraction must be in (0, 1), got "
            + String(val_fraction)
        )
    var n = len(ids)
    var val_count = Int(Float64(n) * val_fraction)  # floor, since n >= 0
    var split_index = n - val_count
    if val_count <= 0 or split_index <= 0:
        raise Error(
            "train_val_split: fraction "
            + String(val_fraction)
            + " leaves an empty split for "
            + String(n)
            + " tokens (need at least one token on each side)"
        )

    var train: List[Int] = []
    for i in range(split_index):
        train.append(ids[i])
    var val: List[Int] = []
    for i in range(split_index, n):
        val.append(ids[i])
    return TrainValSplit(TokenDataset(train^), TokenDataset(val^))
