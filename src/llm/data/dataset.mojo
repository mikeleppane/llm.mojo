# Token datasets and the train/validation split.
#
# A `TokenDataset` is a thin, owning wrapper around a flat sequence of token ids
# (the output of any tokenizer's `encode`). It exists so that later layers pass
# around a named thing with a `size()` rather than a bare `List[Int]`, and so the
# split has a typed result.
#
# Tiny Shakespeare is one continuous document, so the train/val split is a single
# cut: the train set is the prefix, the val set is the suffix. There is nothing
# to shuffle at the document level (there is only one document), and cutting on
# the raw id sequence — before any windowing — guarantees no training window can
# ever contain a validation token.


struct TokenDataset(Copyable, Movable):
    var ids: List[Int]  # the token id sequence, in order

    def __init__(out self, var ids: List[Int]):
        # Take ownership of `ids` (transfer with `^` at the call site).
        self.ids = ids^

    def size(self) -> Int:
        # Number of tokens in the dataset.
        return len(self.ids)


struct TrainValSplit(Copyable, Movable):
    # The result of `train_val_split`. A named struct rather than a tuple so the
    # fields read at the call site (`split.train`, `split.val`) and so we do not
    # lean on tuple-destructuring ergonomics.
    var train: TokenDataset
    var val: TokenDataset

    def __init__(out self, var train: TokenDataset, var val: TokenDataset):
        self.train = train^
        self.val = val^


def train_val_split(
    ids: List[Int], val_fraction: Float64
) raises -> TrainValSplit:
    # Split a single contiguous id sequence into a train prefix and a val suffix.
    # `val_fraction` is the fraction of tokens held out for validation.
    #
    # The validation count is `floor(len(ids) * val_fraction)` and the training
    # set is everything before it. Computing the *val* count directly (rather
    # than `floor(len * (1 - val_fraction))`) sidesteps a floating-point trap:
    # `1.0 - 0.1` is `0.8999999999999999`, so the naive form would slice 100 ids
    # as 89/11 instead of the intended 90/10. Same integer result whenever the
    # arithmetic is exact, correct result when it is not.
    #
    # Raises if `val_fraction` is not strictly inside (0, 1), or if the chosen
    # split would leave either side empty (too small a corpus or too extreme a
    # fraction). Allocates two new lists; does not mutate `ids`.
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
