# A batch of token ids: two flat [B, T] arrays, inputs and targets.
#
# A `TokenBatch` is what the training loop consumes each step: B sequences of
# length T for the model to read (`inputs`) and, aligned position-for-position,
# the next token the model should predict (`targets`). The shift-by-one that
# links them is baked in by whoever builds the batch (the loader), not by this
# struct — here they are just two parallel arrays.
#
# Layout is flat row-major: element (b, t) lives at `b * seq_len + t`. Flat
# storage (rather than a `List[List[Int]]`) mirrors how the model-parameter layer
# will index tokens and keeps the memory contiguous. Ids are integers, so this
# lives in `data/` and depends on nothing float-valued — the tensor layer is a
# separate concern.


struct TokenBatch(Copyable, Movable):
    # inputs and targets are each a flat row-major [B, T] array; element (b, t)
    # is at index b * seq_len + t.
    var inputs: List[Int]
    var targets: List[Int]
    var batch_size: Int  # B: number of sequences
    var seq_len: Int  # T: tokens per sequence

    def __init__(
        out self,
        var inputs: List[Int],
        var targets: List[Int],
        batch_size: Int,
        seq_len: Int,
    ) raises:
        # Take ownership of both arrays. Raises unless each flat array has exactly
        # batch_size * seq_len elements, so a mis-shaped batch is caught at
        # construction rather than as a bad read later.
        var expected = batch_size * seq_len
        if len(inputs) != expected:
            raise Error(
                "TokenBatch: inputs length "
                + String(len(inputs))
                + " != batch_size * seq_len = "
                + String(expected)
            )
        if len(targets) != expected:
            raise Error(
                "TokenBatch: targets length "
                + String(len(targets))
                + " != batch_size * seq_len = "
                + String(expected)
            )
        self.inputs = inputs^
        self.targets = targets^
        self.batch_size = batch_size
        self.seq_len = seq_len

    def _flat_index(self, b: Int, t: Int) raises -> Int:
        # Bounds-checked row-major index for coordinate (b, t). Raises if either
        # index is outside its dimension — the single place both accessors funnel
        # through, so the check can't drift between inputs and targets.
        if b < 0 or b >= self.batch_size:
            raise Error(
                "TokenBatch: batch index "
                + String(b)
                + " out of range [0, "
                + String(self.batch_size)
                + ")"
            )
        if t < 0 or t >= self.seq_len:
            raise Error(
                "TokenBatch: sequence index "
                + String(t)
                + " out of range [0, "
                + String(self.seq_len)
                + ")"
            )
        return b * self.seq_len + t

    def input_at(self, b: Int, t: Int) raises -> Int:
        # The input token at (b, t). Bounds-checked.
        return self.inputs[self._flat_index(b, t)]

    def target_at(self, b: Int, t: Int) raises -> Int:
        # The target token at (b, t). Bounds-checked.
        return self.targets[self._flat_index(b, t)]
