"""A batch of token ids: two flat [B, T] arrays, inputs and targets.

A `TokenBatch` is what the training loop consumes each step: B sequences of
length T for the model to read (`inputs`) and, aligned position-for-position,
the next token to predict (`targets`). The shift-by-one that links them is baked
in by the loader, not here. Layout is flat row-major: element (b, t) lives at
`b * seq_len + t`, keeping the memory contiguous.
"""


struct TokenBatch(Copyable, Movable):
    """A batch of B*T ids: inputs and targets, each a flat row-major [B, T]."""

    var inputs: List[Int]  # flat [B, T]; element (b, t) at b * seq_len + t
    var targets: List[Int]  # flat [B, T]; the inputs shifted left by one
    var batch_size: Int  # B: number of sequences
    var seq_len: Int  # T: tokens per sequence

    def __init__(
        out self,
        var inputs: List[Int],
        var targets: List[Int],
        batch_size: Int,
        seq_len: Int,
    ) raises:
        """Take ownership of both arrays after validating the shape.

        A mis-shaped batch is caught at construction rather than as a bad read
        later. The positivity check matters on its own: a negative batch_size
        and seq_len can multiply to the right length and otherwise slip through.

        Args:
            inputs: Flat [B, T] input ids (ownership transferred).
            targets: Flat [B, T] target ids (ownership transferred).
            batch_size: B, must be >= 1.
            seq_len: T, must be >= 1.

        Raises:
            Error: If a dimension is < 1, or an array length is not
                batch_size * seq_len.
        """
        if batch_size < 1:
            raise Error(
                "TokenBatch: batch_size must be >= 1, got " + String(batch_size)
            )
        if seq_len < 1:
            raise Error(
                "TokenBatch: seq_len must be >= 1, got " + String(seq_len)
            )
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
        """Bounds-checked row-major index for coordinate (b, t).

        The single place both accessors funnel through, so the check can't drift
        between inputs and targets.

        Args:
            b: Batch index, in [0, batch_size).
            t: Sequence index, in [0, seq_len).

        Returns:
            The flat index b * seq_len + t.

        Raises:
            Error: If either index is out of range.
        """
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
        """Return the input token at (b, t), bounds-checked.

        Args:
            b: Batch index.
            t: Sequence index.

        Returns:
            The input token id.

        Raises:
            Error: If (b, t) is out of range.
        """
        return self.inputs[self._flat_index(b, t)]

    def target_at(self, b: Int, t: Int) raises -> Int:
        """Return the target token at (b, t), bounds-checked.

        Args:
            b: Batch index.
            t: Sequence index.

        Returns:
            The target token id.

        Raises:
            Error: If (b, t) is out of range.
        """
        return self.targets[self._flat_index(b, t)]
