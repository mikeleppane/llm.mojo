# Embedding — a [V, C] lookup table gathering rows by integer id.
#
# One struct serves both roles GPT-2 needs: token embeddings (ids are token ids)
# and positional embeddings (ids are positions 0..T-1). There is no separate
# positional type — the caller chooses which ids to pass. The table is a public
# Parameter on purpose: a later part ties the language-model head to the token
# table (the head reuses this exact matrix), and that tying needs the table
# reachable from outside.

from llm.nn.linear import GPT2_INIT_STD
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.utils.random import Rng


@fieldwise_init
struct Embedding(Copyable, Movable):
    var table: Parameter  # [V, C]; public so the LM head can tie to it

    @staticmethod
    def init_random(
        mut rng: Rng, vocab_size: Int, d_model: Int
    ) raises -> Embedding:
        # An Embedding whose table is drawn from normal(0, 0.02) (GPT-2's
        # scheme). Mutates rng (advances its state); allocates the table;
        # deterministic given the generator's state. Raises on non-positive
        # sizes, which would produce a degenerate shape.
        if vocab_size <= 0 or d_model <= 0:
            raise Error(
                "Embedding.init_random: vocab_size and d_model must be"
                " positive, got "
                + String(vocab_size)
                + " and "
                + String(d_model)
            )
        var t = zeros_2d(vocab_size, d_model)  # [V, C]
        for r in range(vocab_size):
            for c in range(d_model):
                t[r, c] = rng.normal(0.0, GPT2_INIT_STD)
        return Embedding(Parameter(t^))

    def forward(self, ids: List[Int]) raises -> Tensor2D:
        # [N] ids -> [N, C] gathered rows, in the order the ids are given. Reads
        # self only; allocates the result. Raises if any id is < 0 or >= V,
        # before writing that row — a bad id fails loudly instead of reading a
        # neighbouring row or running off the buffer.
        var n = len(ids)
        var c = self.table.value.cols
        var vocab_size = self.table.value.rows
        var out = zeros_2d(n, c)
        for i in range(n):
            var idx = ids[i]
            if idx < 0 or idx >= vocab_size:
                raise Error(
                    "Embedding.forward: id out of range, got "
                    + String(idx)
                    + " for vocab size "
                    + String(vocab_size)
                )
            for j in range(c):
                out[i, j] = self.table.value[idx, j]
        return out^
