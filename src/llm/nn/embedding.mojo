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
struct EmbeddingCache(Copyable, Movable):
    # The ids the forward gathered — the only thing backward needs to scatter the
    # upstream gradient back to the right table rows. Valid only for the forward
    # call that produced it.
    var ids: List[Int]  # [N]


@fieldwise_init
struct EmbeddingForward(Copyable, Movable):
    # forward_cached's output plus the cache its backward consumes.
    var output: Tensor2D  # [N, C]
    var cache: EmbeddingCache


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

    def forward_cached(self, ids: List[Int]) raises -> EmbeddingForward:
        # [N] ids -> EmbeddingForward: the same gathered rows as forward, plus the
        # ids cached for backward. Reads self; allocates the output and a copy of
        # the ids; raises on any out-of-range id (via forward). The cache is valid
        # only for this call.
        var output = self.forward(ids)
        return EmbeddingForward(output^, EmbeddingCache(ids.copy()))

    def backward(mut self, cache: EmbeddingCache, d_out: Tensor2D) raises:
        # Reverse of the gather output[i, :] = table[ids[i], :]. The forward copies
        # table row ids[i] into output row i, so the gradient scatters back the
        # other way:
        #     table.grad[ids[i], :] += d_out[i, :].
        # The += (not =) is essential: when an id appears at several positions each
        # occurrence contributes, and those gradients must SUM into that one row —
        # the classic scatter bug is to overwrite and keep only the last. There is
        # NO gradient with respect to the ids: they are integer indices selecting
        # rows, not differentiable inputs, so backward returns nothing. Mutates
        # self.table.grad; allocates nothing; raises on a length/width mismatch or
        # an out-of-range id (guarded before writing, like forward).
        var n = len(cache.ids)
        var c = self.table.value.cols
        var vocab_size = self.table.value.rows
        if d_out.rows != n:
            raise Error(
                "Embedding.backward: d_out rows "
                + String(d_out.rows)
                + " must equal ids length "
                + String(n)
            )
        if d_out.cols != c:
            raise Error(
                "Embedding.backward: d_out width "
                + String(d_out.cols)
                + " must equal table width "
                + String(c)
            )
        for i in range(n):
            var idx = cache.ids[i]
            if idx < 0 or idx >= vocab_size:
                raise Error(
                    "Embedding.backward: id out of range, got "
                    + String(idx)
                    + " for vocab size "
                    + String(vocab_size)
                )
            for j in range(c):
                self.table.grad[idx, j] = self.table.grad[idx, j] + d_out[i, j]
