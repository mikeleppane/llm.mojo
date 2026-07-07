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

    def split(deinit self, mut cache_slot: EmbeddingCache) -> Tensor2D:
        # Consume this forward: the cache moves into the caller's slot and the
        # output is returned. Lets an assembly site take both pieces by move
        # instead of copying each out of a live struct (a field cannot be
        # transferred with `^`).
        cache_slot = self.cache^
        return self.output^


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

    def forward_cached(self, var ids: List[Int]) raises -> EmbeddingForward:
        # [N] ids -> EmbeddingForward: the same gathered rows as forward, plus the
        # ids cached for backward. Takes ids by value and MOVES them into the
        # cache after the gather (no copy) — the caller hands over `ids^` when the
        # list is dead or `ids.copy()` when it still needs it. Reads self;
        # allocates the output; raises on any out-of-range id (via forward). The
        # cache is valid only for this call.
        var output = self.forward(ids)
        return EmbeddingForward(output^, EmbeddingCache(ids^))

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
        # self.table.grad; allocates a small per-distinct-id scratch; raises on a
        # length/width mismatch or an out-of-range id (guarded before writing,
        # like forward).
        #
        # Accumulation order: a repeated id's several cotangent rows are first
        # summed into a per-id local (`sums`), then that finished row delta is
        # added to table.grad ONCE. Adding one fully-formed delta per call — not a
        # running += into table.grad inside the loop — is what makes two backward
        # passes double the grad bit-for-bit (the second call's partial sums would
        # otherwise interleave with the first call's stored result and round
        # differently than 2*grad1). Same discipline as LayerNorm's dγ/dβ. The
        # scratch is sized to the DISTINCT touched ids (via a linear scan over the
        # usually-short id list), not the whole vocabulary, so untouched rows cost
        # nothing.
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
        var uniq_ids = List[Int]()  # distinct touched ids
        var sums = List[Float64]()  # flat [len(uniq_ids), c] row-major
        for i in range(n):
            var idx = cache.ids[i]
            if idx < 0 or idx >= vocab_size:
                raise Error(
                    "Embedding.backward: id out of range, got "
                    + String(idx)
                    + " for vocab size "
                    + String(vocab_size)
                )
            var pos = -1
            for s in range(len(uniq_ids)):
                if uniq_ids[s] == idx:
                    pos = s
                    break
            if pos == -1:
                uniq_ids.append(idx)
                for j in range(c):
                    sums.append(d_out[i, j])
            else:
                var base = pos * c
                for j in range(c):
                    sums[base + j] = sums[base + j] + d_out[i, j]
        for s in range(len(uniq_ids)):
            var idx = uniq_ids[s]
            var base = s * c
            for j in range(c):
                self.table.grad[idx, j] = (
                    self.table.grad[idx, j] + sums[base + j]
                )
