"""Embedding — a [V, C] lookup table gathering rows by integer id.

One struct serves both roles GPT-2 needs: token embeddings (ids are token ids)
and positional embeddings (ids are positions 0..T-1); the caller chooses which
ids to pass. The table is a public Parameter so the language-model head can tie
to it (reuse this exact matrix).
"""

from llm.nn.linear import GPT2_INIT_STD
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.utils.random import Rng


@fieldwise_init
struct EmbeddingCache(Copyable, Movable):
    """The ids the forward gathered — all backward needs to scatter the gradient.

    Valid only for the forward call that produced it.
    """

    var ids: List[Int]  # [N]


@fieldwise_init
struct EmbeddingForward(Copyable, Movable):
    """Output of forward_cached plus the cache its backward consumes."""

    var output: Tensor2D  # [N, C]
    var cache: EmbeddingCache

    def split(deinit self, mut cache_slot: EmbeddingCache) -> Tensor2D:
        """Consume this forward: move the cache into `cache_slot`, return the output.

        Lets an assembly site take both pieces by move instead of copying each
        out of a live struct (a field cannot be transferred with `^`).

        Args:
            cache_slot: Receives the cache by move.

        Returns:
            The output tensor, shape [N, C], by move.
        """
        cache_slot = self.cache^
        return self.output^


@fieldwise_init
struct Embedding(Copyable, Movable):
    """A [V, C] lookup table gathering rows by integer id."""

    var table: Parameter  # [V, C]; public so the LM head can tie to it

    @staticmethod
    def init_random(
        mut rng: Rng, vocab_size: Int, d_model: Int
    ) raises -> Embedding:
        """Build an Embedding whose table is drawn from normal(0, 0.02) (GPT-2's scheme).

        Args:
            rng: Random generator, advanced as the table is drawn.
            vocab_size: Number of rows V.
            d_model: Number of columns C.

        Returns:
            An Embedding with table [V, C]. Allocates; deterministic given rng's
            state.

        Raises:
            Error: If vocab_size or d_model is non-positive.
        """
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
        """Gather table rows by id, in the order the ids are given.

        Args:
            ids: Integer ids, shape [N].

        Returns:
            Gathered rows, shape [N, C]. Allocates; reads self only.

        Raises:
            Error: If any id is < 0 or >= V (checked before writing that row).
        """
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
        """Gather rows as forward does, additionally caching the ids for backward.

        Takes ids by value and moves them into the cache after the gather (no
        copy) — the caller hands over `ids^` or `ids.copy()`.

        Args:
            ids: Integer ids, shape [N]; moved into the cache.

        Returns:
            An EmbeddingForward with output [N, C] and the ids cache. Allocates
            the output; cache valid only for this call.

        Raises:
            Error: If any id is out of range (via forward).
        """
        var output = self.forward(ids)
        return EmbeddingForward(output^, EmbeddingCache(ids^))

    def backward(mut self, cache: EmbeddingCache, d_out: Tensor2D) raises:
        """Scatter the upstream gradient back to the gathered table rows.

        Reverse of `output[i, :] = table[ids[i], :]`, so
        `table.grad[ids[i], :] += d_out[i, :]`. The += is essential: when an id
        repeats, each occurrence contributes and those gradients must sum into
        that one row. Ids are integer indices, not differentiable, so there is no
        input gradient and nothing is returned.

        A repeated id's cotangent rows are first summed into a per-id local, then
        added to table.grad once, so two backward passes double the grad
        bit-for-bit rather than interleaving partial sums. Scratch is sized to the
        distinct touched ids, not the whole vocabulary.

        Args:
            cache: The ids the paired forward gathered.
            d_out: Upstream gradient, shape [N, C].

        Mutates self.table.grad; allocates a small per-distinct-id scratch.

        Raises:
            Error: On a length/width mismatch or an out-of-range id (guarded
                before writing, like forward).
        """
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
