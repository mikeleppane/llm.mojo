"""KVCache: the past keys and values a decode step reuses instead of recomputing.

Distinct from the activation caches (AttentionCache, BlockCache, GPTCache, the
forward_cached family), which remember a forward pass so backward can
differentiate it. This cache remembers the PAST — every position's K/V — so
generating the next token does not recompute the whole prefix. A position's K/V
depend only on that position's residual stream, which never changes as the
sequence grows rightward (causal masking), so the cached K/V are bit-identical
to what a full forward would recompute. Per-token cost drops from O(T·C^2·L) to
O(C^2·L + T·C·L). The decode entry point is `step`, not a forward_cached variant,
to keep the two ideas apart.

Layout: per layer, two [capacity, C] buffers holding the full fused-qkv k/v
thirds PRE-head-split (the same k_all/v_all the batch path materializes), so the
step path's slice_cols head split inherits bit-parity from reusing the layout.
capacity == the model's context_length; buffers are preallocated to zeros so step
never reallocates (~151 MB at GPT-2 124M, allocated once).
"""

from llm.config import GPTConfig
from llm.tensor.tensor2d import Tensor2D, zeros_2d


@fieldwise_init
struct KVCache(Copyable, Movable):
    """Per-layer key/value buffers with a shared fill length and fixed capacity.

    `k[i]` and `v[i]` are [capacity, C]; rows 0..length-1 hold real cached K/V,
    rows past length are dead (never read). `length` is shared across layers
    because GPT.step writes row `length` in every layer, then bumps it once.
    """

    var k: List[Tensor2D]  # one [capacity, C] per layer; rows 0..length-1 valid
    var v: List[Tensor2D]  # one [capacity, C] per layer; same validity
    var length: Int  # positions filled so far (shared by all layers)
    var capacity: Int  # max positions == the model's context_length

    @staticmethod
    def fresh(cfg: GPTConfig) raises -> KVCache:
        """Allocate a zeroed cache sized to `cfg`.

        n_layers pairs of [context_length, d_model] buffers, length 0.
        cfg.validate() runs first so a bad shape fails at the edge.

        Args:
            cfg: Model config supplying n_layers, context_length, d_model.

        Returns:
            A fresh cache. Allocates the 2*n_layers buffers (~151 MB at 124M).

        Raises:
            Error: If the config is invalid.
        """
        cfg.validate()
        var k = List[Tensor2D]()
        var v = List[Tensor2D]()
        for _ in range(cfg.n_layers):
            k.append(zeros_2d(cfg.context_length, cfg.d_model))  # [T, C]
            v.append(zeros_2d(cfg.context_length, cfg.d_model))  # [T, C]
        return KVCache(k^, v^, 0, cfg.context_length)

    def reset(mut self):
        """Rewind to empty: length = 0, buffers left untouched.

        Rows past length are dead by construction, so zeroing them is wasted
        work; a reset cache replays bit-identically to a fresh one. Mutates
        self; allocates nothing; cannot raise.
        """
        self.length = 0

    def check_compatible(self, cfg: GPTConfig) raises:
        """Guard against feeding one model's cache to another.

        GPT.step calls this on entry so a mismatched cache fails loudly instead
        of corrupting a row or slicing a wrong-width head. Reads self; allocates
        nothing.

        Args:
            cfg: Model config to check the cache against.

        Raises:
            Error: If the layer count, width C, or capacity disagree with cfg.
        """
        if len(self.k) != cfg.n_layers:
            raise Error(
                "KVCache.check_compatible: cache has "
                + String(len(self.k))
                + " key layers but model has "
                + String(cfg.n_layers)
            )
        # Validate the value-buffer count too, BEFORE the loop indexes v[i] — a
        # publicly built cache (@fieldwise_init) could carry mismatched k/v list
        # lengths, and that must surface as this named error, not a bounds trap.
        if len(self.v) != cfg.n_layers:
            raise Error(
                "KVCache.check_compatible: cache has "
                + String(len(self.v))
                + " value layers but model has "
                + String(cfg.n_layers)
            )
        if self.capacity != cfg.context_length:
            raise Error(
                "KVCache.check_compatible: cache capacity "
                + String(self.capacity)
                + " does not match model context_length "
                + String(cfg.context_length)
            )
        # Width is checked per layer against the buffers themselves: a cache built
        # for a different d_model would slice wrong-width heads downstream.
        for i in range(len(self.k)):
            if self.k[i].cols != cfg.d_model:
                raise Error(
                    "KVCache.check_compatible: layer "
                    + String(i)
                    + " key width "
                    + String(self.k[i].cols)
                    + " does not match model d_model "
                    + String(cfg.d_model)
                )
            if self.v[i].cols != cfg.d_model:
                raise Error(
                    "KVCache.check_compatible: layer "
                    + String(i)
                    + " value width "
                    + String(self.v[i].cols)
                    + " does not match model d_model "
                    + String(cfg.d_model)
                )
            if (
                self.k[i].rows != self.capacity
                or self.v[i].rows != self.capacity
            ):
                raise Error(
                    "KVCache.check_compatible: layer "
                    + String(i)
                    + " buffer height does not match capacity "
                    + String(self.capacity)
                )
