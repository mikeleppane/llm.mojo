# KVCache — the past keys and values a decode step reuses instead of recomputing.
#
# A NAME COLLISION worth reading before the code. This codebase already has
# things called "cache": AttentionCache, BlockCache, GPTCache, and the
# forward_cached family are ACTIVATION caches — they remember a forward pass so
# the training backward can differentiate it, and they are valid only for the one
# forward call that produced them. This struct is a different animal. The KV
# cache remembers the PAST — every position's keys and values — so that
# generating the next token does not recompute the whole sequence. Activation
# caches remember the forward to differentiate it; the KV cache remembers the
# past to avoid recomputing it. Nothing named `forward_cached*` is touched by
# this file, and the decode entry point is a method named `step`, not a
# `forward_cached` variant, to keep the two ideas from blurring.
#
# Why it makes generation fast. Attention is the only place positions interact,
# and a position's K and V depend only on that position's residual stream, which
# never changes as the sequence grows to its right (causal masking guarantees the
# future cannot reach back). So the K/V computed for positions 0..t-1 while
# emitting token t are BIT-IDENTICAL to what a full forward would recompute at
# step t+1. Cache them once, feed only the new token per step, and per-token cost
# drops from O(T · C²·L) — a full forward over T positions — to O(C²·L + T·C·L):
# one position's linear layers plus attention reads over the cache.
#
# Layout (D-decoupled on purpose). Per layer we keep two [capacity, C] buffers,
# one for keys and one for values, storing the FULL fused-qkv k-third and v-third
# rows PRE-head-split — exactly the k_all/v_all intermediates the batch attention
# already materializes before its per-head column slices. The step path does the
# identical slice_cols head split on the cached rows, so bit-parity with the
# batch path is inherited from reusing the same layout and the same primitives,
# not argued from scratch. capacity == the model's context_length; buffers are
# preallocated to zeros so step never reallocates and row indexing is trivial.
# At GPT-2 124M that is 2 × 12 × 1024 × 768 × 8 B ≈ 151 MB — allocated once,
# never grown.

from llm.config import GPTConfig
from llm.tensor.tensor2d import Tensor2D, zeros_2d


@fieldwise_init
struct KVCache(Copyable, Movable):
    # Per-layer key/value buffers plus the shared fill length and the fixed
    # capacity. `k[i]` and `v[i]` are [capacity, C]; rows 0..length-1 hold real
    # cached K/V, rows length..capacity-1 are dead (never read — every step reads
    # only slice_rows(buffer, 0, pos+1)). `length` is shared across all layers
    # because every layer advances in lockstep: one call to GPT.step writes row
    # `length` in every layer, then bumps `length` once.
    var k: List[Tensor2D]  # one [capacity, C] per layer; rows 0..length-1 valid
    var v: List[Tensor2D]  # one [capacity, C] per layer; same validity
    var length: Int  # positions filled so far (shared by all layers)
    var capacity: Int  # max positions == the model's context_length

    @staticmethod
    def fresh(cfg: GPTConfig) raises -> KVCache:
        # Allocate a zeroed cache sized to `cfg`: n_layers pairs of
        # [context_length, d_model] buffers, length 0, capacity context_length.
        # cfg.validate() runs first so a bad shape fails at the edge rather than
        # as a degenerate allocation. Preallocating at capacity means step never
        # reallocates and indexing is a plain row write. Memory note: at 124M
        # this is ~151 MB on top of the ~2 GB model — generate_cached allocates
        # it only AFTER validating the call, so a bad request never pays it.
        # Allocates the 2·n_layers buffers; raises on an invalid config.
        cfg.validate()
        var k = List[Tensor2D]()
        var v = List[Tensor2D]()
        for _ in range(cfg.n_layers):
            k.append(zeros_2d(cfg.context_length, cfg.d_model))  # [T, C]
            v.append(zeros_2d(cfg.context_length, cfg.d_model))  # [T, C]
        return KVCache(k^, v^, 0, cfg.context_length)

    def reset(mut self):
        # Rewind to empty: length = 0, buffers left untouched. The stale rows past
        # length are dead by construction (every read is bounded by length), so
        # zeroing them would be wasted work. Reusing a cache after reset MUST
        # replay bit-identically to a fresh one — a test pins it. Mutates self;
        # allocates nothing; cannot raise.
        self.length = 0

    def check_compatible(self, cfg: GPTConfig) raises:
        # Guard against feeding one model's cache to another: named raises if the
        # layer count, the width C, or the capacity disagree with `cfg`. GPT.step
        # calls this on entry, so a mismatched cache fails loudly at the edge
        # instead of corrupting a row or reading a wrong-width slice. Reads self;
        # allocates nothing.
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
