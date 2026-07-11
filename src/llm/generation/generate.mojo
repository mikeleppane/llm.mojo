# The autoregressive generation loop.
#
# This is where the model finally speaks: turn the last logits row into a next
# token, append it, and repeat, cropping the context to the model's window and
# stopping on request. There is NO new math here — GPT.forward, the temperature
# softmax, argmax, and the inverse-CDF sampler all exist and are tested. Part XV
# is pure assembly over them.
#
# generate() is ids-in / ids-out and binds to NOTHING but the GPT struct: no
# tokenizer, no config beyond what GPT already carries, no checkpoint knowledge.
# That is the forward contract for the BPE part, which reuses this verbatim with
# real GPT-2 weights and END_OF_TEXT_ID as the stop token.
#
# generate() is deliberately uncached: every step recomputes the FULL forward
# over all T positions, which is O(T^2 * steps). That recompute is the honest
# baseline, and it is what the KV cache removes — see generate_cached below, which
# caches each layer's keys and values and feeds only the new token per step,
# producing token-for-token identical output at a fraction of the per-token cost.
# generate() stays as the reference oracle those two paths are proven equal
# against, so its code is untouched.

from llm.generation.sampler import SamplerConfig, sample_next
from llm.transformer.gpt import GPT
from llm.transformer.kv_cache import KVCache
from llm.utils.random import Rng


def _contains(values: List[Int], target: Int) -> Bool:
    # Linear-scan membership. Stop lists hold 0-3 entries in practice, so no set
    # structure is warranted. Reads only; allocates nothing.
    for i in range(len(values)):
        if values[i] == target:
            return True
    return False


def generate(
    gpt: GPT,
    prompt: List[Int],
    max_new_tokens: Int,
    cfg: SamplerConfig,
    stop_tokens: List[Int],
    mut rng: Rng,
) raises -> List[Int]:
    # Autoregressively extend `prompt` by up to `max_new_tokens` tokens under the
    # policy `cfg`, halting early if an emitted token is in `stop_tokens`.
    #
    # Returns ONLY the newly generated tokens (the caller owns the prompt; the
    # full text is decode(prompt + returned)). Reads gpt; allocates the output and
    # per-step intermediates; mutates rng.
    #
    # rng-draw count: zero if cfg is greedy (temperature 0), otherwise exactly one
    # per emitted token — so a greedy call leaves rng.state bit-untouched and can
    # be dropped into a seeded pipeline without perturbing it.
    #
    # Raises (named): an empty prompt (the model has no BOS convention — seed
    # generation with at least one real token); a negative max_new_tokens; or an
    # invalid cfg. max_new_tokens == 0 is a valid no-op returning an empty list.
    #
    # Per step: crop the running sequence to its LAST min(len, context_length)
    # tokens (the sliding window — without it, step context_length + 1 would raise
    # on the positional-embedding bounds), run the full inference forward, take the
    # last logits row, sample the next id, append it, and check for a stop token.
    #
    # Stop semantics: the triggering token IS appended, THEN the loop halts, so the
    # output self-documents why it stopped (its last element is the stop token);
    # callers that don't want it strip one element. A stop id occurring in the
    # PROMPT never halts anything — only emitted tokens are checked. An empty
    # stop_tokens list never stops early.
    if len(prompt) == 0:
        raise Error(
            "generate: empty prompt — the model has no BOS token, so seed"
            " generation with at least one real token"
        )
    if max_new_tokens < 0:
        raise Error(
            "generate: max_new_tokens must be non-negative, got "
            + String(max_new_tokens)
        )
    cfg.validate()

    var context_length = gpt.cfg.context_length
    var seq = prompt.copy()  # prompt + everything emitted so far
    var emitted = List[Int]()

    for _ in range(max_new_tokens):
        # Sliding-window crop: the last min(len(seq), context_length) tokens.
        var start = len(seq) - context_length
        if start < 0:
            start = 0
        var context = List[Int]()
        for i in range(start, len(seq)):
            context.append(seq[i])

        var logits = gpt.forward(context)  # [T', V]
        # The next token is conditioned on the whole prefix, so only the final
        # row matters. Borrow it as a view [V] rather than copying — logits is
        # read, not mutated, for the rest of the step, so the borrow is safe.
        var next_id = sample_next(logits.row(logits.rows - 1), cfg, rng)

        emitted.append(next_id)
        seq.append(next_id)
        if _contains(stop_tokens, next_id):
            break  # append-then-halt: the output records why it stopped

    return emitted^


def generate_cached(
    gpt: GPT,
    prompt: List[Int],
    max_new_tokens: Int,
    cfg: SamplerConfig,
    stop_tokens: List[Int],
    mut rng: Rng,
) raises -> List[Int]:
    # The KV-cached twin of generate: SAME contract, same output, a fraction of
    # the per-token cost. It returns ONLY the new tokens; stop semantics are
    # append-then-halt; a stop id in the PROMPT never halts; an empty stop list
    # never stops; an empty prompt raises; a negative max_new_tokens raises; 0 is
    # a no-op returning an empty list with rng bit-untouched; greedy draws zero
    # rng, sampled draws exactly one per emitted token via the SAME sample_next.
    # Within the validated length domain it is token-for-token identical to
    # generate AND leaves rng.state identical (stream parity), so generate remains
    # the oracle this is proven against.
    #
    # The KVCache is an internal detail: allocated here AFTER validation (so a bad
    # call never pays the ~151 MB) and never escaping — callers should not have to
    # manage inference plumbing. (KVCache and GPT.step stay public for the tests.)
    #
    # OVERFLOW POLICY — up-front named raise if len(prompt) + max_new_tokens >
    # context_length. generate slides a window over its context; the cached path
    # CANNOT slide cheaply, and this is a teaching point rather than a limitation
    # to paper over. GPT-2's positions are ABSOLUTE learned embeddings, so evicting
    # the oldest token shifts every survivor's position and invalidates EVERY
    # cached K/V row — a "sliding" cached generation would have to re-prime the
    # whole window every step, i.e. the uncached algorithm wearing a cache costume.
    # Within the validated domain neither path's window ever slides, so the two
    # see identical contexts and token-for-token parity is well-defined.
    #
    # Flow: validate -> KVCache.fresh -> PRIME the prompt token-by-token through
    # gpt.step (discard all but the last logits row) -> LOOP: sample the last row,
    # append, check stop, and step the emitted token to advance the cache. Priming
    # token-by-token is the same cost order as the ONE batch forward the uncached
    # path spends on the prompt, and routing priming through the same `step` keeps
    # a single code path so parity covers it too. Reads gpt; allocates the cache,
    # the output, and per-step intermediates; mutates rng (sampled path only).
    if len(prompt) == 0:
        raise Error(
            "generate_cached: empty prompt — the model has no BOS token, so"
            " seed generation with at least one real token"
        )
    if max_new_tokens < 0:
        raise Error(
            "generate_cached: max_new_tokens must be non-negative, got "
            + String(max_new_tokens)
        )
    cfg.validate()

    var context_length = gpt.cfg.context_length
    if len(prompt) + max_new_tokens > context_length:
        raise Error(
            "generate_cached: len(prompt) + max_new_tokens = "
            + String(len(prompt))
            + " + "
            + String(max_new_tokens)
            + " exceeds context_length "
            + String(context_length)
            + " — the cached path cannot slide the window (GPT-2's"
            " positions are"
            " absolute), so shorten the prompt or the budget"
        )

    var emitted = List[Int]()
    if max_new_tokens == 0:
        return emitted^  # valid no-op, rng bit-untouched, cache never allocated

    # Validation passed and there is real work to do — now pay for the cache.
    var cache = KVCache.fresh(gpt.cfg)

    # Prime: feed prompt[0 .. n-2] discarding their logits, then prompt[n-1] and
    # keep its logits row — the distribution over the first token to emit.
    var n = len(prompt)
    for i in range(n - 1):
        _ = gpt.step(prompt[i], cache)
    var logits = gpt.step(prompt[n - 1], cache)  # [1, V]

    for step_i in range(max_new_tokens):
        # The next token is conditioned on the whole prefix; the cached step
        # returns a [1, V] tensor, so the last (only) row IS that distribution —
        # the same borrow-a-row-view sample_next generate passes.
        var next_id = sample_next(logits.row(logits.rows - 1), cfg, rng)
        emitted.append(next_id)
        if _contains(stop_tokens, next_id):
            break  # append-then-halt: the output records why it stopped
        if step_i < max_new_tokens - 1:
            # Advance the cache with the emitted token to get the next row. Skipped
            # on the final iteration (no further token is sampled from it), which
            # is also what keeps the position count within the validated capacity.
            logits = gpt.step(next_id, cache)

    return emitted^
