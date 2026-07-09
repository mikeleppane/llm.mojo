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
# Deliberately uncached: every step recomputes the FULL forward over all T
# positions, which is O(T^2 * steps). That waste is the motivation for the KV
# cache two parts on; here correctness comes first and the recompute is honest.

from llm.generation.sampler import SamplerConfig, sample_next
from llm.transformer.gpt import GPT
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
