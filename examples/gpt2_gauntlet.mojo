"""The Tier 2 validation gauntlet: does our 124M port still match the reference?

Where examples/gpt2_parity_check.mojo checks ONE prompt, the gauntlet checks a
curated set spanning the input space (unicode, code, punctuation, the 1024-token
context boundary — see data/gauntlet/prompts.txt), so a bug that only manifests on
real token statistics has somewhere to show up: a BPE edge case on adversarial
text, a numerical drift a doll-house tensor never accumulates, an off-by-one at
position 1023, a seam between tokenizer -> loader -> forward -> sampler that unit
tests each cover only in isolation.

It loads the model ONCE, then per prompt:
  (a) the Mojo tokenizer's ids must equal the golden ids EXACTLY — tokenization is
      discrete, so there is no tolerance;
  (b) the full forward's final-row logits must match the golden probes at 1e-6
      (the float64-vs-float64 bar), and the argmax and top-5 ids EXACTLY;
  (c) the mean next-token NLL on the prompt's own tokens must match at 1e-6 (the
      whole-model-in-one-scalar drift detector);
  (d) for a designated short subset, uncached generate and generate_cached must
      agree token-for-token (our-vs-our — the KV-cache parity claim on real text).

The parity discipline: cross-implementation checks stay at LOGIT level with a
tolerance; token-SEQUENCE exactness is only ever our-vs-our. A cross-implementation
greedy-text golden would flake on genuine near-ties; logits at 1e-6 plus exact
probe argmax ids give the same protection without that flake channel.

GOLDEN LIFECYCLE — read before re-pinning anything. A red gauntlet after a code
change indicts THE CHANGE, not the goldens. Re-pinning data/gauntlet/goldens.txt
is legitimate only with documented evidence in the part's notes: either the oracle
side changed (a new .bin — visible in the goldens' sha256 header — or a converter
fix) or a near-tie logit delta at ~1e-13 scale is shown. "The new number looks
close enough" is not evidence. Goldens regenerate ONLY via
scripts/gpt2_gauntlet_reference.py, never by hand.

Run (after the weights exist — see scripts/convert_gpt2_weights.py):
    pixi run gauntlet
    pixi run mojo run -I build examples/gpt2_gauntlet.mojo
"""

from std.collections.string import atof

from llm.generation.generate import generate, generate_cached
from llm.generation.sampler import SamplerConfig
from llm.tensor.tensor2d import Tensor2D
from llm.tokenizer.gpt2 import GPT2Tokenizer
from llm.transformer.gpt import GPT
from llm.transformer.gpt2_weights import load_gpt2
from llm.utils.random import Rng

comptime WEIGHTS_PATH = "checkpoints/gpt2-124m.bin"
comptime VOCAB_PATH = "data/gpt2/vocab.json"
comptime MERGES_PATH = "data/gpt2/merges.txt"
comptime PROMPTS_PATH = "data/gauntlet/prompts.txt"
comptime GOLDENS_PATH = "data/gauntlet/goldens.txt"

# 1e-6 is the float64-vs-float64 bar the single-prompt gate established: both sides
# read the same .bin bytes and compute in float64, so the only gap is kernel
# reassociation. Tokenization, argmax, and top-5 are DISCRETE — checked exactly.
comptime PROBE_TOL = 1e-6
comptime NLL_TOL = 1e-6
comptime SEPARATOR_PREFIX = "=== id: "

# The generate-vs-generate_cached agreement check (d) runs only on this short
# subset: the uncached path recomputes the whole forward every step, so it is kept
# to a few short prompts and a small budget. It is our-vs-our — no golden needed.
comptime GREEDY_NEW_TOKENS = 8


struct PromptRecord(Movable):
    """One parsed prompt: its id and its verbatim text."""

    var name: String
    var text: String

    def __init__(out self, var name: String, var text: String):
        self.name = name^
        self.text = text^


struct Golden(Movable):
    """One parsed golden block: the frozen reference for one prompt id."""

    var name: String
    var tokens: List[Int]
    var argmax: Int
    var top5: List[Int]
    var probe_idx: List[Int]
    var probe_val: List[Float64]
    var has_nll: Bool
    var nll: Float64

    def __init__(
        out self,
        var name: String,
        var tokens: List[Int],
        argmax: Int,
        var top5: List[Int],
        var probe_idx: List[Int],
        var probe_val: List[Float64],
        has_nll: Bool,
        nll: Float64,
    ):
        self.name = name^
        self.tokens = tokens^
        self.argmax = argmax
        self.top5 = top5^
        self.probe_idx = probe_idx^
        self.probe_val = probe_val^
        self.has_nll = has_nll
        self.nll = nll


def _require_file(path: String) raises:
    """Raise a converter-pointing error unless `path` exists.

    Cheap existence probe (open, do not read the 498 MB payload); there is no
    fallback to random weights — the gauntlet validates the REAL model.

    Args:
        path: File that must exist.

    Raises:
        Error: If the file cannot be opened.
    """
    try:
        var probe = open(path, "r")
        probe.close()
    except:
        raise Error(
            "missing '"
            + path
            + "' — download the HF gpt2 model.safetensors and run"
            " scripts/convert_gpt2_weights.py to produce it (see that script's"
            " header). No random-weight fallback: the gauntlet needs the real"
            " weights."
        )


def _split_lines(content: String) raises -> List[String]:
    """Split on '\\n' ONLY and drop the single file-terminating empty line.

    The canonical line model shared with scripts/gpt2_gauntlet_reference.py: never
    a unicode-aware splitlines (which would treat CJK or exotic codepoints as line
    breaks), and the trailing '' produced by the file's final newline is dropped so
    both parsers see identical records.

    Args:
        content: The whole file as one String.

    Returns:
        The lines, without newline terminators. Allocates.
    """
    var raw = content.split("\n")
    var lines = List[String]()
    for i in range(len(raw)):
        lines.append(String(raw[i]))
    if len(lines) != 0 and lines[len(lines) - 1] == "":
        _ = (
            lines.pop()
        )  # drop the mandatory file-terminating newline's empty tail
    return lines^


def _record_id(line: String) raises -> String:
    """Extract `<name>` from a `=== id: <name> === [rationale]` separator line.
    """
    var rest = String(line.removeprefix(SEPARATOR_PREFIX))
    var parts = rest.split(" ===")
    if len(parts) < 2:
        raise Error("gauntlet: malformed separator line '" + line + "'")
    return String(parts[0])


def parse_prompts(content: String) raises -> List[PromptRecord]:
    """Parse prompts.txt into (id, text) records (see the file's own header)."""
    var lines = _split_lines(content)
    var records = List[PromptRecord]()
    var have = False
    var name = String("")
    var body = List[String]()
    for i in range(len(lines)):
        var line = lines[i]
        if line.startswith(SEPARATOR_PREFIX):
            if have:
                records.append(PromptRecord(name, "\n".join(body)))
            name = _record_id(line)
            body = List[String]()
            have = True
        elif have:
            body.append(line)
        # lines before the first separator are the header comment — ignored.
    if have:
        records.append(PromptRecord(name, "\n".join(body)))
    return records^


def _ints_after(line: String, prefix: String) raises -> List[Int]:
    """Parse the space-separated ints following `prefix` on `line`."""
    var rest = String(line.removeprefix(prefix))
    var out = List[Int]()
    var parts = rest.split(" ")
    for i in range(len(parts)):
        var tok = String(parts[i])
        if tok != "":
            out.append(Int(tok))
    return out^


def parse_goldens(content: String) raises -> List[Golden]:
    """Parse goldens.txt blocks into Golden records (see that file's own header).
    """
    var lines = _split_lines(content)
    var out = List[Golden]()
    var have = False
    var name = String("")
    var tokens = List[Int]()
    var argmax = 0
    var top5 = List[Int]()
    var probe_idx = List[Int]()
    var probe_val = List[Float64]()
    var has_nll = False
    var nll = 0.0

    for i in range(len(lines)):
        var line = lines[i]
        if line.startswith(SEPARATOR_PREFIX):
            if have:
                out.append(
                    Golden(
                        name.copy(),
                        tokens.copy(),
                        argmax,
                        top5.copy(),
                        probe_idx.copy(),
                        probe_val.copy(),
                        has_nll,
                        nll,
                    )
                )
            name = _record_id(line)
            tokens = List[Int]()
            argmax = 0
            top5 = List[Int]()
            probe_idx = List[Int]()
            probe_val = List[Float64]()
            has_nll = False
            nll = 0.0
            have = True
        elif line.startswith("tokens: "):
            tokens = _ints_after(line, "tokens: ")
        elif line.startswith("argmax: "):
            argmax = Int(String(line.removeprefix("argmax: ")))
        elif line.startswith("top5: "):
            top5 = _ints_after(line, "top5: ")
        elif line.startswith("probe: "):
            var rest = String(line.removeprefix("probe: "))
            var pairs = rest.split(" ")
            for j in range(len(pairs)):
                var pair = String(pairs[j])
                if pair == "":
                    continue
                var kv = pair.split(":")
                if len(kv) != 2:
                    raise Error("gauntlet: malformed probe pair '" + pair + "'")
                probe_idx.append(Int(String(kv[0])))
                probe_val.append(atof(String(kv[1])))
        elif line.startswith("nll: "):
            var v = String(line.removeprefix("nll: "))
            if v == "none":
                has_nll = False
            else:
                has_nll = True
                nll = atof(v)
        # blank / unknown lines within a block are ignored.
    if have:
        out.append(
            Golden(
                name.copy(),
                tokens.copy(),
                argmax,
                top5.copy(),
                probe_idx.copy(),
                probe_val.copy(),
                has_nll,
                nll,
            )
        )
    return out^


def _abs(x: Float64) -> Float64:
    return -x if x < 0.0 else x


def _argmax_row(row: Span[Float64, _]) -> Int:
    """Column index of the max in one logit row (first on a tie)."""
    var best = 0
    var best_v = row[0]
    for c in range(1, len(row)):
        if row[c] > best_v:
            best_v = row[c]
            best = c
    return best


def _top_k_ids(row: Span[Float64, _], k: Int) -> List[Int]:
    """The k highest-logit ids, highest first (first index wins a tie).

    Repeated max-selection excluding already-picked ids — k is tiny (5), so the
    O(k*V) scan is trivial and avoids sorting the whole row.
    """
    var picked = List[Int]()
    for _ in range(k):
        var best = -1
        var best_v = 0.0
        for c in range(len(row)):
            var already = False
            for p in range(len(picked)):
                if picked[p] == c:
                    already = True
                    break
            if already:
                continue
            if best == -1 or row[c] > best_v:
                best_v = row[c]
                best = c
        picked.append(best)
    return picked^


def _fail(prompt_id: String, check: String, detail: String) raises:
    """Raise a gauntlet failure that names the prompt, the check, and the value.
    """
    raise Error("GAUNTLET FAILED [" + prompt_id + "] " + check + ": " + detail)


def _greedy_subset(name: String) -> Bool:
    """Which prompts run the uncached-vs-cached agreement check (short only)."""
    return (
        name == "short-english"
        or name == "contractions"
        or name == "single-token"
    )


def _check_prompt(
    gpt: GPT, tok: GPT2Tokenizer, prompt: PromptRecord, golden: Golden
) raises -> Bool:
    """Run every check for one prompt; return True if the greedy subset ran.

    Raises on the first failure, naming the prompt id, the check, and the offending
    value, so the caller's non-zero exit is self-documenting.
    """
    if prompt.name != golden.name:
        _fail(
            prompt.name,
            "pairing",
            "prompt paired with golden '" + golden.name + "'",
        )

    # (a) Tokenization — EXACT (discrete, no tolerance).
    var ids = tok.encode(prompt.text)
    if len(ids) != len(golden.tokens):
        _fail(
            prompt.name,
            "tokens",
            "encoded "
            + String(len(ids))
            + " ids, golden has "
            + String(len(golden.tokens)),
        )
    for i in range(len(ids)):
        if ids[i] != golden.tokens[i]:
            _fail(
                prompt.name,
                "tokens",
                "id["
                + String(i)
                + "] = "
                + String(ids[i])
                + ", golden "
                + String(golden.tokens[i]),
            )

    # (b) Forward: probe logits at 1e-6, argmax and top-5 ids exact.
    var logits = gpt.forward(ids)  # [T, V]
    var last = logits.rows - 1
    var last_row = logits.row(last)
    for p in range(len(golden.probe_idx)):
        var idx = golden.probe_idx[p]
        var got = logits[last, idx]
        var want = golden.probe_val[p]
        if _abs(got - want) > PROBE_TOL:
            _fail(
                prompt.name,
                "probe[" + String(idx) + "]",
                "got "
                + String(got)
                + ", golden "
                + String(want)
                + " (gap "
                + String(_abs(got - want))
                + ")",
            )
    var argmax = _argmax_row(last_row)
    if argmax != golden.argmax:
        _fail(
            prompt.name,
            "argmax",
            "got " + String(argmax) + ", golden " + String(golden.argmax),
        )
    var top5 = _top_k_ids(last_row, len(golden.top5))
    for i in range(len(top5)):
        if top5[i] != golden.top5[i]:
            _fail(
                prompt.name,
                "top5",
                "position "
                + String(i)
                + ": got "
                + String(top5[i])
                + ", golden "
                + String(golden.top5[i]),
            )

    # (c) Mean next-token NLL on the prompt's own tokens — GPT.loss over
    # ids[:-1] -> ids[1:]. Skipped for a single-token prompt (no next-token pair).
    if golden.has_nll:
        var ids_in = List[Int]()
        var targets = List[Int]()
        for i in range(len(ids) - 1):
            ids_in.append(ids[i])
            targets.append(ids[i + 1])
        var nll = gpt.loss(ids_in, targets)
        if _abs(nll - golden.nll) > NLL_TOL:
            _fail(
                prompt.name,
                "nll",
                "got "
                + String(nll)
                + ", golden "
                + String(golden.nll)
                + " (gap "
                + String(_abs(nll - golden.nll))
                + ")",
            )

    # (d) generate vs generate_cached — token-for-token, our-vs-our (short subset).
    var ran_greedy = False
    if _greedy_subset(prompt.name):
        ran_greedy = True
        var cfg = SamplerConfig.greedy()
        var no_stop = List[Int]()
        var rng_a = Rng(1337)
        var rng_b = Rng(1337)
        var uncached = generate(
            gpt, ids, GREEDY_NEW_TOKENS, cfg, no_stop, rng_a
        )
        var cached = generate_cached(
            gpt, ids, GREEDY_NEW_TOKENS, cfg, no_stop, rng_b
        )
        if len(uncached) != len(cached):
            _fail(
                prompt.name,
                "greedy-parity",
                "uncached emitted "
                + String(len(uncached))
                + ", cached "
                + String(len(cached)),
            )
        for i in range(len(uncached)):
            if uncached[i] != cached[i]:
                _fail(
                    prompt.name,
                    "greedy-parity",
                    "token "
                    + String(i)
                    + ": uncached "
                    + String(uncached[i])
                    + ", cached "
                    + String(cached[i]),
                )

    # Per-prompt PASS line.
    var nll_str = String("   n/a")
    if golden.has_nll:
        nll_str = String(golden.nll)
    print(
        "PASS  "
        + prompt.name
        + "  T="
        + String(len(ids))
        + "  argmax="
        + String(golden.argmax)
        + "  probes="
        + String(len(golden.probe_idx))
        + "  nll="
        + nll_str
        + ("  greedy=ok" if ran_greedy else "")
    )
    return ran_greedy


def main() raises:
    """Load the model once, then run every gauntlet check; PASS table or a named
    failure and non-zero exit."""
    _require_file(WEIGHTS_PATH)
    _require_file(VOCAB_PATH)
    _require_file(MERGES_PATH)
    _require_file(PROMPTS_PATH)
    _require_file(GOLDENS_PATH)

    var prompts = parse_prompts(open(PROMPTS_PATH, "r").read())
    var goldens = parse_goldens(open(GOLDENS_PATH, "r").read())
    if len(prompts) != len(goldens):
        raise Error(
            "gauntlet: "
            + String(len(prompts))
            + " prompts but "
            + String(len(goldens))
            + " golden blocks — regenerate goldens.txt from prompts.txt"
        )

    print(
        "loading GPT-2 124M weights (~2 GB resident: f64 values + zero"
        " grads)..."
    )
    var gpt = load_gpt2(WEIGHTS_PATH)
    var tok = GPT2Tokenizer.from_files(String(VOCAB_PATH), String(MERGES_PATH))
    print(
        "running the gauntlet: "
        + String(len(prompts))
        + " prompts, probes @ "
        + String(PROBE_TOL)
        + ", tokens/argmax/top5 exact"
    )
    print("")

    var greedy_runs = 0
    for i in range(len(prompts)):
        if _check_prompt(gpt, tok, prompts[i], goldens[i]):
            greedy_runs += 1

    print("")
    print(
        "GAUNTLET OK — "
        + String(len(prompts))
        + "/"
        + String(len(prompts))
        + " prompts matched the float64 reference (tokens/argmax/top5 exact,"
        " probes & nll @ 1e-6); generate vs generate_cached agreed on "
        + String(greedy_runs)
        + " short prompts."
    )
