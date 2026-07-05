# The fast go/no-go gate: does our forward pass reproduce GPT-2's logits?
#
# Loads the converted real GPT-2 124M weights, reconciles the parameter count,
# runs ONE forward over a fixed prompt, and asserts a handful of frozen logit
# goldens BEFORE printing anything — so a broken port fails loudly here instead of
# emitting plausible-looking token soup. The goldens come from
# scripts/gpt2_reference_logits.py (NumPy float64 over the SAME .bin bytes this
# loads), so the comparison is a tight f64-vs-f64 check at 1e-6, not a
# cross-precision one. Then it prints the top-5 next-token candidates with their
# decoded strings — the human-readable "yes, this is really GPT-2" signal.
#
# Run (after downloading the weights and running the converter — see
# scripts/convert_gpt2_weights.py):
#   pixi run mojo run -I build examples/gpt2_parity_check.mojo
#
# This is the CHEAP gate (one forward, seconds). examples/gpt2_generate.mojo is
# the slow MVP that actually generates text.

from llm.tensor.tensor2d import Tensor2D
from llm.tokenizer.gpt2 import GPT2Tokenizer
from llm.transformer.gpt2_weights import load_gpt2

comptime WEIGHTS_PATH = "checkpoints/gpt2-124m.bin"
comptime VOCAB_PATH = "data/gpt2/vocab.json"
comptime MERGES_PATH = "data/gpt2/merges.txt"
comptime PROMPT = "Hello, I'm a language model,"

# Frozen last-row logit goldens from scripts/gpt2_reference_logits.py on the
# converted weights (never from memory). The last-row argmax id is 407 (" not").
comptime GOLDEN_ARGMAX = 407


def _require_file(path: String) raises:
    # A cheap existence probe (open, do not read the 498 MB payload) with a message
    # pointing at the converter, so a missing weights file is a clear instruction
    # rather than an opaque IO error. There is NO fallback to random weights.
    try:
        var probe = open(path, "r")
        probe.close()
    except:
        raise Error(
            "missing '"
            + path
            + "' — download the HF gpt2 model.safetensors and run"
            " scripts/convert_gpt2_weights.py to produce it (see that script's"
            " header). No random-weight fallback: this gate needs the real"
            " weights."
        )


def _check(got: Float64, want: Float64, label: String) raises:
    # Raise (with the gap) unless got matches want within 1e-6 — the f64-vs-f64
    # parity tolerance. A broken transpose or a swapped slot lands here.
    var diff = got - want
    if diff < 0.0:
        diff = -diff
    if diff > 1e-6:
        raise Error(
            "PARITY FAILED at "
            + label
            + ": got "
            + String(got)
            + ", want "
            + String(want)
            + " (gap "
            + String(diff)
            + ") — indict the converter's layout mapping first, the loader's"
            " walk"
            " second"
        )


def _argmax_row(logits: Tensor2D, row: Int) -> Int:
    var best = 0
    var best_v = logits[row, 0]
    for c in range(1, logits.cols):
        if logits[row, c] > best_v:
            best_v = logits[row, c]
            best = c
    return best


def main() raises:
    _require_file(WEIGHTS_PATH)
    _require_file(VOCAB_PATH)
    _require_file(MERGES_PATH)

    print(
        "loading GPT-2 124M weights (~2 GB resident: f64 values + zero"
        " grads)..."
    )
    var gpt = load_gpt2(WEIGHTS_PATH)

    # Reconcile the parameter count against the comptime-pinned architecture total.
    var actual = gpt.parameter_count_actual()
    var expected = gpt.cfg.parameter_count()
    print("parameters:", actual, "(architecture total:", expected, ")")
    if actual != expected or actual != 124_439_808:
        raise Error(
            "parameter-count reconciliation failed: actual "
            + String(actual)
            + " vs expected "
            + String(expected)
        )

    var tok = GPT2Tokenizer.from_files(String(VOCAB_PATH), String(MERGES_PATH))
    var ids = tok.encode(String(PROMPT))
    # Pin that the tokenizer reproduced the ids the goldens assume: the GPT-2 BPE
    # encoding of PROMPT is [15496, 11, 314, 1101, 257, 3303, 2746, 11].
    var expected_ids = [15496, 11, 314, 1101, 257, 3303, 2746, 11]
    if len(ids) != len(expected_ids):
        raise Error("tokenizer produced an unexpected id count for the prompt")
    for i in range(len(ids)):
        if ids[i] != expected_ids[i]:
            raise Error(
                "tokenizer id mismatch at "
                + String(i)
                + ": got "
                + String(ids[i])
                + ", expected "
                + String(expected_ids[i])
            )

    var logits = gpt.forward(ids)  # [T, V]
    var last = logits.rows - 1

    # Assert the frozen goldens BEFORE printing any generated text.
    _check(logits[last, 0], -103.75580955455361, "logit[0]")
    _check(logits[last, 1], -105.5973120971462, "logit[1]")
    _check(logits[last, 50], -106.168011167856, "logit[50]")
    _check(logits[last, 100], -114.72522745337663, "logit[100]")
    _check(logits[last, 1000], -110.03431804960914, "logit[1000]")
    _check(logits[last, 10000], -107.4166916819877, "logit[10000]")
    _check(logits[last, 40000], -109.97812702387057, "logit[40000]")
    _check(logits[last, 407], -94.68738997956149, "logit[407] (argmax)")
    var argmax = _argmax_row(logits, last)
    if argmax != GOLDEN_ARGMAX:
        raise Error(
            "argmax mismatch: got "
            + String(argmax)
            + ", want "
            + String(GOLDEN_ARGMAX)
        )
    print("PARITY OK — logits match the float64 reference at 1e-6.")
    print()

    # Top-5 next-token candidates (by last-row logit), decoded.
    print('prompt: "' + PROMPT + '"')
    print("top-5 next tokens:")
    var picked = List[Int]()
    for _ in range(5):
        var best = -1
        var best_v = 0.0
        for c in range(logits.cols):
            var already = False
            for p in range(len(picked)):
                if picked[p] == c:
                    already = True
                    break
            if already:
                continue
            if best == -1 or logits[last, c] > best_v:
                best_v = logits[last, c]
                best = c
        picked.append(best)
        var one = List[Int]()
        one.append(best)
        print(
            "  id",
            best,
            " logit",
            logits[last, best],
            ' "' + tok.decode(one) + '"',
        )
