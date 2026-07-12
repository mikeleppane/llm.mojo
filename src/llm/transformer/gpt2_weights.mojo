"""Load released GPT-2 weights into our GPT: the `GPT2W v1` reader.

Pours OpenAI's GPT-2 124M weights into the from-scratch GPT struct so the Mojo
forward pass generates real English. Reads one file format, `GPT2W v1`, produced
offline by scripts/convert_gpt2_weights.py, which owns all GPT-2-specific
knowledge (HF tensor names, Conv1D transposes, reshapes). This loader is
deliberately dumb: it validates a header, streams the float32 payload in the
model's walk order, and builds the model.

Format: an ASCII newline-terminated header line

    GPT2W v1 <V> <T> <C> <L> <H> <param_count>

then the raw little-endian float32 payload, every parameter tensor back to back
in the model's walk order (wte, wpe, then per block ln1/qkv/proj/ln2/up/down,
then ln_f), row-major within each tensor. There are no per-tensor shape records:
shapes are derived from (V,T,C,L,H) by both writer and reader, so the walk is the
single source of truth. The declared param_count and the payload byte length
cross-check each other and the dims.

Distinct from the `GPTCKPT 1` checkpoint, a hex-text format for bit-exact trainer
resume (AdamW moments, step counter, rng state). A released model is just
parameters, so `GPT2W v1` stores raw float32 bytes (~475 MB) widened to f64
exactly on read. The returned GPT is ~2 GB resident at 124M (values plus zero
grads); load_gpt2 consumes no rng, raises named on any bad input, and sets
cfg.dropout to 0.0 (an inference artifact).
"""

from std.memory import bitcast

from llm.config import GPTConfig
from llm.nn.embedding import Embedding
from llm.nn.layernorm import LayerNorm
from llm.nn.linear import Linear
from llm.nn.mlp import MLP
from llm.nn.parameter import Parameter
from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.attention import MultiHeadAttention
from llm.transformer.block import TransformerBlock
from llm.transformer.gpt import GPT

# The format's magic/version tag: "GPT2W" is the file family, "v1" the version.
# The header's first two tokens must be exactly these, else the loader raises, so
# a future v2 payload can never be misread as a v1.
comptime GPT2W_MAGIC = "GPT2W v1"
comptime GPT2W_FAMILY = "GPT2W"
comptime GPT2W_VERSION_TAG = "v1"

# The header carries exactly these whitespace tokens: FAMILY VERSION V T C L H
# count.
comptime GPT2W_HEADER_TOKENS = 8


def _f32_bytes_to_f64(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> Float64:
    """Reconstruct a little-endian IEEE-754 float32 from 4 bytes, widened to float64.

    The widening is exact: every float32 is representable as a float64, so no
    rounding occurs. Allocates nothing; cannot raise.

    Args:
        b0: Least-significant byte.
        b1: Second byte.
        b2: Third byte.
        b3: Most-significant byte.

    Returns:
        The reconstructed value as a Float64.
    """
    var bits = (
        UInt32(Int(b0))
        | (UInt32(Int(b1)) << 8)
        | (UInt32(Int(b2)) << 16)
        | (UInt32(Int(b3)) << 24)
    )
    var f32 = bitcast[DType.float32, 1](SIMD[DType.uint32, 1](bits))[0]
    return Float64(f32)


def _read_tensor(
    raw: List[UInt8], mut cursor: Int, rows: Int, cols: Int
) raises -> Tensor2D:
    """Read rows*cols little-endian float32 values into a fresh [rows, cols] f64 tensor.

    Reads row-major from `raw` at byte `cursor`, advancing 4 bytes per value. The
    caller has already validated the payload length; the bound check is a cheap
    defensive guard.

    Args:
        raw: The payload bytes.
        cursor: Byte offset to start reading; advanced past the tensor.
        rows: Tensor row count.
        cols: Tensor column count.

    Returns:
        The [rows, cols] tensor. Allocates it; mutates the cursor.

    Raises:
        Error: If the payload runs out mid-tensor.
    """
    var out = zeros_2d(rows, cols)
    for i in range(rows):
        for j in range(cols):
            if cursor + 4 > len(raw):
                raise Error(
                    "load_gpt2: truncated payload (ran out of bytes mid-tensor)"
                )
            out[i, j] = _f32_bytes_to_f64(
                raw[cursor], raw[cursor + 1], raw[cursor + 2], raw[cursor + 3]
            )
            cursor += 4
    return out^


def _read_param(
    raw: List[UInt8], mut cursor: Int, rows: Int, cols: Int
) raises -> Parameter:
    """Read one [rows, cols] tensor off the payload and wrap it as a Parameter.

    The Parameter allocates a matching zero gradient. Mutates the cursor.

    Args:
        raw: The payload bytes.
        cursor: Byte offset to start reading; advanced past the tensor.
        rows: Tensor row count.
        cols: Tensor column count.

    Returns:
        The Parameter (value plus a zero grad). Allocates both.

    Raises:
        Error: On truncation.
    """
    var value = _read_tensor(raw, cursor, rows, cols)
    return Parameter(value^)


def load_gpt2(path: String) raises -> GPT:
    """Read a `GPT2W v1` file and build the GPT it describes.

    Validates the header (family, version, dims, declared count, payload byte
    length) with named errors, streams the float32 payload in walk order widening
    f32 -> f64 exactly, and constructs the model directly via fieldwise
    constructors — no donor model, so no rng is drawn. The returned model's
    cfg.dropout is 0.0.

    Args:
        path: Path to the GPT2W v1 file.

    Returns:
        The loaded GPT (~2 GB at 124M incl. the zero grads). Reads the file;
        allocates the model; consumes no rng.

    Raises:
        Error: On any malformed or mismatched input.
    """
    var raw = open(path, "r").read_bytes()

    # --- header line: bytes up to the first newline ---------------------------
    var newline_at = -1
    for i in range(len(raw)):
        if Int(raw[i]) == 10:  # '\n'
            newline_at = i
            break
    if newline_at < 0:
        raise Error(
            "load_gpt2: '"
            + path
            + "' is not a GPT2W file (no header line / newline found)"
        )
    var header_bytes = List[UInt8]()
    for i in range(newline_at):
        header_bytes.append(raw[i])
    var header = String(from_utf8_lossy=Span(header_bytes))

    var raw_tokens = header.split(" ")
    var tokens = List[String]()
    for i in range(len(raw_tokens)):
        tokens.append(String(raw_tokens[i]))
    if len(tokens) != GPT2W_HEADER_TOKENS:
        raise Error(
            "load_gpt2: malformed header '"
            + header
            + "' (expected "
            + String(GPT2W_HEADER_TOKENS)
            + " space-separated fields 'GPT2W v1 V T C L H count', got "
            + String(len(tokens))
            + ")"
        )
    if tokens[0] != GPT2W_FAMILY:
        raise Error(
            "load_gpt2: bad magic (expected file family '"
            + GPT2W_FAMILY
            + "', got '"
            + tokens[0]
            + "')"
        )
    if tokens[1] != GPT2W_VERSION_TAG:
        raise Error(
            "load_gpt2: unsupported version (this loader reads '"
            + GPT2W_MAGIC
            + "', got version tag '"
            + tokens[1]
            + "')"
        )

    var vocab_size = Int(tokens[2])
    var context_length = Int(tokens[3])
    var d_model = Int(tokens[4])
    var n_layers = Int(tokens[5])
    var n_heads = Int(tokens[6])
    var declared_count = Int(tokens[7])

    # cfg.validate() catches every degenerate/inconsistent dim (non-positive dims,
    # d_model not divisible by n_heads) with its own named error. dropout 0.0: a
    # loaded released model is an inference artifact.
    var cfg = GPTConfig(
        vocab_size, context_length, d_model, n_layers, n_heads, 0.0
    )
    cfg.validate()

    # The declared count must match the count the dims imply, and the payload's
    # byte length must match that count exactly — truncation and trailing bytes
    # are both rejected, so a partial download or a wrong-model file fails loudly.
    var expected_count = cfg.parameter_count()
    if declared_count != expected_count:
        raise Error(
            "load_gpt2: header parameter-count mismatch (header declares "
            + String(declared_count)
            + ", the dims V="
            + String(vocab_size)
            + " T="
            + String(context_length)
            + " C="
            + String(d_model)
            + " L="
            + String(n_layers)
            + " imply "
            + String(expected_count)
            + ")"
        )
    var payload_start = newline_at + 1
    var payload_len = len(raw) - payload_start
    var expected_bytes = expected_count * 4  # float32 = 4 bytes
    if payload_len < expected_bytes:
        raise Error(
            "load_gpt2: truncated payload (expected "
            + String(expected_bytes)
            + " bytes for "
            + String(expected_count)
            + " float32 parameters, file has "
            + String(payload_len)
            + ")"
        )
    if payload_len > expected_bytes:
        raise Error(
            "load_gpt2: trailing bytes after payload (expected "
            + String(expected_bytes)
            + " bytes, file has "
            + String(payload_len)
            + " — "
            + String(payload_len - expected_bytes)
            + " extra)"
        )

    # --- build the model in walk order as we stream the payload ----------------
    var c = d_model
    var d_hidden = 4 * c  # GPT-2's 4x feed-forward ratio
    var cursor = payload_start

    var wte = Embedding(_read_param(raw, cursor, vocab_size, c))  # [V, C]
    var wpe = Embedding(_read_param(raw, cursor, context_length, c))  # [T, C]

    var blocks = List[TransformerBlock]()
    for _ in range(n_layers):
        var ln1_w = _read_param(raw, cursor, 1, c)
        var ln1_b = _read_param(raw, cursor, 1, c)
        var ln1 = LayerNorm(ln1_w^, ln1_b^)

        var qkv_w = _read_param(raw, cursor, 3 * c, c)  # [3C, C]
        var qkv_b = _read_param(raw, cursor, 1, 3 * c)  # [1, 3C]
        var qkv = Linear(qkv_w^, qkv_b^)
        var proj_w = _read_param(raw, cursor, c, c)  # [C, C] square
        var proj_b = _read_param(raw, cursor, 1, c)  # [1, C]
        var proj = Linear(proj_w^, proj_b^)
        var attn = MultiHeadAttention(qkv^, proj^, n_heads)

        var ln2_w = _read_param(raw, cursor, 1, c)
        var ln2_b = _read_param(raw, cursor, 1, c)
        var ln2 = LayerNorm(ln2_w^, ln2_b^)

        var up_w = _read_param(raw, cursor, d_hidden, c)  # [4C, C]
        var up_b = _read_param(raw, cursor, 1, d_hidden)  # [1, 4C]
        var up = Linear(up_w^, up_b^)
        var down_w = _read_param(raw, cursor, c, d_hidden)  # [C, 4C]
        var down_b = _read_param(raw, cursor, 1, c)  # [1, C]
        var down = Linear(down_w^, down_b^)
        var mlp = MLP(up^, down^)

        blocks.append(TransformerBlock(ln1^, attn^, ln2^, mlp^))

    var ln_f_w = _read_param(raw, cursor, 1, c)
    var ln_f_b = _read_param(raw, cursor, 1, c)
    var ln_f = LayerNorm(ln_f_w^, ln_f_b^)

    var gpt = GPT(cfg.copy(), wte^, wpe^, blocks^, ln_f^)

    # Belt-and-suspenders: the model we just built must have exactly the parameter
    # float count the header/dims promised. This reconciles the walk we streamed
    # against the model's own parameter inventory; a mismatch means the loader's
    # walk drifted from the model's.
    var actual = gpt.parameter_count_actual()
    if actual != expected_count:
        raise Error(
            "load_gpt2: built model has "
            + String(actual)
            + " parameters but the header/dims imply "
            + String(expected_count)
            + " — the loader's walk drifted from the model"
        )
    return gpt^
