# Training checkpoints — versioned, bit-exact, resume-proven.
#
# A checkpoint captures everything an interrupted training run needs to resume
# IDENTICALLY: every model parameter, both AdamW moment lists (m, v), the step
# counter t, and the training rng state. The gate that makes it real is the
# resume equality test — train k steps, checkpoint, load into a fresh model,
# train n-k more, and the parameters are BIT-IDENTICAL to an uninterrupted
# n-step run. That only holds if the round-trip is exact, not
# approximately-parsed, so every Float64 is stored as its raw IEEE-754 bit
# pattern (a 16-digit hex UInt64) rather than a decimal that must be re-rounded.
#
# Format (text, version 1):
#
#   line 0:            "GPTCKPT 1"                 magic + version
#   line 1:            N                            parameter tensor count
#   line 2:            t                            step counter
#   line 3:            <rng_state as 16 hex>        one UInt64, our LCG's state
#   lines 4..4+N-1:    "rows cols"                  each parameter's shape, walk order
#   then, in walk order and row-major within each tensor, one hex Float64 per line:
#     parameters (N tensors), then m (N tensors), then v (N tensors).
#
# Load validates the header against the LIVE model's parameter_shapes() and
# raises with a named mismatch — a checkpoint for a different architecture must
# fail loudly, never read garbage. This is a TRAINING artifact; loading real
# GPT-2 weights is a separate format and a separate part.
#
# Scope cut (documented): a shuffled-loader run's cursor is NOT checkpointed, so
# resuming such a run mid-epoch is only approximate. The exactness gate runs on
# the overfit-batch setup, where a fixed batch means there is no loader state to
# lose.

from std.memory import bitcast

from llm.tensor.tensor2d import Tensor2D, zeros_2d
from llm.transformer.gpt import GPT

comptime CKPT_MAGIC = String("GPTCKPT")
comptime CKPT_VERSION = 1
comptime HEX_DIGITS = String("0123456789abcdef")

# Flush the per-tensor hex buffer to disk at least this often (in lines) so a
# large checkpoint never materializes as one giant string.
comptime WRITE_FLUSH_LINES = 8192


@fieldwise_init
struct CheckpointState(Copyable, Movable):
    # What load_checkpoint returns alongside the parameters it restores into the
    # model: the two AdamW moment lists (walk order), the step counter, and the
    # rng state — the trainer-owned state the model itself does not hold.
    var m: List[Tensor2D]
    var v: List[Tensor2D]
    var t: Int
    var rng_state: UInt64


def u64_to_hex16(bits: UInt64) -> String:
    # A UInt64 as exactly 16 lowercase hex digits, most-significant nibble first.
    # Fixed width so every value occupies one full line; allocates the string.
    var out = String("")
    for i in range(16):
        var shift = UInt64((15 - i) * 4)
        var nibble = Int((bits >> shift) & 0xF)
        out += HEX_DIGITS[byte=nibble]
    return out


def parse_hex_u64(s: String) raises -> UInt64:
    # Parse exactly 16 lowercase hex digits back to a UInt64. Raises on a wrong
    # length or a non-hex character (a corrupt or truncated line).
    if s.byte_length() != 16:
        raise Error(
            "checkpoint: expected 16 hex digits, got '"
            + s
            + "' ("
            + String(s.byte_length())
            + " chars)"
        )
    var acc: UInt64 = 0
    for i in range(16):
        var ch = ord(s[byte=i])
        if ch >= ord("0") and ch <= ord("9"):
            acc = acc * 16 + UInt64(ch - ord("0"))
        elif ch >= ord("a") and ch <= ord("f"):
            acc = acc * 16 + UInt64(ch - ord("a") + 10)
        else:
            raise Error("checkpoint: bad hex digit in '" + s + "'")
    return acc


def f64_to_hex(x: Float64) -> String:
    # A Float64 as its 16-hex-digit IEEE-754 bit pattern (exact, no rounding).
    return u64_to_hex16(x.to_bits[DType.uint64]())


def hex_to_f64(s: String) raises -> Float64:
    # Reconstruct a Float64 from its 16-hex bit pattern. Raises on a bad line.
    return bitcast[DType.float64, 1](SIMD[DType.uint64, 1](parse_hex_u64(s)))[0]


def _write_tensors(mut f: FileHandle, tensors: List[Tensor2D]) raises:
    # Write every tensor's values as hex lines, row-major, walk order, flushing
    # in bounded chunks so no single string grows to the whole file.
    var buf = String("")
    var pending = 0
    for k in range(len(tensors)):
        for i in range(tensors[k].rows):
            for j in range(tensors[k].cols):
                buf += f64_to_hex(tensors[k][i, j]) + "\n"
                pending += 1
                if pending >= WRITE_FLUSH_LINES:
                    f.write(buf)
                    buf = String("")
                    pending = 0
    if buf.byte_length() > 0:
        f.write(buf)


def _read_tensor(
    lines: List[String], mut cursor: Int, rows: Int, cols: Int
) raises -> Tensor2D:
    # Read rows*cols hex lines starting at `cursor` into a fresh tensor (row
    # major), advancing the cursor. Raises if the file runs out of lines (a
    # truncated checkpoint).
    var out = zeros_2d(rows, cols)
    for i in range(rows):
        for j in range(cols):
            if cursor >= len(lines):
                raise Error(
                    "checkpoint: truncated file (ran out of values at tensor"
                    " element)"
                )
            out[i, j] = hex_to_f64(String(lines[cursor]))
            cursor += 1
    return out^


def save_checkpoint(
    path: String,
    gpt: GPT,
    m: List[Tensor2D],
    v: List[Tensor2D],
    t: Int,
    rng_state: UInt64,
) raises:
    # Write a complete resume checkpoint. Reads gpt (via export_parameters /
    # parameter_shapes) and the trainer-owned m, v, t, rng_state. Raises if m or v
    # do not have one tensor per parameter, or if any moment tensor's shape does
    # not match its parameter — either would write a file whose payload no longer
    # lines up with the header shapes, so load would silently shift values between
    # tensors. Overwrites `path`; allocates the export copies and the write buffers.
    var shapes = gpt.parameter_shapes()
    var params = gpt.export_parameters()
    var n = len(shapes)
    if len(m) != n or len(v) != n:
        raise Error(
            "save_checkpoint: m/v must have one tensor per parameter ("
            + String(n)
            + "), got len(m)="
            + String(len(m))
            + ", len(v)="
            + String(len(v))
        )
    for k in range(n):
        if m[k].rows != shapes[k].rows or m[k].cols != shapes[k].cols:
            raise Error(
                "save_checkpoint: m["
                + String(k)
                + "] shape does not match parameter "
                + String(k)
            )
        if v[k].rows != shapes[k].rows or v[k].cols != shapes[k].cols:
            raise Error(
                "save_checkpoint: v["
                + String(k)
                + "] shape does not match parameter "
                + String(k)
            )

    var header = CKPT_MAGIC + " " + String(CKPT_VERSION) + "\n"
    header += String(n) + "\n"
    header += String(t) + "\n"
    header += u64_to_hex16(rng_state) + "\n"
    for k in range(n):
        header += String(shapes[k].rows) + " " + String(shapes[k].cols) + "\n"

    with open(path, "w") as f:
        f.write(header)
        _write_tensors(f, params)
        _write_tensors(f, m)
        _write_tensors(f, v)


def load_checkpoint(path: String, mut gpt: GPT) raises -> CheckpointState:
    # Restore a checkpoint into `gpt` (in place) and return the trainer-owned
    # state (m, v, t, rng_state). Validates the header against the live model's
    # parameter_shapes() and raises with a NAMED mismatch on a bad magic, an
    # unsupported version, a parameter-count or shape disagreement, or a truncated
    # file — never reads garbage. Mutates gpt's parameter values; allocates the
    # returned state.
    var content = open(path, "r").read()
    # split() yields slices into `content`; own them as Strings so the helpers
    # take a plain List[String] and the values outlive `content`.
    var raw = content.split("\n")
    var lines = List[String]()
    for i in range(len(raw)):
        lines.append(String(raw[i]))
    # A checkpoint always ends with a trailing newline, so split() leaves one
    # empty final element; drop it so a genuinely-short (truncated) file runs out
    # of lines cleanly rather than surfacing as a blank-line parse error.
    if len(lines) > 0 and lines[len(lines) - 1].byte_length() == 0:
        _ = lines.pop()

    if len(lines) < 4:
        raise Error("load_checkpoint: file too short to hold a header")
    if String(lines[0]) != CKPT_MAGIC + " " + String(CKPT_VERSION):
        raise Error(
            "load_checkpoint: bad magic/version line (expected '"
            + CKPT_MAGIC
            + " "
            + String(CKPT_VERSION)
            + "', got '"
            + String(lines[0])
            + "')"
        )

    var n = Int(String(lines[1]))
    var shapes = gpt.parameter_shapes()
    if n != len(shapes):
        raise Error(
            "load_checkpoint: parameter count mismatch — file has "
            + String(n)
            + ", model has "
            + String(len(shapes))
        )

    var t = Int(String(lines[2]))
    var rng_state = parse_hex_u64(String(lines[3]))

    # Validate each shape line against the live model.
    if len(lines) < 4 + n:
        raise Error("load_checkpoint: truncated file (missing shape lines)")
    for k in range(n):
        var parts = String(lines[4 + k]).split(" ")
        if len(parts) != 2:
            raise Error(
                "load_checkpoint: malformed shape line " + String(4 + k)
            )
        var rows = Int(String(parts[0]))
        var cols = Int(String(parts[1]))
        if rows != shapes[k].rows or cols != shapes[k].cols:
            raise Error(
                "load_checkpoint: shape mismatch at tensor "
                + String(k)
                + " — file ("
                + String(rows)
                + ", "
                + String(cols)
                + ") vs model ("
                + String(shapes[k].rows)
                + ", "
                + String(shapes[k].cols)
                + ")"
            )

    # Read the three float sections (params, m, v) in walk order.
    var cursor = 4 + n
    var params = List[Tensor2D]()
    for k in range(n):
        params.append(
            _read_tensor(lines, cursor, shapes[k].rows, shapes[k].cols)
        )
    var m = List[Tensor2D]()
    for k in range(n):
        m.append(_read_tensor(lines, cursor, shapes[k].rows, shapes[k].cols))
    var v = List[Tensor2D]()
    for k in range(n):
        v.append(_read_tensor(lines, cursor, shapes[k].rows, shapes[k].cols))

    # The three sections must consume the file exactly. Extra values mean the file
    # does not match this model (or is corrupt); reject rather than ignore them.
    if cursor != len(lines):
        raise Error(
            "load_checkpoint: file has "
            + String(len(lines) - cursor)
            + " extra value line(s) after params, m, v"
        )

    gpt.import_parameters(params)
    return CheckpointState(m^, v^, t, rng_state)
