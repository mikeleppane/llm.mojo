# Tests for the GPT2W v1 weight loader (transformer/gpt2_weights.mojo).
#
# The doll-house fixture (V 11, T 8, C 8, L 1, H 2) is written at TEST TIME by
# the Python oracle tests/oracles/gpt2_weights_reference.py — the same live-Python
# arrangement the tokenizer test uses — so the suite stays offline and touches no
# large files. The oracle's sentinel weights are ASYMMETRIC (a per-slot base plus
# different row/column coefficients), so the pins below catch the failure this
# whole part exists to defeat: a missing transpose on a SQUARE kernel, a swapped
# walk slot, an ingested buffer — none of which a shape check can see.
#
# Every golden here is ALSO frozen inline (printed once by the oracle's main and
# pasted), so a silently broken fixture writer is caught: the file under test and
# the numbers it is checked against come from independent transcriptions.

from std.memory import bitcast
from std.tempfile import gettempdir
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)
from std.python import Python, PythonObject

from llm.generation.generate import generate
from llm.generation.sampler import SamplerConfig
from llm.transformer.gpt2_weights import load_gpt2
from llm.utils.random import Rng

comptime V = 11
comptime T = 8
comptime C = 8
comptime L = 1
comptime H = 2
comptime DOLLHOUSE_PARAM_COUNT = 1040


def _oracle() raises -> PythonObject:
    # The doll-house fixture writer + reference (tests/oracles/).
    Python.add_to_path("tests/oracles")
    return Python.import_module("gpt2_weights_reference")


def _tmp_path(name: String) raises -> String:
    var d = gettempdir()
    if not d:
        raise Error("no temp directory available")
    return d.value() + "/" + name


def _write_and_path(oracle_fn: String, name: String) raises -> String:
    # Call one of the oracle's writer functions to drop a fixture at a tempdir
    # path, returning that path.
    var path = _tmp_path(name)
    var oracle = _oracle()
    _ = oracle.__getattr__(oracle_fn)(path)
    return path


def test_load_reconciles_shapes_and_count() raises:
    # The happy path loads, the dims come back as written, and the built model's
    # actual float count reconciles with the doll-house parameter count.
    var path = _write_and_path("write_fixture", "gpt2w_ok.bin")
    var gpt = load_gpt2(path)
    assert_equal(gpt.cfg.vocab_size, V)
    assert_equal(gpt.cfg.context_length, T)
    assert_equal(gpt.cfg.d_model, C)
    assert_equal(gpt.cfg.n_layers, L)
    assert_equal(gpt.cfg.n_heads, H)
    assert_almost_equal(gpt.cfg.dropout, 0.0, atol=0.0)  # inference artifact
    assert_equal(gpt.parameter_count_actual(), DOLLHOUSE_PARAM_COUNT)
    # Shapes land where the walk says they should.
    assert_true(
        gpt.wte.table.value.rows == V and gpt.wte.table.value.cols == C,
        "wte must be [V, C]",
    )
    assert_true(
        gpt.wpe.table.value.rows == T and gpt.wpe.table.value.cols == C,
        "wpe must be [T, C]",
    )
    assert_true(
        gpt.blocks[0].attn.qkv.weight.value.rows == 3 * C
        and gpt.blocks[0].attn.qkv.weight.value.cols == C,
        "qkv.weight must be [3C, C]",
    )
    assert_true(
        gpt.blocks[0].mlp.down.weight.value.rows == C
        and gpt.blocks[0].mlp.down.weight.value.cols == 4 * C,
        "down.weight must be [C, 4C]",
    )


def test_transpose_and_walk_pins() raises:
    # Asymmetric sentinels pin the layout per slot. proj is the SQUARE [C, C]
    # kernel — proj.w[0,1] != proj.w[1,0], so a loader that transposed it (a bug a
    # shape check cannot catch) would swap these two and fail. The distinct wte,
    # qkv, and ln_f values pin that each of the 16 tensors landed in its named
    # field: a swapped walk slot (ln1<->ln2, up<->down) lands a wrong base.
    var path = _write_and_path("write_fixture", "gpt2w_pins.bin")
    var gpt = load_gpt2(path)

    # SQUARE proj kernel: the two off-diagonal entries differ, proving no transpose.
    assert_almost_equal(
        gpt.blocks[0].attn.proj.weight.value[0, 1],
        0.04699999839067459,
        atol=1e-12,
    )
    assert_almost_equal(
        gpt.blocks[0].attn.proj.weight.value[1, 0],
        0.07999999821186066,
        atol=1e-12,
    )
    # qkv kernel corners (asymmetric, [3C, C]).
    assert_almost_equal(
        gpt.blocks[0].attn.qkv.weight.value[0, 0],
        0.03999999910593033,
        atol=1e-12,
    )
    assert_almost_equal(
        gpt.blocks[0].attn.qkv.weight.value[3 * C - 1, C - 1],
        0.4090000092983246,
        atol=1e-12,
    )
    # wte last row/col (base 0) and ln_f (bases 14, 15) — distinct walk slots.
    assert_almost_equal(
        gpt.wte.table.value[V - 1, C - 1], 0.10899999737739563, atol=1e-12
    )
    assert_almost_equal(
        gpt.ln_f.weight.value[0, 0], 0.14000000059604645, atol=1e-12
    )
    assert_almost_equal(
        gpt.ln_f.bias.value[0, 0], 0.15000000596046448, atol=1e-12
    )


def test_widening_is_bit_exact() raises:
    # A probe file sets wte[0,0] to float32(0.1); after loading, its float64 bit
    # pattern must be EXACTLY the float64 image of 0.1f32 (0x3FB99999A0000000 =
    # 4591870180174331904). This is exact equality on the raw bits, not a
    # tolerance: the f32 -> f64 read is a widening (every float32 is a float64),
    # never a decimal re-parse.
    var path = _write_and_path("write_widen_probe", "gpt2w_widen.bin")
    var gpt = load_gpt2(path)
    var bits = gpt.wte.table.value[0, 0].to_bits[DType.uint64]()
    assert_equal(bits, UInt64(4591870180174331904))


def test_header_errors_are_named() raises:
    # Each malformed file raises. Bad family token, wrong version tag, a dim that
    # fails validation (C not divisible by H), a payload short by one float, and a
    # payload long by one float — the five ways a GPT2W file can be wrong.
    with assert_raises():
        _ = load_gpt2(_write_and_path("write_bad_magic", "gpt2w_badmagic.bin"))
    with assert_raises():
        _ = load_gpt2(_write_and_path("write_wrong_version", "gpt2w_ver.bin"))
    with assert_raises():
        _ = load_gpt2(_write_and_path("write_wrong_dims", "gpt2w_dims.bin"))
    with assert_raises():
        _ = load_gpt2(_write_and_path("write_truncated", "gpt2w_trunc.bin"))
    with assert_raises():
        _ = load_gpt2(_write_and_path("write_trailing", "gpt2w_trail.bin"))


def _reference_logits() raises -> List[Float64]:
    # The oracle's frozen doll-house logits [T=5, V=11] for ids=[1,3,4,0,2], row
    # major. Frozen inline from `pixi run python
    # tests/oracles/gpt2_weights_reference.py` (never from memory).
    var vals = List[Float64]()
    vals.append(-0.019278374374598506)
    vals.append(0.002201473184772337)
    vals.append(0.023681320694769862)
    vals.append(0.045161167602985774)
    vals.append(0.06664101587802089)
    vals.append(0.08812086317006772)
    vals.append(0.10960070932454913)
    vals.append(0.1310805597093618)
    vals.append(0.1525604046782658)
    vals.append(0.1740402547529772)
    vals.append(0.19552010208571877)
    vals.append(-0.019278374387635806)
    vals.append(0.0022014731715223157)
    vals.append(0.023681320681307128)
    vals.append(0.04516116758931033)
    vals.append(0.06664101586413272)
    vals.append(0.08812086315596683)
    vals.append(0.1096007093102355)
    vals.append(0.13108055969483542)
    vals.append(0.15256040466352677)
    vals.append(0.17404025473802542)
    vals.append(0.19552010207055426)
    vals.append(-0.01927837439473181)
    vals.append(0.0022014731645951637)
    vals.append(0.023681320674548832)
    vals.append(0.04516116758272089)
    vals.append(0.06664101585771215)
    vals.append(0.08812086314971508)
    vals.append(0.10960070930415262)
    vals.append(0.13108055968892146)
    vals.append(0.15256040465778162)
    vals.append(0.17404025473244913)
    vals.append(0.19552010206514678)
    vals.append(-0.01927837425124538)
    vals.append(0.002201473307898149)
    vals.append(0.023681320817668404)
    vals.append(0.04516116772565705)
    vals.append(0.06664101600046489)
    vals.append(0.08812086329228445)
    vals.append(0.10960070944653856)
    vals.append(0.131080559831124)
    vals.append(0.1525604047998007)
    vals.append(0.17404025487428487)
    vals.append(0.19552010220679913)
    vals.append(-0.019278374247101385)
    vals.append(0.0022014733122010083)
    vals.append(0.023681320822130092)
    vals.append(0.045161167730277584)
    vals.append(0.06664101600524422)
    vals.append(0.08812086329722264)
    vals.append(0.10960070945163553)
    vals.append(0.13108055983637978)
    vals.append(0.1525604048052154)
    vals.append(0.17404025487985836)
    vals.append(0.19552010221253144)
    return vals^


def _fixture_ids() raises -> List[Int]:
    var out = List[Int]()
    out.append(1)
    out.append(3)
    out.append(4)
    out.append(0)
    out.append(2)
    return out^


def test_forward_matches_reference() raises:
    # End to end: file -> loader -> forward -> the NumPy f64 reference logits, one
    # assertion chain at 1e-9. A wrong transpose, a swapped slot, an off-by-one
    # walk, or a missing tensor lands a wrong number somewhere in the 55-value
    # comparison.
    var path = _write_and_path("write_fixture", "gpt2w_fwd.bin")
    var gpt = load_gpt2(path)
    var logits = gpt.forward(_fixture_ids())
    assert_true(logits.rows == 5 and logits.cols == V, "logits must be [T, V]")
    var expected = _reference_logits()
    for r in range(5):
        for c in range(V):
            assert_almost_equal(logits[r, c], expected[r * V + c], atol=1e-9)


def test_loaded_model_generates() raises:
    # The XV/XVI seam: a loaded model runs greedily for 3 tokens without raising,
    # every emitted id in range. Greedy (temperature 0) draws no rng.
    var path = _write_and_path("write_fixture", "gpt2w_gen.bin")
    var gpt = load_gpt2(path)
    var prompt = List[Int]()
    prompt.append(1)
    prompt.append(2)
    var rng = Rng(0)
    var emitted = generate(
        gpt, prompt, 3, SamplerConfig.greedy(), List[Int](), rng
    )
    assert_equal(len(emitted), 3)
    for i in range(len(emitted)):
        assert_true(
            emitted[i] >= 0 and emitted[i] < V, "emitted id must be in [0, V)"
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
