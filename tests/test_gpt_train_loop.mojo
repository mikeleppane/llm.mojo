# Tests for the real training loop: train_gpt (AdamW + warmup/cosine schedule +
# global-norm clipping over a BatchLoader) and estimate_loss.
#
# The capstone is overfit-one-batch through the REAL trainer: a correct
# model + loop crushes the loss on one fixed batch far below the log V uniform
# baseline. Dropout is 0 so the run is fully determined by the seed, and a
# second run from the same seed reproduces the loss history bit-for-bit. A short
# dropout = 0.1 run must still end below its starting loss (no per-step
# monotonicity claim under dropout noise). Plus: estimate_loss matches a
# hand-averaged dropout-free loss, TrainReport history lengths, and the AdamW
# preset (beta2 = 0.95, the GPT-family value) is pinned.

from std.math import log

from std.testing import (
    assert_almost_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from llm.config import GPTConfig, TrainingConfig
from llm.data.dataset import TokenDataset
from llm.data.loader import BatchLoader
from llm.tensor.ops import cross_entropy_rows
from llm.training.gpt_trainer import estimate_loss, train_gpt
from llm.training.optimizer import AdamWConfig
from llm.training.schedule import ScheduleConfig
from llm.transformer.gpt import GPT
from llm.utils.random import Rng

comptime VOCAB = 8
comptime SEQ_LEN = 4
comptime BATCH = 2


def _dataset() raises -> TokenDataset:
    # A tiny fixed corpus; two non-overlapping windows of length SEQ_LEN fill one
    # batch of BATCH sequences (size 9 -> window starts 0 and 4).
    var ids = List[Int]()
    var pattern = List[Int]()
    pattern.append(1)
    pattern.append(4)
    pattern.append(2)
    pattern.append(6)
    pattern.append(3)
    pattern.append(5)
    pattern.append(0)
    pattern.append(7)
    pattern.append(2)
    pattern.append(6)
    for i in range(len(pattern)):
        ids.append(pattern[i])
    return TokenDataset(ids^)


def _loader() raises -> BatchLoader:
    return BatchLoader(_dataset(), BATCH, SEQ_LEN, stride=SEQ_LEN)


def _tiny_gpt(dropout: Float64, seed: UInt64) raises -> GPT:
    var cfg = GPTConfig(VOCAB, 8, SEQ_LEN, 2, 2, dropout)
    var rng = Rng(seed)
    return GPT.init_random(cfg, rng)


def _run(
    dropout: Float64, seed: UInt64, max_steps: Int
) raises -> List[Float64]:
    # Run train_gpt and return the per-step training loss history.
    var gpt = _tiny_gpt(dropout, seed)
    var train_loader = _loader()
    var val_loader = _loader()
    var tc = TrainingConfig(BATCH, 0.02, max_steps, seed)
    var oc = AdamWConfig.gpt2_defaults()
    var sc = ScheduleConfig(5, 0.001)
    var rng = Rng(seed + 1234567)  # dropout stream, distinct from loader seeds
    var report = train_gpt(
        gpt, train_loader, val_loader, tc, oc, sc, rng, 20, 1
    )
    return report.train_losses.copy()


def test_overfit_one_batch_crushes_loss() raises:
    # dropout 0: the trainer drives the fixed batch's loss far below log V.
    var losses = _run(0.0, 7, 80)
    var log_v = log(Float64(VOCAB))  # ~2.079
    var init = losses[0]
    var final = losses[len(losses) - 1]
    assert_true(
        init > log_v - 0.8, "init loss " + String(init) + " not near log V"
    )
    assert_true(
        final < log_v - 1.2,
        "final loss " + String(final) + " did not crush below log V",
    )


def test_overfit_is_deterministic() raises:
    # dropout 0 draws no rng, and the loader reshuffle is seeded, so two runs from
    # the same seed produce bit-identical loss histories.
    var a = _run(0.0, 11, 40)
    var b = _run(0.0, 11, 40)
    assert_true(len(a) == len(b), "history length drift")
    for i in range(len(a)):
        assert_true(a[i] == b[i], "run not deterministic at step " + String(i))


def test_dropout_run_ends_below_init() raises:
    # dropout 0.1: no per-step monotonicity claim, but the run must still end well
    # below its starting loss.
    var losses = _run(0.1, 21, 80)
    var init = losses[0]
    var final = losses[len(losses) - 1]
    assert_true(
        final < init - 0.5,
        "dropout run final "
        + String(final)
        + " not below init "
        + String(init),
    )


def test_estimate_loss_matches_hand_average() raises:
    # estimate_loss uses the dropout-free forward, averaged over the batch. Build
    # it by hand from gpt.forward over the same sequences and compare.
    var gpt = _tiny_gpt(0.0, 3)
    var loader = _loader()
    var got = estimate_loss(gpt, loader, 1)

    # Hand average: mean cross_entropy_rows over the one batch's BATCH sequences.
    var ds = _dataset()
    var starts = List[Int]()
    starts.append(0)
    starts.append(SEQ_LEN)
    var total = 0.0
    for s in range(len(starts)):
        var ids = List[Int]()
        var targets = List[Int]()
        for t in range(SEQ_LEN):
            ids.append(ds.ids[starts[s] + t])
            targets.append(ds.ids[starts[s] + 1 + t])
        total += cross_entropy_rows(gpt.forward(ids), targets)
    var expected = total / Float64(len(starts))
    assert_almost_equal(got, expected, atol=1e-12)


def test_estimate_loss_preserves_cursor() raises:
    # estimate_loss must not disturb a shared loader's position.
    var gpt = _tiny_gpt(0.0, 3)
    var loader = _loader()
    var before = loader.cursor
    _ = estimate_loss(gpt, loader, 1)
    assert_true(loader.cursor == before, "estimate_loss moved the cursor")


def test_train_report_history_lengths() raises:
    var gpt = _tiny_gpt(0.0, 5)
    var train_loader = _loader()
    var val_loader = _loader()
    var tc = TrainingConfig(BATCH, 0.02, 60, 5)
    var oc = AdamWConfig.gpt2_defaults()
    var sc = ScheduleConfig(5, 0.001)
    var rng = Rng(999)
    var report = train_gpt(
        gpt, train_loader, val_loader, tc, oc, sc, rng, 20, 1
    )
    # Per-step histories have one entry per step.
    assert_true(len(report.train_losses) == 60, "train_losses length")
    assert_true(len(report.grad_norms) == 60, "grad_norms length")
    assert_true(len(report.lrs) == 60, "lrs length")
    # Evals at steps 19, 39, 59 (every 20, plus the final step already at 59).
    assert_true(len(report.eval_steps) == 3, "eval_steps length")
    assert_true(len(report.eval_train_losses) == 3, "eval_train length")
    assert_true(len(report.eval_val_losses) == 3, "eval_val length")


def test_segmented_resume_matches_straight_run() raises:
    # Training in two segments through train_gpt — [0, k) then [k, n) with the
    # optimizer moments carried across via init_m/init_v and start_step — must
    # reproduce a single [0, n) run bit-for-bit (dropout 0, fixed batch). This
    # exercises train_gpt's own resume path: the step-counter/schedule offset, the
    # threaded m/v, and the loader-position reconstruction.
    var n = 12
    var k = 5

    # Straight run.
    var full = _tiny_gpt(0.0, 4)
    var tl_full = _loader()
    var vl_full = _loader()
    var tc = TrainingConfig(BATCH, 0.02, n, 4)
    var oc = AdamWConfig.gpt2_defaults()
    var sc = ScheduleConfig(3, 0.001)
    var rng_full = Rng(0)  # dropout 0 -> unused
    var rep_full = train_gpt(
        full, tl_full, vl_full, tc, oc, sc, rng_full, 100, 1
    )

    # Segmented run: [0, k) then [k, n), carrying m/v across.
    var part = _tiny_gpt(0.0, 4)
    var tl_part = _loader()
    var vl_part = _loader()
    var rng_p = Rng(0)
    var seg1 = train_gpt(
        part, tl_part, vl_part, tc, oc, sc, rng_p, 100, 1, 0, [], [], k
    )
    var seg2 = train_gpt(
        part,
        tl_part,
        vl_part,
        tc,
        oc,
        sc,
        rng_p,
        100,
        1,
        k,
        seg1.m.copy(),
        seg1.v.copy(),
        n,
    )

    var pf = full.export_parameters()
    var pp = part.export_parameters()
    for j in range(len(pf)):
        for r in range(pf[j].rows):
            for c in range(pf[j].cols):
                assert_true(
                    pf[j][r, c] == pp[j][r, c],
                    "segmented resume diverged at tensor " + String(j),
                )


def test_adamw_config_defaults_pinned() raises:
    # The GPT-training preset — beta2 is 0.95, NOT Adam's 0.999 habit.
    var oc = AdamWConfig.gpt2_defaults()
    assert_true(oc.beta1 == 0.9, "beta1 default")
    assert_true(oc.beta2 == 0.95, "beta2 must be 0.95 (GPT-family), not 0.999")
    assert_true(oc.eps == 1e-8, "eps default")
    assert_true(oc.weight_decay == 0.1, "weight_decay default")
    assert_true(oc.grad_clip == 1.0, "grad_clip default")


def test_adamw_config_validate() raises:
    AdamWConfig.gpt2_defaults().validate()
    with assert_raises(contains="beta2 must be in"):
        AdamWConfig(0.9, 1.0, 1e-8, 0.1, 1.0).validate()
    with assert_raises(contains="eps must be positive"):
        AdamWConfig(0.9, 0.95, 0.0, 0.1, 1.0).validate()
    with assert_raises(contains="grad_clip must be positive"):
        AdamWConfig(0.9, 0.95, 1e-8, 0.1, 0.0).validate()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
