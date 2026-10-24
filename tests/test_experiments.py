"""Phase 8 experiment machinery: baseline freeze guard, Exp B class weights, Exp C real-pair sampling, comparison."""

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data.dataset import prepare_dataset
from src.data.texture_dataset import TextureDataset
from src.evaluation.compare import compare_results, format_comparison
from src.training.checkpointing import CheckpointManager, FrozenRunError, freeze_run, is_frozen, prune_snapshots
from src.training.train_texture import TextureTrainer
from src.training.train_warp import WarpTrainer, class_weights_from_counts
from tests.test_tryon import ROOT, cfg_for


def test_class_weights_from_counts():
    counts = np.array([1000.0, 0, 100, 10, 0, 50])
    w = class_weights_from_counts(counts, "median_freq")
    assert w[1] == 0 and w[4] == 0  # absent classes
    assert w[3] > w[2] > w[0]  # rarer -> larger
    ws = class_weights_from_counts(counts, "sqrt_median_freq")
    assert ws[0] > w[0] and ws[3] < w[3]  # milder in both directions
    assert ((w[w > 0] >= 0.2) & (w[w > 0] <= 5.0)).all()
    with pytest.raises(ValueError):
        class_weights_from_counts(counts, "bogus")


def test_baseline_configs_unchanged_and_experiment_configs_isolated():
    b0 = load_config(ROOT / "configs" / "colab_pairfilter.yaml")
    assert b0.experiment.name == "baseline_128" and b0.training.warp_run_name == "warp_pf"
    assert b0.training.warp_class_weights == "none" and b0.training.texture_real_pair_prob == 0.0
    assert b0.evaluation.warp_checkpoint == "last" and b0.evaluation.texture_checkpoint == "last"
    for name, field, value in (("exp_b_classweights", "warp_class_weights", "sqrt_median_freq"),
                               ("exp_c_realpairs", "texture_real_pair_prob", 0.8)):
        c = load_config(ROOT / "configs" / f"{name}.yaml", ["paths.drive_root=/d"])
        assert c.experiment.name == name != b0.experiment.name  # own folders for checkpoints / outputs / logs
        assert getattr(c.training, field) == value
        # everything else that defines the model and data is the same as B0
        for sect in ("data", "model", "labels", "segmentation"):
            a, b = getattr(c, sect).__dict__, getattr(load_config(ROOT / "configs" / "colab_pairfilter.yaml",
                                                                  ["paths.drive_root=/d"]), sect).__dict__
            assert a == b, sect
    c = load_config(ROOT / "configs" / "exp_c_realpairs.yaml", ["paths.drive_root=/d"])
    assert c.evaluation.warp_checkpoint == "/d/checkpoints/baseline_128/warp_pf/last.pt"


def test_freeze_guard_blocks_training_and_saving(tmp_path):
    cfg = cfg_for(tmp_path)
    prepared = prepare_dataset(cfg)
    tr = WarpTrainer(cfg, prepared, verbose=False)
    tr.fit(max_steps=1, resume=False)
    marker = freeze_run(tr.ckpt.dir, "test baseline")
    assert is_frozen(tr.ckpt.dir) and marker.read_text(encoding="utf-8").startswith("test baseline")
    before = tr.ckpt.last.stat().st_mtime_ns
    with pytest.raises(FrozenRunError, match="FROZEN"):
        WarpTrainer(cfg, prepared, verbose=False).fit(max_steps=2)
    with pytest.raises(FrozenRunError):
        CheckpointManager(tr.ckpt.dir).save({"x": torch.zeros(1)}, step=5)
    assert prune_snapshots(tr.ckpt.dir)["skipped"] == "frozen"
    assert tr.ckpt.last.stat().st_mtime_ns == before  # untouched


def test_class_weighted_warp_trains(tmp_path):
    cfg = cfg_for(tmp_path, "training.warp_class_weights=sqrt_median_freq")
    tr = WarpTrainer(cfg, prepare_dataset(cfg), verbose=False)
    assert tr.ce_weight is not None and tr.signature["class_weights"] == "sqrt_median_freq"
    assert float(tr.ce_weight[tr.untrained_ids].sum()) == 0  # ignored classes carry no weight
    out = tr.fit(max_steps=2, resume=False)
    assert out is not None and tr.step == 2
    base = WarpTrainer(cfg_for(tmp_path / "b"), verbose=False)
    assert base.ce_weight is None and "class_weights" not in base.signature


def test_real_pair_sampling_share(tmp_path):
    def share(p):
        cfg = cfg_for(tmp_path, f"training.texture_real_pair_prob={p}")
        ds = TextureDataset(prepare_dataset(cfg), "train", cfg, train=True)
        assert ds.partnered
        modes = [ds[i]["mode"] for i in range(len(ds))]
        return np.mean([m == "cross_view" for m in modes])
    base, more = share(0.0), share(0.9)
    assert more > base and more >= 0.85
    cfg = cfg_for(tmp_path, "training.texture_real_pair_prob=0.9")
    val = TextureDataset(prepare_dataset(cfg), "val", cfg, train=False)
    assert val.p_real == 0.0  # validation sampling never changes
    TextureTrainer(cfg, verbose=False).fit(max_steps=1, resume=False)


def _fake_results(shift):
    rng = np.random.default_rng(0)
    pairs = [[f"s{i}", f"t{i}"] for i in range(50)]
    pp = {"pairs": pairs}
    for v in ("gt_layout", "e2e_raw", "auto", "copy"):
        s = rng.uniform(0.3, 0.6, 50)
        pp[v] = {"ssim": list(s + (shift if v != "copy" else 0)), "l1": list(1 - s), "lpips": list(1 - s)}
    return {"split": "val", "started": "x", "checkpoints": {}, "paired": {"per_pair": pp},
            "warp": {"mean_iou_7": 0.7 + shift, "class_area_agreement": 0.88, "class_fidelity": 0.98,
                     "per_class_iou": {"top": 0.78, "outer": 0.55 + shift}},
            "realism": {"fid": 23.7, "kid": 0.0074}}


def test_compare_results():
    base, run = _fake_results(0.0), _fake_results(0.03)
    run["paired"]["per_pair"]["pairs"] = run["paired"]["per_pair"]["pairs"][::-1]  # order must not matter
    for v in ("gt_layout", "e2e_raw", "auto", "copy"):
        for m in ("ssim", "l1", "lpips"):
            run["paired"]["per_pair"][v][m] = run["paired"]["per_pair"][v][m][::-1]
    c = compare_results(base, run)
    rows = {(r["variant"], r["metric"]): r for r in c["paired"]}
    assert c["paired_common_pairs"] == 50
    assert rows[("e2e_raw", "ssim")]["verdict"] == "better" and abs(rows[("e2e_raw", "ssim")]["mean_diff"] - 0.03) < 1e-9
    assert rows[("copy", "ssim")]["mean_diff"] == 0
    assert abs(c["warp"]["per_class"]["outer"]["delta"] - 0.03) < 1e-9
    assert "| T5 end-to-end" in format_comparison(c, "exp_test")
    with pytest.raises(ValueError):
        compare_results(base, {**run, "split": "test"})
