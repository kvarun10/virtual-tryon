"""Phase 7 final-evaluation runner on the SYNTHETIC fixture with tiny 2-step models (wiring only, no downloads:
LPIPS uses a random AlexNet, FID/KID a mean-colour feature extractor)."""

import json

import numpy as np
import pytest

from src.evaluation.final_eval import (PARTS, TestSplitLockedError, format_results_table, paired_bootstrap_ci,
                                       read_test_log, run_final_evaluation)
from tests.test_tryon import trained  # noqa: F401  (module fixture: tiny trained models)


def _extractor(x):
    return x.mean((2, 3)).repeat(1, 4) + 1e-3 * x.std((2, 3)).repeat(1, 4)


def test_test_split_is_locked(trained):
    cfg, prepared, models = trained
    with pytest.raises(TestSplitLockedError, match="confirm_test"):
        run_final_evaluation(cfg, prepared, models, split="test", parts=["warp"])
    with pytest.raises(ValueError, match="Unknown parts"):
        run_final_evaluation(cfg, prepared, models, split="val", parts=["fid"])


def test_full_run_writes_results_and_audit_log(trained, tmp_path):
    cfg, prepared, models = trained
    before = len(read_test_log(cfg))
    res = run_final_evaluation(cfg, prepared, models, split="train", parts=PARTS, out_dir=tmp_path / "run",
                               limit=4, lpips_random_backbone=True, feature_extractor=_extractor, progress=False)
    assert not res["official"] and len(read_test_log(cfg)) == before  # non-test runs are not logged
    w, p, t = res["warp"], res["paired"], res["timing"]
    assert 0 <= w["mean_iou_7"] <= 1 and w["n_images"] == 4
    for v in ("gt_layout", "e2e_raw", "e2e", "auto_raw", "auto", "copy"):
        for m in ("ssim", "l1", "lpips"):
            assert np.isfinite(p[f"{v}_{m}"]), (v, m)
    assert p["copy_lpips"] >= 0 and np.isfinite(res["realism"]["fid"]) and np.isfinite(res["noise_floor"]["kid"])
    assert t["total_ms"] >= t["warp_ms"] > 0 and t["runs"] == 4
    assert np.isfinite(res["parser"]["mean_iou"])
    pp = p["per_pair"]
    assert len(pp["pairs"]) == p["pairs"] == len(pp["e2e"]["ssim"]) == len(pp["auto"]["lpips"])
    saved = json.loads((tmp_path / "run" / "results.json").read_text(encoding="utf-8"))
    assert saved["finished"] and set(PARTS) <= set(saved)
    table = (tmp_path / "run" / "results_table.md").read_text(encoding="utf-8")
    assert "NOT official" in table and "T6 fully automatic" in table and "T10 Cost" in table
    assert (tmp_path / "run" / "paired_grid.png").is_file() and (tmp_path / "run" / "realism_grid.png").is_file()


def test_confirmed_test_run_is_logged(trained, tmp_path):
    cfg, prepared, models = trained
    before = [e for e in read_test_log(cfg) if e["event"] == "start"]
    res = run_final_evaluation(cfg, prepared, models, split="test", confirm_test=True, parts=["warp", "timing"],
                               out_dir=tmp_path / "t", limit=2, progress=False)
    log = read_test_log(cfg)
    starts = [e for e in log if e["event"] == "start"]
    assert len(starts) == len(before) + 1 and log[-1]["event"] == "finished"
    assert res["previous_test_runs"] == len(before) and starts[-1]["official"] is False
    assert "T2/T3 Warp" in format_results_table(res)


def test_paired_bootstrap_ci():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 300)
    better = paired_bootstrap_ci(a, a + 0.02 + rng.normal(0, 0.01, 300))
    assert better["ci_low"] > 0 and abs(better["mean_diff"] - 0.02) < 0.005
    same = paired_bootstrap_ci(a, a + rng.normal(0, 0.01, 300))
    assert same["ci_low"] < 0 < same["ci_high"]
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [2.0])
