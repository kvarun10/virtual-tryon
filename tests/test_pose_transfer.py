"""Standard pose-transfer pairs: name mapping, leakage categories, protocol frame, benchmark wiring (synthetic)."""

import numpy as np
import pytest

from src.data.pose_transfer import (PairFilesUnavailableError, categorise_pairs, ensure_pair_files, leakage_report,
                                    model_output_to_protocol, to_dfmm_stem, to_protocol_frame)
from src.data.preprocessing import resize_image
from src.evaluation.image_metrics import LPIPSMetric
from src.evaluation.pose_transfer_eval import evaluate_pose_transfer, format_pose_transfer
from tests.test_tryon import trained  # noqa: F401


def test_name_mapping():
    assert to_dfmm_stem("fashionWOMENBlouses_Shirtsid0000337203_3back.jpg") == "WOMEN-Blouses_Shirts-id_00003372-03_3_back"
    assert to_dfmm_stem("fashionMENTees_Tanksid0000730104_7additional.jpg") == "MEN-Tees_Tanks-id_00007301-04_7_additional"
    with pytest.raises(ValueError):
        to_dfmm_stem("WOMEN-Dresses-id_00000002-02_1_front.jpg")


def test_pair_files_gated(tmp_path):
    with pytest.raises(PairFilesUnavailableError, match="allow_download"):
        ensure_pair_files(tmp_path)
    (tmp_path / "fasion-resize-pairs-test.csv").write_text("from,to\n")
    (tmp_path / "fasion-resize-pairs-train.csv").write_text("from,to\n")
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_pair_files(tmp_path)


def test_leakage_categories():
    s = lambda item, v: f"MEN-Tees-id_{item}-01_{v}"
    splits = {"train": [s("00000001", "1_front")], "val": [s("00000002", "1_front")], "test": [s("00000003", "1_front")]}
    pairs = [(s("00000001", "1_front"), s("00000001", "2_side")),   # trained item -> leaked
             (s("00000002", "1_front"), s("00000002", "3_back")),   # val item -> scorable
             (s("00000003", "1_front"), s("00000003", "2_side")),   # test item -> locked
             (s("00000004", "1_front"), s("00000004", "2_side")),   # never used -> scorable
             (s("00000005", "1_front"), s("00000005", "2_side"))]   # photo missing
    avail = {p for pr in pairs[:4] for p in pr}
    c = categorise_pairs(pairs, splits, avail)
    assert [c[p] for p in pairs] == ["our_train_items_LEAKED", "clean_val_items", "our_test_items_LOCKED",
                                     "clean_unseen", "image_missing"]
    r = leakage_report(pairs, pairs[:1], splits, avail)
    assert r["scorable_now"] == 2 and r["standard_train_test_item_overlap"] == 1


def test_protocol_frame_alignment():
    full = np.zeros((1101, 750, 3), np.uint8)
    full[300:800, 200:550] = 200  # a box
    gt = to_protocol_frame(full)
    assert gt.shape == (256, 176, 3)
    model, info = resize_image(full, (128, 128), "letterbox", 255)
    back = model_output_to_protocol(model, info)
    assert back.shape == gt.shape
    box_gt, box_back = (gt[..., 0] > 100), (back[..., 0] > 100)
    assert (box_gt ^ box_back).mean() < 0.02  # same geometry up to resampling


def test_benchmark_runs_on_fixture(trained, tmp_path):
    cfg, prepared, models = trained
    recs = prepared.records("train")
    by_item = {}
    for r in recs:
        by_item.setdefault(r.item_id, []).append(r.name)
    pairs = [(v[0], v[1]) for v in by_item.values() if len(v) >= 2][:3]
    cats = {p: "clean_unseen" for p in pairs}
    cats[pairs[-1]] = "our_train_items_LEAKED"  # must be skipped
    feats = __import__("src.evaluation.image_metrics", fromlist=["InceptionFeatures"]).InceptionFeatures(
        models.device, extractor=lambda x: x.mean((2, 3)).repeat(1, 4))
    res = evaluate_pose_transfer(models, cfg, pairs, cats, prepared.index.root / "images",
                                 LPIPSMetric(models.device, random_backbone=True), feats, grid_path=tmp_path / "g.png",
                                 progress=False)
    assert res["pairs_scored"] == len(pairs) - 1 and (tmp_path / "g.png").is_file()
    for k in ("ssim_raw", "ssim_pasted", "ssim_copy", "l1_raw", "lpips_raw"):
        assert np.isfinite(res[k]), k
    assert len(res["per_pair"]["ssim_raw"]) == res["pairs_scored"]
    leak = {"standard_test_pairs": 3, "pairs_by_category": {}}
    assert "ours, raw output" in format_pose_transfer(res, leak)


def test_benchmark_runs_with_a_loader_and_no_image_dir(trained):
    """Modal path: photos come from a loader (packed In-shop store), image_dir is None."""
    from src.data.preprocessing import load_image

    cfg, prepared, models = trained
    by_item = {}
    for r in prepared.records("train"):
        by_item.setdefault(r.item_id, []).append(r)
    pairs = [(v[0].name, v[1].name) for v in by_item.values() if len(v) >= 2][:2]
    paths = {r.name: prepared.index.root / r.image for r in prepared.records("train")}
    res = evaluate_pose_transfer(models, cfg, pairs, {p: "clean_unseen" for p in pairs}, None, progress=False,
                                 load=lambda stem: load_image(paths[stem]))
    assert res["pairs_scored"] == 2 and np.isfinite(res["ssim_raw"])


def test_validation_pairs_are_same_item_filtered_and_capped():
    from src.evaluation.pose_transfer_eval import validation_pairs

    names = [f"MEN-Denim-id_0000000{i}-01_{v}" for i in (1, 2) for v in ("1_front", "2_side", "3_back", "4_full")]
    pairs = validation_pairs(names, lambda a, b: "4_full" not in a + b, max_per_item=4)
    assert all(a.split("-")[2] == b.split("-")[2] and a != b for a, b in pairs)
    assert all("4_full" not in a + b for a, b in pairs)
    assert len(pairs) == 8 and pairs == validation_pairs(names, lambda a, b: "4_full" not in a + b, max_per_item=4)
