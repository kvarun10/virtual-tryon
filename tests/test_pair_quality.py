"""Pair-consistency analysis and filter (same item ID != same outfit in DeepFashion)."""

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.data.dataset import prepare_dataset
from src.data.pair_quality import (PairFilter, build_pair_table, compare_signatures, garment_signature, rgb_to_lab,
                                   summarise, write_pair_table)
from src.data.synthetic import FixtureSpec, make_fixture
from src.data.texture_dataset import TextureDataset
from src.data.warp_dataset import WarpTripletDataset

ROOT = Path(__file__).resolve().parents[1]


def test_rgb_to_lab_reference_values():
    lab = rgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], np.uint8))
    np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.05)
    np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.05)
    np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.1)  # standard sRGB red


def test_signature_comparison():
    img = np.zeros((10, 10, 3), np.uint8); img[:5] = (200, 30, 30); img[5:] = (30, 30, 200)
    lab = np.zeros((10, 10), np.uint8); lab[:5] = 4; lab[5:] = 8  # top, pants
    a = garment_signature(img, lab, [4, 8])
    same = compare_signatures(a, a)
    assert same["same_classes"] == 1.0 and same["max_delta_e"] == 0.0
    img2 = img.copy(); img2[:5] = (30, 200, 30)  # top changes colour
    diff = compare_signatures(a, garment_signature(img2, lab, [4, 8]))
    assert diff["same_classes"] == 1.0 and diff["max_delta_e"] > 50
    lab3 = lab.copy(); lab3[5:] = 7  # pants -> skirt
    cls = compare_signatures(a, garment_signature(img, lab3, [4, 7, 8]))
    assert cls["same_classes"] == 0.0 and cls["class_jaccard"] == pytest.approx(1 / 3)


def _cfg(tmp_path, *extra):
    return load_config(ROOT / "configs" / "colab_smoke_test.yaml",
                       [f"paths.drive_root={tmp_path.as_posix()}/drive", f"data.root={(tmp_path / 'fx').as_posix()}",
                        "data.max_samples=null", "data.image_size=[64,64]", *extra])


def test_analysis_detects_outfit_changes_and_filter_removes_them(tmp_path):
    make_fixture(tmp_path / "fx", FixtureSpec(n_items=6, views_per_item=3, parsing_fraction=1.0, outfit_change_items=2, seed=3))
    cfg = _cfg(tmp_path)
    prepared = prepare_dataset(cfg)
    recs = [r for s in ("train", "val", "test") for r in prepared.records(s)]
    rows = build_pair_table(recs, prepared.index.root, cfg.labels.transfer_classes, progress=False)
    changed = {f"{90000000 + i:08d}" for i in range(2)}
    bad = [r for r in rows if r["item_id"] in changed and ("1_front" in r["a"] or "1_front" in r["b"])]
    good = [r for r in rows if r["item_id"] not in changed]
    assert bad and all(r["same_classes"] == 0.0 for r in bad)  # pants -> skirt detected
    assert all(r["same_classes"] == 1.0 and r["max_delta_e"] < 10 for r in good)
    s = summarise(rows)
    assert 0 < s["share_same_classes"] < 1 and s["pairs"] == len(rows)
    write_pair_table(rows, Path(cfg.data.pair_table))
    # filter on: the changed-outfit pairs disappear from both training datasets
    fcfg = _cfg(tmp_path, "data.pair_filter=classes_color", "data.pair_max_delta_e=15")
    pf = PairFilter("classes_color", fcfg.data.pair_table, 15)
    assert not any(pf.ok(r["a"], r["b"]) for r in bad) and all(pf.ok(r["a"], r["b"]) for r in good)
    for ds in (WarpTripletDataset(prepared, "train", fcfg, train=True), TextureDataset(prepared, "train", fcfg, train=True)):
        for i, sib in ds.siblings.items():
            for j in sib:
                assert pf.ok(ds.records[i].name, ds.records[j].name)
    # filter off: baseline keeps every same-item pair
    base = WarpTripletDataset(prepared, "train", _cfg(tmp_path), train=True)
    filt = WarpTripletDataset(prepared, "train", fcfg, train=True)
    assert sum(map(len, base.siblings.values())) >= sum(map(len, filt.siblings.values()))


def test_filter_without_table_explains(tmp_path):
    with pytest.raises(FileNotFoundError, match="analyze_pairs"):
        PairFilter("classes", tmp_path / "missing.csv")
    assert PairFilter("none").ok("a", "b")
