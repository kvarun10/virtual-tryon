import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data.dataset import DatasetUnavailableError, PreTreatmentDataset, make_loader, prepare_dataset
from src.data.deepfashion_mm import DFMMIndex, build_same_item_pairs, parse_name
from src.data.splits import SplitError, check_disjoint, group_disjoint_split, load_splits
from src.data.synthetic import FixtureSpec, make_fixture


def test_parse_inshop_names():
    m = parse_name("WOMEN-Blouses_Shirts-id_00000001-02_1_front")
    assert m == {"gender": "WOMEN", "category": "Blouses_Shirts", "item": "00000001", "seq": "02", "view": "1_front"}
    assert parse_name("MEN-Denim-id_00000080-01_7_additional")["item"] == "00000080"
    assert parse_name("random_photo") is None


def test_split_is_group_disjoint_reproducible_and_balanced():
    rng = np.random.default_rng(0)
    items = [(f"img{i}", f"item{rng.integers(0, 300)}") for i in range(3000)]
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    a = group_disjoint_split(items, ratios, seed=3)
    b = group_disjoint_split(list(reversed(items)), ratios, seed=3)  # input order must not matter
    assert a == b
    assert a != group_disjoint_split(items, ratios, seed=4)
    check_disjoint(a, dict(items))
    for split, r in ratios.items():
        assert abs(len(a[split]) / 3000 - r) < 0.03


def test_check_disjoint_detects_leak():
    with pytest.raises(SplitError, match="leaks"):
        check_disjoint({"train": ["a"], "val": ["b"], "test": []}, {"a": "g1", "b": "g1"})


def test_index_validation_reports_faults(tmp_path):
    root = tmp_path / "fx"
    make_fixture(root, FixtureSpec(n_items=4, views_per_item=3, corrupt_images=1, drop_densepose=1, orphan_parsing=2, parsing_fraction=1.0))
    cfg = load_config(overrides=[f"data.root={root.as_posix()}"])
    index, report = DFMMIndex.build(root, cfg.data.layout)
    report = index.validate(report, image_check="full", annotation_sample=None)
    assert report.n_images == 12
    assert len(report.corrupt_images) == 1
    assert report.n_with_densepose == 11
    assert len(report.orphan_parsing) == 2
    assert not report.bad_parsing and not report.bad_densepose
    assert report.size_counts == {"150x220": 11}
    stats = DFMMIndex.statistics(index.records)
    assert stats["items"] == 4 and stats["items_with_2plus_images"] == 4


def test_missing_dataset_gives_instructions(tmp_path):
    cfg = load_config(overrides=[f"data.root={(tmp_path / 'nope').as_posix()}", f"paths.drive_root={tmp_path.as_posix()}"])
    with pytest.raises(DatasetUnavailableError, match="archive_dir"):
        prepare_dataset(cfg)


def test_wrong_layout_gives_folder_listing(tmp_path, fixture_root):
    cfg = load_config(overrides=[f"data.root={fixture_root.as_posix()}", "data.layout.images_dir=img",
                                 f"paths.drive_root={tmp_path.as_posix()}"])
    with pytest.raises(FileNotFoundError, match="images_dir"):
        prepare_dataset(cfg)


def test_prepare_and_dataset_items(smoke_cfg):
    prepared = prepare_dataset(smoke_cfg)
    assert prepared.synthetic
    group = {r.name: r.group for r in prepared.index.records}
    check_disjoint(prepared.splits, group)
    # split file is reused (and fingerprint-checked) on the next call
    splits2, meta = load_splits(smoke_cfg.data.split_file)
    assert meta["synthetic"] is True
    ds = PreTreatmentDataset(prepared, "train", smoke_cfg)
    item = ds[0]
    H, W = smoke_cfg.data.image_size
    assert item["image"].shape == (3, H, W) and item["image"].dtype == torch.float32
    assert item["garment"].shape == (H, W) and item["garment"].dtype == torch.int64
    assert item["body"].shape == (H, W) and int(item["body"].max()) < 15
    assert int(item["garment"].max()) < 12
    assert item["valid"].sum() < H * W  # letterbox padding is marked invalid
    assert (item["garment"][~item["valid"]] == 0).all() and (item["body"][~item["valid"]] == 0).all()
    batch = next(iter(make_loader(ds, smoke_cfg, shuffle=False, batch_size=2)))
    assert batch["image"].shape == (2, 3, H, W)
    report_path = Path(smoke_cfg.paths.cache_dir) / "reports" / "synthetic_data_report.json"
    assert json.loads(report_path.read_text())["synthetic"] is True


def test_body_label_set_pascal(smoke_cfg, tmp_path):
    cfg = load_config(overrides=[f"paths.drive_root={tmp_path.as_posix()}/d2", "data.dataset=synthetic",
                                 "data.root={cache_dir}/fx", "labels.body_label_set=pascal_person_part"])
    prepared = prepare_dataset(cfg)
    item = PreTreatmentDataset(prepared, "train", cfg)[0]
    assert int(item["body"].max()) < 7


def test_same_item_pairs_stay_in_split(smoke_cfg):
    prepared = prepare_dataset(smoke_cfg)
    for split in ("train", "val", "test"):
        recs = prepared.records(split)
        names = {r.name for r in recs}
        for a, b in build_same_item_pairs(recs):
            assert a in names and b in names and a != b
            assert prepared.index[a].item_id == prepared.index[b].item_id
