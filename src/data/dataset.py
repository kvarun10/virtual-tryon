"""Dataset preparation (index -> validation -> splits) and PyTorch datasets.

Typical use (also what ``scripts/prepare_dataset.py`` and notebook Section 6 do)::

    cfg = load_config("configs/colab_baseline.yaml")
    prepared = prepare_dataset(cfg)            # index, validate, split, write reports
    ds = PreTreatmentDataset(prepared, "train", cfg)
    sample = ds[0]                              # dict of tensors, see PreTreatmentDataset

The "pre-treatment" dataset yields the annotations Stage A would produce, taken from the
dataset's ground truth: C_seg-style garment maps (from DFMM manual parsing) and B_seg-style body
maps (from DFMM DensePose). Warp/texture datasets built on top of it belong to Phases 4-5.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import ProjectConfig
from src.data.deepfashion_mm import DFMMIndex, ImageRecord, ValidationReport, inspect_root
from src.data.labels import convert_labels
from src.data.preprocessing import image_to_tensor, labels_to_tensor, load_image, load_label_map, resize_image, resize_labels
from src.data.splits import check_disjoint, group_disjoint_split, load_splits, save_splits
from src.data.synthetic import FixtureSpec, is_synthetic_root, make_fixture
from src.utils.reproducibility import make_generator, seed_worker


class DatasetUnavailableError(FileNotFoundError):
    """Raised with download instructions when the dataset is missing."""


DFMM_INSTRUCTIONS = (
    "DeepFashion-MultiModal was not found at '{root}' (the extracted working copy).\n"
    "On Colab, run notebook Section 6a or `python scripts/fetch_dfmm.py --config <cfg>`: it puts the zip "
    "archives in Google Drive (data.archive_dir, once) and extracts them to data.root (every session).\n"
    "Licence: non-commercial research only - https://github.com/yumingj/DeepFashion-MultiModal#agreement"
)


@dataclass
class PreparedData:
    index: DFMMIndex
    report: ValidationReport
    splits: Dict[str, List[str]]
    stats: Dict[str, Dict]
    synthetic: bool

    def records(self, split: str) -> List[ImageRecord]:
        return [self.index[n] for n in self.splits[split]]


def ensure_synthetic_fixture(root: Path, spec: Optional[FixtureSpec] = None) -> Path:
    root = Path(root)
    if not is_synthetic_root(root):
        make_fixture(root, spec)
    return root


def prepare_dataset(
    cfg: ProjectConfig,
    rebuild_index: bool = False,
    rebuild_splits: bool = False,
    annotation_sample: Optional[int] = 200,
    progress: bool = False,
    write_reports: bool = True,
) -> PreparedData:
    root = Path(cfg.data.root)
    synthetic = cfg.data.dataset == "synthetic"
    if synthetic:
        ensure_synthetic_fixture(root)
    elif not root.is_dir():
        raise DatasetUnavailableError(DFMM_INSTRUCTIONS.format(root=root))

    index, report = DFMMIndex.build(root, cfg.data.layout)
    if len(index) == 0:
        raise DatasetUnavailableError(
            f"No '*{cfg.data.layout.image_ext}' images in {root / cfg.data.layout.images_dir}. "
            f"Folder overview: {json.dumps(inspect_root(root))[:800]}"
        )
    report = index.validate(report, cfg.data.image_check, annotation_sample, seed=cfg.experiment.seed, progress=progress)

    excluded = set(report.corrupt_images) | set(report.bad_parsing) | set(report.bad_densepose)
    usable = index.filter(require_parsing=cfg.data.require_parsing, require_densepose=cfg.data.require_densepose,
                          exclude=excluded)
    if not usable:
        raise DatasetUnavailableError(
            f"No usable images after filtering (require_parsing={cfg.data.require_parsing}, "
            f"require_densepose={cfg.data.require_densepose}). Without the DensePose archive set "
            "data.require_densepose=false.\n" + report.summary()
        )
    names = [r.name for r in usable]
    split_path = Path(cfg.data.split_file)
    if split_path.is_file() and not rebuild_splits:
        splits, _ = load_splits(split_path, expected_names=names)
    else:
        splits = group_disjoint_split([(r.name, r.group) for r in usable], cfg.data.split_ratios, cfg.data.split_seed)
        save_splits(
            split_path,
            splits,
            {"dataset": cfg.data.dataset, "seed": cfg.data.split_seed, "ratios": cfg.data.split_ratios,
             "group_key": "item_id", "require_parsing": cfg.data.require_parsing, "synthetic": synthetic},
        )
    check_disjoint(splits, {r.name: r.group for r in usable})
    if cfg.data.max_samples:
        splits = {k: v[: cfg.data.max_samples] for k, v in splits.items()}
    stats = {"all": DFMMIndex.statistics(index.records), "usable": DFMMIndex.statistics(usable)}
    for split, split_names in splits.items():
        stats[split] = DFMMIndex.statistics([index[n] for n in split_names])

    if write_reports:
        index.save(Path(cfg.data.index_file))
        report_path = Path(cfg.paths.cache_dir) / "reports" / f"{cfg.data.dataset}_data_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump({"validation": asdict(report), "stats": stats, "synthetic": synthetic}, fh, indent=1)
    return PreparedData(index, report, splits, stats, synthetic)


class PreTreatmentDataset(Dataset):
    """Images with ground-truth Stage-A representations.

    Returns a dict:
        ``image``   float32 [3, H, W] in [-1, 1]   (letterboxed to data.image_size)
        ``garment`` int64 [H, W]  garment12 labels (C_seg-style, from manual parsing)
        ``body``    int64 [H, W]  body15 labels (B_seg-style, from DensePose)
        ``valid``   bool  [H, W]  True inside the original image area (False on padding)
        ``name``, ``item_id`` strings
    """

    def __init__(self, prepared: PreparedData, split: str, cfg: ProjectConfig):
        if split not in prepared.splits:
            raise KeyError(f"Unknown split '{split}'")
        self.records = prepared.records(split)
        self.root = prepared.index.root
        self.cfg = cfg
        self.size = tuple(cfg.data.image_size)
        self.mode = cfg.data.resize_mode
        self.dp_channel = cfg.data.layout.densepose_part_channel
        if not self.records:
            raise ValueError(f"Split '{split}' is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Dict:
        rec = self.records[i]
        img = load_image(self.root / rec.image)
        if rec.parsing is None:
            raise FileNotFoundError(f"{rec.name}: no parsing file (set data.require_parsing=true)")
        if rec.densepose is None:
            raise FileNotFoundError(
                f"{rec.name}: no DensePose file, so no body map can be built. Stage the 'densepose' archive "
                "(scripts/fetch_dfmm.py --parts densepose) or wait for the Phase 4 SCHP-based body maps."
            )
        garment = convert_labels(load_label_map(self.root / rec.parsing), "dfmm", self.cfg.labels.garment_label_set)
        body = convert_labels(load_label_map(self.root / rec.densepose, channel=self.dp_channel), "densepose", self.cfg.labels.body_label_set)
        for what, lab in (("parsing", garment), ("densepose", body)):
            if lab.shape != img.shape[:2]:
                raise ValueError(f"{rec.name}: {what} size {lab.shape} != image size {img.shape[:2]}")
        img_r, info = resize_image(img, self.size, self.mode, self.cfg.data.pad_value_image)
        garment_r, _ = resize_labels(garment, self.size, self.mode)
        body_r, _ = resize_labels(body, self.size, self.mode)
        valid = np.zeros(self.size, dtype=bool)
        ch, cw = info.content_size
        valid[info.pad_top : info.pad_top + ch, info.pad_left : info.pad_left + cw] = True
        return {
            "image": image_to_tensor(img_r),
            "garment": labels_to_tensor(garment_r),
            "body": labels_to_tensor(body_r),
            "valid": torch.from_numpy(valid),
            "name": rec.name,
            "item_id": rec.item_id or "",
        }


def make_loader(dataset: Dataset, cfg: ProjectConfig, shuffle: bool, batch_size: Optional[int] = None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=make_generator(cfg.experiment.seed),
        drop_last=False,
    )
