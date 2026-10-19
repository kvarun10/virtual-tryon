"""Evaluate a clothing parser against DFMM manual parsing (Experiment A baseline)."""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from src.config import ProjectConfig
from src.data.dataset import PreparedData
from src.data.labels import convert_labels, get_label_set
from src.data.preprocessing import load_image, load_label_map
from src.evaluation.metrics import ConfusionMatrix
from src.models.segmentation import SegmentationBackend, to_project_labels


def evaluate_clothing_parser(
    backend: SegmentationBackend,
    prepared: PreparedData,
    cfg: ProjectConfig,
    split: str = "val",
    limit: Optional[int] = 50,
    progress: bool = True,
) -> Dict:
    """Compare ``backend`` with manual DFMM parsing in garment12 space at full image resolution.

    Background is included in pixel accuracy but excluded from mIoU (it dominates the image).
    """
    if prepared.synthetic:
        print("WARNING: synthetic fixture - these numbers only test the code path.")
    ls = get_label_set(cfg.labels.garment_label_set)
    cm = ConfusionMatrix(ls.num_classes)
    records = [r for r in prepared.records(split) if r.parsing][: limit or None]
    if not records:
        raise ValueError(f"No images with manual parsing in split '{split}'")
    it = records
    if progress:
        from tqdm.auto import tqdm

        it = tqdm(records, desc=f"eval {backend.name}")
    t0 = time.time()
    for rec in it:
        image = load_image(prepared.index.root / rec.image)
        gt = convert_labels(load_label_map(prepared.index.root / rec.parsing), "dfmm", ls.name)
        pred = to_project_labels(backend.predict(image, key=rec.name), ls.name)
        cm.update(pred, gt)
    out = cm.summary(ls.names, exclude=[ls.index("background")])
    out.update({"backend": backend.name, "split": split, "n_images": len(records),
                "seconds_per_image": (time.time() - t0) / len(records), "synthetic": prepared.synthetic,
                "reference": "DFMM manual parsing (mapped to garment12)"})
    return out
