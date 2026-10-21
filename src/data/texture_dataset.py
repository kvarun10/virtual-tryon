"""Training pairs for the texture stage (Stage C).

Each sample asks: "render photo t, given its garment layout and a photo s of the same garment".

* ``cross_view`` (probability ``texture_cross_view_prob``, needs a second photo of the same item):
  exemplar = photo s, its parsing; layout + ground truth = photo t. Real pose change, as at try-on time.
* ``pseudo`` (CoCosNet): exemplar = photo t after a random affine distortion and optional flip
  (image and parsing transformed together); layout + ground truth = undistorted photo t.

Returned tensors (H x W = data.image_size, letterboxed, white padding):
  ``layout``           float [12, H, W] garment12 one-hot of photo t (DFMM manual parsing)
  ``image``            float [3, H, W]  photo t in [-1, 1]   (ground truth)
  ``exemplar``         float [3, H, W]  photo s (or distorted t) in [-1, 1]
  ``exemplar_parsing`` float [12, H, W] garment12 one-hot of the exemplar
Sample ``i`` in epoch ``e`` uses ``default_rng([seed, e, i])``; validation is fixed per index.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import ProjectConfig
from src.data.dataset import PreparedData
from src.data.labels import convert_labels, get_label_set
from src.data.pair_quality import PairFilter
from src.data.preprocessing import image_to_tensor, load_image, load_label_map, resize_image, resize_labels
from src.data.transforms import apply_joint_affine, sample_affine
from src.data.warp_dataset import one_hot_t


class TextureDataset(Dataset):
    def __init__(self, prepared: PreparedData, split: str, cfg: ProjectConfig, train: bool,
                 cross_view_prob: Optional[float] = None):
        self.records = [r for r in prepared.records(split) if r.parsing]
        if not self.records:
            raise ValueError(f"No annotated images in split '{split}'")
        self.cfg, self.train = cfg, train
        self.root = prepared.index.root
        self.size = tuple(cfg.data.image_size)
        self.mode = cfg.data.resize_mode
        self.pad = cfg.data.pad_value_image
        self.G = get_label_set(cfg.labels.garment_label_set)
        self.p_cross = cfg.training.texture_cross_view_prob if cross_view_prob is None else cross_view_prob
        by_item: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(self.records):
            if r.item_id:
                by_item[r.item_id].append(i)
        pf = PairFilter(cfg.data.pair_filter, cfg.data.pair_table, cfg.data.pair_max_delta_e)
        self.siblings = {i: [j for j in by_item[r.item_id] if j != i and pf.ok(r.name, self.records[j].name)] if r.item_id else []
                         for i, r in enumerate(self.records)}
        self.seed, self.epoch = cfg.experiment.seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def n_cross_view_available(self) -> int:
        return sum(1 for s in self.siblings.values() if s)

    def _load(self, rec):
        raw = load_image(self.root / rec.image)
        lab = convert_labels(load_label_map(self.root / rec.parsing), "dfmm", self.G.name)
        if lab.shape != raw.shape[:2]:
            raise ValueError(f"{rec.name}: parsing size {lab.shape} != image size {raw.shape[:2]}")
        return resize_image(raw, self.size, self.mode, self.pad)[0], resize_labels(lab, self.size, self.mode)[0]

    def pair(self, i_target: int, i_exemplar: Optional[int], rng: np.random.Generator) -> Dict:
        """``i_exemplar=None`` builds a pseudo pair from the target itself."""
        img_t, lab_t = self._load(self.records[i_target])
        if i_exemplar is None:
            params = sample_affine(rng, max_angle=10, scale_range=(0.9, 1.1), max_translate=0.1, hflip_prob=0.5)
            img_s, labs = apply_joint_affine(img_t, {"p": lab_t}, params, image_fill=(self.pad,) * 3, label_fill=0)
            lab_s, mode, src = labs["p"], "pseudo", self.records[i_target].name
        else:
            img_s, lab_s = self._load(self.records[i_exemplar])
            mode, src = "cross_view", self.records[i_exemplar].name
        K = self.G.num_classes
        return {"layout": one_hot_t(lab_t, K), "image": image_to_tensor(img_t), "exemplar": image_to_tensor(img_s),
                "exemplar_parsing": one_hot_t(lab_s, K), "mode": mode, "src": src, "tgt": self.records[i_target].name}

    def __getitem__(self, i: int) -> Dict:
        rng = np.random.default_rng([self.seed, self.epoch if self.train else 10_000, i])
        sib = self.siblings[i]
        use_cross = bool(sib) and rng.random() < self.p_cross
        j = int(sib[rng.integers(0, len(sib))]) if use_cross else None
        return self.pair(i, j, rng)


class TextureSwapDataset(Dataset):
    """Try-on situation for visual checks: exemplar from one item, layout from a different item (no ground truth)."""

    def __init__(self, base: TextureDataset, n: int = 8, seed: int = 0):
        self.base = base
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(base.records))[: 2 * n]
        self.pairs = [(int(a), int(b)) for a, b in zip(idx[:n], idx[n:])
                      if base.records[a].item_id != base.records[b].item_id]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, k: int) -> Dict:
        target, exemplar = self.pairs[k]
        out = self.base.pair(target, exemplar, np.random.default_rng(k))
        out["mode"] = "swap"
        return out


def texture_collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ("layout", "image", "exemplar", "exemplar_parsing")}
    for k in ("mode", "src", "tgt"):
        out[k] = [b[k] for b in batch]
    return out
