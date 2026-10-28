"""Training triplets for the warp network: (C_seg input, B_seg, C'_seg target).

Two ways to build a triplet (``training.warp_pairing``):

* ``self`` (paper Sec. 5.1, following SwapNet): one photo provides everything. The target is its
  garment map; the input C_seg is the *same* map with an independent random affine transform per
  garment class, so the network cannot copy positions and has to use B_seg for placement.
* ``cross_view`` (our addition): two photos of the *same item* (DeepFashion item ID). Input C_seg
  comes from photo s, B_seg and the target from photo t, which gives real pose-change supervision.
  Items with a single annotated photo fall back to ``self``.
* ``mixed``: cross-view with probability ``warp_cross_view_prob`` when available, otherwise self.

Ground truth is DFMM manual parsing mapped to garment12. Target classes that the warp cannot infer
from its inputs are set to ``IGNORE`` (no loss): hair and face (replaced by the target's own parse,
paper Sec. 4.2), plus headwear and accessories, which are not transferred. ``skin`` stays a target
class: uncovered body regions are predictable from B_seg.

Randomness is reproducible: sample ``i`` in epoch ``e`` uses ``default_rng([seed, e, i])``.
Validation uses a fixed RNG per index and always applies the affine to self pairs, so a self
triplet is never the trivial identity.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from src.config import ProjectConfig
from src.data.body_maps import BodyMapSource
from src.data.dataset import PreparedData
from src.data.labels import convert_labels, get_label_set, keep_classes
from src.data.pair_quality import PairFilter
from src.data.preprocessing import load_label_map, resize_labels
from src.data.transforms import AffineParams, _apply, sample_affine
from PIL import Image

IGNORE = 255


def per_class_affine(label_map: np.ndarray, num_classes: int, rng: np.random.Generator, max_angle: float,
                     scale_range, max_translate: float) -> np.ndarray:
    """SwapNet-style augmentation: one independent random affine per garment class.

    Returns a ``num_classes x H x W`` float32 one-hot-like tensor. Transformed classes may overlap
    (as in SwapNet's per-channel augmentation). Background = pixels covered by no class.
    """
    h, w = label_map.shape
    out = np.zeros((num_classes, h, w), dtype=np.float32)
    for k in np.unique(label_map):
        if k == 0:
            continue
        params = sample_affine(rng, max_angle, tuple(scale_range), max_translate)
        mask = (label_map == k).astype(np.uint8) * 255
        out[k] = _apply(mask, params, Image.NEAREST, 0) > 127
    out[0] = out[1:].max(axis=0) == 0
    return out


def jitter_body_parts(body: np.ndarray, rng: np.random.Generator, max_px: int) -> np.ndarray:
    """Randomly dilate or erode each body part by up to ``max_px`` pixels.

    Weakens the "copy the clothed silhouette" shortcut (docs/body_representation.md). Dilation only
    grows into background; erosion turns removed pixels into background.
    """
    if max_px < 1:
        return body
    out = body.copy()
    for part in np.unique(body):
        if part == 0:
            continue
        r = int(rng.integers(1, max_px + 1))
        op = rng.integers(0, 3)  # 0 keep, 1 dilate, 2 erode
        mask = body == part
        if op == 1:
            grown = ndimage.binary_dilation(mask, iterations=r)
            out[grown & (out == 0)] = part
        elif op == 2:
            shrunk = ndimage.binary_erosion(mask, iterations=r)
            out[mask & ~shrunk] = 0
    return out


def one_hot_t(label_map: np.ndarray, num_classes: int) -> torch.Tensor:
    t = torch.from_numpy(label_map.astype(np.int64))
    return torch.nn.functional.one_hot(t, num_classes).permute(2, 0, 1).float()


class WarpTripletDataset(Dataset):
    """Each item is a dict:

    ``cloth``  float [Kg, H, W]  input C_seg (one-hot-like, transfer classes only, Kg = 12)
    ``body``   float [Kb, H, W]  B_seg one-hot (Kb = 7 for pascal_person_part, 15 for body15)
    ``target`` int64 [H, W]      C'_seg ground truth in garment12, IGNORE where not supervised
    ``mode``   "self" | "cross_view";  ``src`` / ``tgt`` image names
    """

    def __init__(self, prepared: PreparedData, split: str, cfg: ProjectConfig, train: bool,
                 body_source: Optional[BodyMapSource] = None, pairing: Optional[str] = None):
        store = getattr(prepared, "store", None)
        self.extra = cfg.model.warp_extra_body
        self.ref_body = cfg.model.warp_ref_body
        if self.extra != "none" and store is None:
            raise ValueError("model.warp_extra_body needs the packed store (data.packed_dir)")
        self.records = [r for r in prepared.records(split) if r.parsing and (store is None or store.has_body(r.name))
                        and (self.extra == "none" or store.has_dp(r.name))]
        if not self.records:
            raise ValueError(f"No annotated images in split '{split}'")
        self.cfg, self.t = cfg, cfg.training
        self.train = train
        self.pairing = pairing or self.t.warp_pairing
        self.root = prepared.index.root
        self.size = tuple(cfg.data.image_size)
        self.mode = cfg.data.resize_mode
        self.G = get_label_set(cfg.labels.garment_label_set)
        self.store = getattr(prepared, "store", None)  # packed 256 px store (src/data/packed.py) or None
        if self.store is not None:
            self.body = None
            self.Kb = get_label_set(cfg.labels.body_label_set).num_classes
        else:
            self.body = body_source or BodyMapSource(cfg, self.root)
            self.Kb = self.body.num_classes
        self.Kb_base = self.Kb
        if self.extra != "none":
            self.Kx = get_label_set("body15").num_classes
            self.Kb = self.Kb_base + self.Kx
        self.transfer = cfg.labels.transfer_classes
        supervised = set(self.transfer) | {"background", "skin"}
        self.ignore_ids = [i for i, n in enumerate(self.G.names) if n not in supervised]
        by_item: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(self.records):
            if r.item_id:
                by_item[r.item_id].append(i)
        pf = PairFilter(cfg.data.pair_filter, cfg.data.pair_table, cfg.data.pair_max_delta_e, cfg.data.pair_color_metric)
        self.siblings = {i: [j for j in by_item[r.item_id] if j != i and pf.ok(r.name, self.records[j].name)] if r.item_id else []
                         for i, r in enumerate(self.records)}
        self.epoch = 0
        self.seed = cfg.experiment.seed
        self.body_aug_px = max(0, round(self.t.warp_body_aug_max_px * self.size[0] / 128)) if self.t.warp_body_aug else 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def n_cross_view_available(self) -> int:
        return sum(1 for s in self.siblings.values() if s)

    def _fit(self, lab: np.ndarray) -> np.ndarray:
        return lab if lab.shape == self.size else resize_labels(lab, self.size, self.mode)[0]

    def _garment(self, rec) -> np.ndarray:
        if self.store is not None:
            return self._fit(self.store.garment(rec.name))
        g = convert_labels(load_label_map(self.root / rec.parsing), "dfmm", self.G.name)
        return resize_labels(g, self.size, self.mode)[0]

    def _body(self, rec) -> np.ndarray:
        if self.store is not None:
            return self._fit(self.store.body(rec.name))
        return resize_labels(self.body.load(rec), self.size, self.mode)[0]

    def _body_tensor(self, rec, rng=None, augment: bool = False) -> torch.Tensor:
        """Body encoder input: one-hot body map (+ one-hot DensePose parts when model.warp_extra_body)."""
        body_map = self._body(rec)
        if augment:
            body_map = jitter_body_parts(body_map, rng, self.body_aug_px)
        t = one_hot_t(body_map, self.Kb_base)
        if self.extra != "none":
            dp = self._fit(self.store.body_dp(rec.name))
            if augment:
                dp = jitter_body_parts(dp, rng, self.body_aug_px)
            t = torch.cat([t, one_hot_t(dp, self.Kx)], 0)
        return t

    def _rng(self, i: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, self.epoch if self.train else 10_000, i])

    def _choose_source(self, i: int, rng) -> int:
        sib = self.siblings[i]
        if not sib or self.pairing == "self":
            return i
        if self.pairing == "cross_view" or rng.random() < self.t.warp_cross_view_prob:
            return int(sib[rng.integers(0, len(sib))])
        return i

    def build(self, i_body: int, i_cloth: int, rng, augment_cloth: bool, augment_body: bool) -> Dict:
        tgt, src = self.records[i_body], self.records[i_cloth]
        cloth_map = keep_classes(self._garment(src), self.G, self.transfer)
        if augment_cloth:
            t = self.t
            cloth = torch.from_numpy(per_class_affine(cloth_map, self.G.num_classes, rng, t.warp_affine_max_angle,
                                                      t.warp_affine_scale, t.warp_affine_max_translate))
        else:
            cloth = one_hot_t(cloth_map, self.G.num_classes)
        body = self._body_tensor(tgt, rng, augment_body)
        if self.ref_body:  # reference body maps for the clothes encoder; zeros for self pairs (garments displaced)
            ref = torch.zeros_like(body) if i_body == i_cloth else self._body_tensor(src)
            cloth = torch.cat([cloth, ref], 0)
        target = self._garment(tgt).astype(np.int64)
        target[np.isin(target, self.ignore_ids)] = IGNORE
        return {"cloth": cloth, "body": body, "target": torch.from_numpy(target),
                "mode": "self" if i_body == i_cloth else "cross_view", "src": src.name, "tgt": tgt.name}

    def __getitem__(self, i: int) -> Dict:
        rng = self._rng(i)
        j = self._choose_source(i, rng)
        is_self = j == i
        # Self pairs always need the affine (else input == target); cross-view gets it only in training.
        aug_cloth = self.t.warp_cloth_affine and (is_self or self.train)
        if is_self and not self.t.warp_cloth_affine:
            raise ValueError("Self-supervised pairs require training.warp_cloth_affine=true (input would equal target)")
        return self.build(i, j, rng, aug_cloth, self.train and self.body_aug_px > 0)


class SwapDiagnosticDataset(Dataset):
    """C_seg from one item, B_seg from a *different* item (no ground truth).

    Used to measure ``class_fidelity``: does the predicted garment follow the input garment (good) or
    the target person's own clothed silhouette (shortcut)?
    """

    def __init__(self, base: WarpTripletDataset, n: int = 64, seed: int = 0):
        self.base = base
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(base.records))[: n * 2]
        pairs = []
        for a, b in zip(idx[: len(idx) // 2], idx[len(idx) // 2:]):
            if base.records[a].item_id != base.records[b].item_id:
                pairs.append((int(a), int(b)))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, k: int) -> Dict:
        a, b = self.pairs[k]
        out = self.base.build(i_body=b, i_cloth=a, rng=np.random.default_rng(k), augment_cloth=False, augment_body=False)
        out["mode"] = "swap"
        return out
