"""Segmentation metrics (semantic MASKS only; RGB image metrics come in Phase 7).

All metrics are computed from an accumulated confusion matrix so that they are exact over a
dataset (not an average of per-image scores, which over-weights small images/classes).

* pixel accuracy = correctly labelled pixels / all evaluated pixels
* IoU_c          = TP_c / (TP_c + FP_c + FN_c)
* mIoU           = mean of IoU_c over classes that occur in the ground truth OR the prediction;
                   classes absent from both are reported as NaN and excluded.

These numbers are only meaningful against *manually annotated* masks (e.g. DFMM parsing). A
comparison against another model's output (e.g. SCHP-Pascal vs DensePose) measures agreement,
not accuracy, and must be labelled as such.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)  # rows = GT, cols = prediction

    def update(self, pred: np.ndarray, target: np.ndarray, valid: Optional[np.ndarray] = None) -> None:
        if pred.shape != target.shape:
            raise ValueError(f"prediction {pred.shape} and target {target.shape} differ")
        mask = np.ones(target.shape, dtype=bool) if valid is None else valid.astype(bool)
        if self.ignore_index is not None:
            mask &= target != self.ignore_index
        p = pred[mask].astype(np.int64)
        t = target[mask].astype(np.int64)
        if p.size and (p.max() >= self.num_classes or t.max() >= self.num_classes):
            raise ValueError("label value >= num_classes")
        self.matrix += np.bincount(t * self.num_classes + p, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def pixel_accuracy(self) -> float:
        total = self.matrix.sum()
        return float(np.trace(self.matrix) / total) if total else float("nan")

    def iou(self) -> np.ndarray:
        tp = np.diag(self.matrix).astype(np.float64)
        denom = self.matrix.sum(0) + self.matrix.sum(1) - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(denom > 0, tp / denom, np.nan)

    def mean_iou(self, exclude: Sequence[int] = ()) -> float:
        iou = self.iou()
        keep = [i for i in range(self.num_classes) if i not in set(exclude) and not np.isnan(iou[i])]
        return float(np.mean(iou[keep])) if keep else float("nan")

    def summary(self, class_names: Sequence[str], exclude: Sequence[int] = ()) -> Dict:
        iou = self.iou()
        return {
            "pixel_accuracy": self.pixel_accuracy(),
            "mean_iou": self.mean_iou(exclude),
            "per_class_iou": {n: (None if np.isnan(v) else float(v)) for n, v in zip(class_names, iou)},
            "gt_pixels": {n: int(c) for n, c in zip(class_names, self.matrix.sum(1))},
            "excluded_from_mean": [class_names[i] for i in exclude],
        }


# ======================================================================================
# Image metrics (RGB) - valid only when a paired ground-truth image exists
# ======================================================================================

def _gaussian_window(size: int = 11, sigma: float = 1.5):
    import torch

    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(img1, img2, data_range: float = 1.0, window: int = 11, sigma: float = 1.5, mask=None):
    """Mean SSIM per image (Wang et al. 2004): Gaussian 11x11 window, sigma 1.5, K1 = 0.01, K2 = 0.03.

    ``img1``, ``img2``: torch tensors [B, 3, H, W] in [0, data_range]. Channels are averaged.
    Returns a tensor [B]. Use only with a real paired reference image (e.g. cross-view validation pairs).
    With ``mask`` the SSIM map is averaged over masked pixels only (e.g. excluding plain white background,
    which otherwise inflates the score).
    """
    import torch
    import torch.nn.functional as F

    if img1.shape != img2.shape or img1.ndim != 4:
        raise ValueError(f"ssim needs equal [B,C,H,W] tensors, got {tuple(img1.shape)} / {tuple(img2.shape)}")
    c = img1.shape[1]
    w = _gaussian_window(window, sigma).to(img1.device).expand(c, 1, window, window)
    x, y = img1.float(), img2.float()
    pad = window // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    sxx = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    m = m.mean(1)[..., pad:-pad, pad:-pad]  # channel mean; drop borders affected by zero padding -> [B, H', W']
    if mask is None:
        return m.flatten(1).mean(1)
    # ``mask`` [B, H, W] bool (e.g. the person region): average the SSIM map over masked pixels only.
    mk = mask[..., pad:-pad, pad:-pad].to(m.dtype)
    return (m * mk).flatten(1).sum(1) / mk.flatten(1).sum(1).clamp(min=1)
