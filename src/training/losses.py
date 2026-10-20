"""Losses for the warp stage (paper Eqs. 2-4): L_warp = L_CE + lambda_adv * L_adv.

* :func:`segmentation_ce`: pixel-wise cross-entropy (Eq. 2) over K garment classes. It pushes every
  pixel towards the right class, which teaches the silhouette and the garment boundaries, but tends to
  leave uncertain boundary regions blurry (averaged probabilities).
* :class:`GANLoss`: adversarial term (Eq. 3). D sees (garment map, B_seg) and judges whether the shape is
  a plausible garment on that body; this sharpens ragged or fragmented shapes that CE accepts.
  Eq. 3 is written without logarithms and is ambiguous, so the formulation is configurable:
  ``lsgan`` (least squares), ``hinge`` or ``vanilla`` (BCE).

Numerics: CE is always computed in float32, even under autocast (fp16 log-softmax over many pixels loses
precision).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = 255


def segmentation_ce(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE,
                    weight: torch.Tensor = None) -> torch.Tensor:
    """Mean CE over supervised pixels. ``logits`` [B,K,H,W], ``target`` [B,H,W] int64."""
    if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError(f"logits {tuple(logits.shape)} vs target {tuple(target.shape)}")
    valid = target != ignore_index
    if not valid.any():
        return logits.float().sum() * 0.0  # keeps the graph; no supervised pixel in this batch
    if target[valid].max() >= logits.shape[1]:
        raise ValueError(f"target class {int(target[valid].max())} >= {logits.shape[1]} output channels")
    return F.cross_entropy(logits.float(), target, weight=weight, ignore_index=ignore_index)


def masked_probs(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE) -> torch.Tensor:
    """Softmax probabilities with unsupervised pixels zeroed (D must not judge them)."""
    mask = (target != ignore_index).unsqueeze(1).to(logits.dtype)
    return torch.softmax(logits.float(), dim=1).to(logits.dtype) * mask


def real_maps(target: torch.Tensor, num_classes: int, smoothing: float = 0.0, ignore_index: int = IGNORE) -> torch.Tensor:
    """One-hot real garment maps for D, optionally label-smoothed.

    Smoothing reduces the trivial cue "real maps are exactly one-hot, generated maps are soft".
    """
    mask = (target != ignore_index)
    oh = F.one_hot(target.clamp(max=num_classes - 1) * mask, num_classes).permute(0, 3, 1, 2).float()
    if smoothing > 0:
        oh = oh * (1 - smoothing) + smoothing / num_classes
    return oh * mask.unsqueeze(1)


class GANLoss:
    def __init__(self, mode: str = "lsgan"):
        if mode not in {"lsgan", "hinge", "vanilla"}:
            raise ValueError(mode)
        self.mode = mode

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        r, f = real_logits.float(), fake_logits.float()
        if self.mode == "lsgan":
            return 0.5 * (((r - 1) ** 2).mean() + (f ** 2).mean())
        if self.mode == "hinge":
            return F.relu(1 - r).mean() + F.relu(1 + f).mean()
        return F.binary_cross_entropy_with_logits(r, torch.ones_like(r)) + F.binary_cross_entropy_with_logits(f, torch.zeros_like(f))

    def g_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        f = fake_logits.float()
        if self.mode == "lsgan":
            return 0.5 * ((f - 1) ** 2).mean()
        if self.mode == "hinge":
            return -f.mean()
        return F.binary_cross_entropy_with_logits(f, torch.ones_like(f))
