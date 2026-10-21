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


# ======================================================================================
# Texture stage (Stage C)
# ======================================================================================

from pathlib import Path  # noqa: E402

import torch.nn as nn  # noqa: E402


class VGGWeightsUnavailableError(FileNotFoundError):
    pass


class VGGPerceptualLoss(nn.Module):
    """L1 between VGG-19 features of generated and real images (pix2pixHD / SPADE weighting).

    Layers relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 with weights 1/32, 1/16, 1/8, 1/4, 1. Inputs are
    RGB in [-1, 1] and are converted to ImageNet normalisation. Weights: torchvision VGG19_Weights.IMAGENET1K_V1
    (download.pytorch.org, ~548 MB). They are cached under ``cache_dir`` and downloaded only with ``allow_download``.
    ``init='random'`` uses an untrained VGG; this exists only for offline tests and gives meaningless perceptual values.
    """

    LAYER_ENDS = (2, 7, 12, 21, 30)  # indices in vgg19.features after relu1_1 ... relu5_1
    WEIGHTS = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0)

    def __init__(self, init: str = "imagenet", cache_dir=None, allow_download: bool = False):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        if init == "random":
            features = vgg19(weights=None).features
        else:
            url = VGG19_Weights.IMAGENET1K_V1.url
            cache = Path(cache_dir) if cache_dir else Path(torch.hub.get_dir()) / "checkpoints"
            file = cache / url.rsplit("/", 1)[-1]
            if not file.is_file():
                if not allow_download:
                    raise VGGWeightsUnavailableError(
                        f"VGG-19 ImageNet weights not found at {file} (~548 MB, BSD licence, {url}). Set "
                        "training.allow_vgg_download=true to download them once, or copy the file there."
                    )
                cache.mkdir(parents=True, exist_ok=True)
                torch.hub.download_url_to_file(url, str(file), progress=True)
            model = vgg19(weights=None)
            model.load_state_dict(torch.load(file, map_location="cpu", weights_only=True))
            features = model.features
        self.slices = nn.ModuleList()
        start = 0
        for end in self.LAYER_ENDS:
            self.slices.append(nn.Sequential(*[features[i] for i in range(start, end)]))
            start = end
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.init = init

    def train(self, mode: bool = True):  # always frozen / eval
        return super().train(False)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        x = ((fake.float() + 1) / 2 - self.mean) / self.std
        y = ((real.float() + 1) / 2 - self.mean) / self.std
        loss = fake.new_zeros((), dtype=torch.float32)
        for w, block in zip(self.WEIGHTS, self.slices):
            x, y = block(x), block(y)
            loss = loss + w * F.l1_loss(x, y.detach())
        return loss


def multiscale_hinge_d(real_outs, fake_outs) -> torch.Tensor:
    """Paper Eq. 8 summed over discriminator scales; ``*_outs`` = [(logits, features), ...]."""
    gan = GANLoss("hinge")
    return sum(gan.d_loss(r[0], f[0]) for r, f in zip(real_outs, fake_outs)) / len(real_outs)


def multiscale_hinge_g(fake_outs) -> torch.Tensor:
    """Paper Eq. 9 summed over scales."""
    gan = GANLoss("hinge")
    return sum(gan.g_loss(f[0]) for f in fake_outs) / len(fake_outs)


def feature_matching(real_outs, fake_outs) -> torch.Tensor:
    """L1 between D's intermediate features for real and generated images (real side detached)."""
    loss, n = 0.0, 0
    for (_, rf), (_, ff) in zip(real_outs, fake_outs):
        for r, f in zip(rf, ff):
            loss = loss + F.l1_loss(f.float(), r.detach().float())
            n += 1
    return loss / max(n, 1)
