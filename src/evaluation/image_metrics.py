"""LPIPS, FID and KID for the final evaluation (docs/final_evaluation.md).

* LPIPS (Zhang et al. 2018, ``lpips`` package, AlexNet v0.1): a learned perceptual *distance* between two images
  (0 = identical; lower is better). Optional mask: pixels outside the mask are set to white in BOTH images, so only
  the masked region (garment + skin) can contribute.
* FID (Heusel et al. 2017) with the pytorch-fid Inception network (2,048-d pool features); Fréchet distance between
  Gaussian fits of real and generated features. Biased upwards for small sample sizes.
* KID (Bińkowski et al. 2018): unbiased squared MMD with the cubic polynomial kernel, averaged over random subsets
  (mean ± std); preferred for ~1,000 images.

Backbone weights are downloaded only with ``allow_download=True`` into ``cache_dir/checkpoints`` (torch hub layout),
e.g. ``<drive>/pretrained/torch``, so Colab downloads them once.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch

ALEXNET_FILE = "alexnet-owt-7be5be79.pth"  # torchvision AlexNet (LPIPS backbone), ~233 MB
INCEPTION_FILE = "pt_inception-2015-12-05-6726825d.pth"  # pytorch-fid Inception, ~92 MB


class MetricWeightsUnavailableError(FileNotFoundError):
    pass


@contextlib.contextmanager
def _hub_dir(cache_dir: Optional[Path]):
    """Temporarily point torch.hub (and torchvision downloads) at ``cache_dir``."""
    if cache_dir is None:
        yield
        return
    old = torch.hub.get_dir()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_dir))
    try:
        yield
    finally:
        torch.hub.set_dir(old)


def _require(cache_dir: Optional[Path], filename: str, allow_download: bool, what: str, size: str, url: str) -> None:
    base = Path(cache_dir) if cache_dir else Path(torch.hub.get_dir())
    if not (base / "checkpoints" / filename).is_file() and not allow_download:
        raise MetricWeightsUnavailableError(
            f"{what} weights not found at {base / 'checkpoints' / filename} ({size}, {url}). Set "
            "evaluation.allow_metric_download=true (notebook: ALLOW_METRIC_DOWNLOAD=True) to download them once."
        )


class LPIPSMetric:
    def __init__(self, device: torch.device, cache_dir: Optional[Path] = None, allow_download: bool = False,
                 random_backbone: bool = False):
        import lpips

        if not random_backbone:
            _require(cache_dir, ALEXNET_FILE, allow_download, "LPIPS AlexNet", "~233 MB", "download.pytorch.org")
        with _hub_dir(cache_dir):
            # random_backbone only for offline tests: the LPIPS linear layers still come from the package
            self.model = lpips.LPIPS(net="alex", version="0.1", pnet_rand=random_backbone, verbose=False).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``img1``, ``img2``: [B, 3, H, W] in [0, 1]; ``mask``: [B, H, W] bool. Returns [B] distances."""
        if img1.shape != img2.shape or img1.ndim != 4:
            raise ValueError(f"LPIPS needs equal [B,3,H,W] tensors, got {tuple(img1.shape)} / {tuple(img2.shape)}")
        a, b = img1.float().to(self.device), img2.float().to(self.device)
        if mask is not None:
            m = mask.to(self.device).unsqueeze(1).float()
            a = a * m + (1 - m)
            b = b * m + (1 - m)
        return self.model(a * 2 - 1, b * 2 - 1).flatten()


class InceptionFeatures:
    """2,048-d Inception pool features for FID/KID (inputs [B, 3, H, W] in [0, 1]; resized to 299 internally)."""

    def __init__(self, device: torch.device, cache_dir: Optional[Path] = None, allow_download: bool = False,
                 extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.device = device
        if extractor is not None:  # tests inject a small network
            self.net = extractor
            return
        from pytorch_fid.inception import FID_WEIGHTS_URL, InceptionV3

        _require(cache_dir, INCEPTION_FILE, allow_download, "FID Inception", "~92 MB", FID_WEIGHTS_URL)
        with _hub_dir(cache_dir):
            net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=True)
        net = net.eval().to(device)
        self.net = lambda x: net(x)[0].flatten(1)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        return self.net(images.float().to(self.device)).float().cpu().numpy().astype(np.float64)


def frechet_distance(feats1: np.ndarray, feats2: np.ndarray) -> float:
    """FID between two feature sets (rows = images)."""
    from pytorch_fid.fid_score import calculate_frechet_distance

    if len(feats1) < 2 or len(feats2) < 2:
        raise ValueError("FID needs at least 2 images per set")
    mu1, mu2 = feats1.mean(0), feats2.mean(0)
    s1, s2 = np.cov(feats1, rowvar=False), np.cov(feats2, rowvar=False)
    return float(calculate_frechet_distance(mu1, s1, mu2, s2))


def kernel_inception_distance(feats1: np.ndarray, feats2: np.ndarray, n_subsets: int = 100, subset_size: int = 500,
                              seed: int = 0) -> Tuple[float, float]:
    """KID: unbiased MMD^2 with k(x, y) = (x.y / d + 1)^3, averaged over random subsets. Returns (mean, std)."""
    rng = np.random.default_rng(seed)
    n = min(subset_size, len(feats1), len(feats2))
    if n < 2:
        raise ValueError("KID needs at least 2 images per set")
    d = feats1.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = feats1[rng.choice(len(feats1), n, replace=False)]
        y = feats2[rng.choice(len(feats2), n, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
               - 2 * kxy.mean())
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))
