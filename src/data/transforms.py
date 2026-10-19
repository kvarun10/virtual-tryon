"""Joint geometric augmentation for an image and its aligned label maps.

The same random transform must be applied to the RGB image (bilinear) and to every label map
(nearest), otherwise the supervision no longer lines up with the pixels.

The per-garment random affine augmentation of C_seg described in paper Sec. 5.1 (used to build
self-supervised warp triplets) belongs to Phase 4 and is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class AffineParams:
    angle: float = 0.0  # degrees, counter-clockwise
    scale: float = 1.0
    translate: Tuple[float, float] = (0.0, 0.0)  # fraction of (W, H)
    hflip: bool = False


def sample_affine(
    rng: np.random.Generator,
    max_angle: float = 10.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    max_translate: float = 0.05,
    hflip_prob: float = 0.0,
) -> AffineParams:
    return AffineParams(
        angle=float(rng.uniform(-max_angle, max_angle)),
        scale=float(rng.uniform(*scale_range)),
        translate=(float(rng.uniform(-max_translate, max_translate)), float(rng.uniform(-max_translate, max_translate))),
        hflip=bool(rng.random() < hflip_prob),
    )


def _apply(arr: np.ndarray, p: AffineParams, resample, fill) -> np.ndarray:
    img = Image.fromarray(arr)
    w, h = img.size
    if p.hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Inverse mapping matrix for PIL: output pixel -> input pixel, around the image centre.
    cx, cy = w / 2.0, h / 2.0
    tx, ty = p.translate[0] * w, p.translate[1] * h
    rad = np.deg2rad(p.angle)
    cos, sin = np.cos(rad) / p.scale, np.sin(rad) / p.scale
    a, b = cos, -sin
    d, e = sin, cos
    c = cx - a * (cx + tx) - b * (cy + ty)
    f = cy - d * (cx + tx) - e * (cy + ty)
    out = img.transform((w, h), Image.AFFINE, (a, b, c, d, e, f), resample=resample, fillcolor=fill)
    return np.asarray(out)


def apply_joint_affine(
    image: np.ndarray,
    labels: Dict[str, np.ndarray],
    params: AffineParams,
    image_fill=(255, 255, 255),
    label_fill: int = 0,
    swap_lr: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Apply one affine transform to an image and its label maps.

    ``swap_lr`` maps a label-map key to a LUT that exchanges left/right classes; it is applied
    after a horizontal flip (a flipped left arm becomes a right arm). Required for ``body15``.
    """
    out_img = _apply(image, params, Image.BILINEAR, tuple(image_fill))
    out_labels = {}
    for key, lab in labels.items():
        warped = _apply(lab, params, Image.NEAREST, label_fill)
        if params.hflip and swap_lr and key in swap_lr:
            warped = swap_lr[key][warped]
        out_labels[key] = warped
    return out_img, out_labels


def left_right_swap_lut(names) -> np.ndarray:
    """LUT that swaps classes named ``left_*`` <-> ``right_*`` (identity for the rest)."""
    names = list(names)
    lut = np.arange(len(names), dtype=np.uint8)
    for i, n in enumerate(names):
        for a, b in (("left_", "right_"), ("right_", "left_")):
            if n.startswith(a) and n.replace(a, b, 1) in names:
                lut[i] = names.index(n.replace(a, b, 1))
    return lut
