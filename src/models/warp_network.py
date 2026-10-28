"""Stage B: dual-path U-Net warp generator, C'_seg = G_warp(B_seg, C_seg)  (paper Eq. 1, Fig. 8).

What the paper fixes [P] and what we chose [E]
----------------------------------------------
[P] two encoders (body / clothes) and one decoder.
[P] body encoder output 512 x 16 x 16 at 256 px input  -> 4 stride-2 levels, channels 64-128-256-512.
[P] clothes encoder squeezed to a narrow 2 x 2 x 1024 code ("weakly conditioned"), then up-sampled
    to 512 x 16 x 16.
[P] concatenation followed by 4 residual blocks; decoder up-samples to the output segmentation.
[E] 1x1 fusion conv 1024 -> 512 (Fig. 8 draws the residual blocks at 512 channels).
[E] U-Net skip connections from the body encoder only (``skip='body'``); ``'dual'`` adds clothes-encoder
    skips as in the community SwapNet code; ``'none'`` disables them. The paper calls the network a
    U-Net but does not show skips.
[E] conv 4x4 stride 2 + norm + LeakyReLU(0.2) downsampling (pix2pix/SwapNet convention); the deepest
    clothes layer has no normalisation (a 2 x 2 map has too few pixels for instance statistics).
[E] decoder: nearest upsample + 3x3 conv (avoids checkerboard artefacts of transposed convs).
[E] output: K-class logits (softmax/CE), not SwapNet's tanh, because the loss is cross-entropy (Eq. 2).

Tensor shapes for the 128 px baseline (B = batch):
    cloth  [B, 12, 128, 128] -> 6 levels -> [B, 1024, 2, 2] -> up x2 -> [B, 512, 8, 8]
    body   [B,  7, 128, 128] -> 4 levels -> [B, 512, 8, 8]
    fuse   cat -> [B, 1024, 8, 8] -> 1x1 -> [B, 512, 8, 8] -> 4 res blocks
    decode 8 -> 16 -> 32 -> 64 (+ body skips) -> 128 -> logits [B, 12, 128, 128]
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(kind: str, ch: int) -> nn.Module:
    if kind == "instance":
        return nn.InstanceNorm2d(ch, affine=True)
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm '{kind}'")


class Down(nn.Module):
    def __init__(self, cin: int, cout: int, norm: str, use_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 4, stride=2, padding=1, bias=not use_norm or norm == "none"),
            make_norm(norm, cout) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.body(x)


class ConvBlock(nn.Module):
    """3x3 conv + norm + ReLU (used after up-sampling)."""

    def __init__(self, cin: int, cout: int, norm: str, dropout: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=norm == "none"),
            make_norm(norm, cout),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.body(x)


class ResBlock(nn.Module):
    def __init__(self, ch: int, norm: str, dropout: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=norm == "none"), make_norm(norm, ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=norm == "none"), make_norm(norm, ch),
        )

    def forward(self, x):
        return x + self.body(x)


class DualPathUNet(nn.Module):
    """See module docstring. ``forward(cloth, body) -> logits [B, out_channels, H, W]``."""

    BODY_LEVELS = 4  # [P] 512 x 16 x 16 at 256 px

    def __init__(self, cloth_channels: int, body_channels: int, out_channels: int, image_size: Sequence[int],
                 base: int = 64, max_channels: int = 1024, cloth_bottleneck: int = 2, res_blocks: int = 4,
                 skip: str = "body", norm: str = "instance", dropout: float = 0.0):
        super().__init__()
        h, w = int(image_size[0]), int(image_size[1])
        side = min(h, w)
        if cloth_bottleneck < 1 or side % cloth_bottleneck or (side // cloth_bottleneck) & (side // cloth_bottleneck - 1):
            raise ValueError(f"min(H, W)={side} / cloth_bottleneck={cloth_bottleneck} must be a power of two")
        self.cloth_levels = int(math.log2(side // cloth_bottleneck))
        if self.cloth_levels < self.BODY_LEVELS:
            raise ValueError(f"Image {h}x{w} too small for a {cloth_bottleneck}x{cloth_bottleneck} bottleneck "
                             f"below {self.BODY_LEVELS} body levels")
        div = 2 ** self.cloth_levels
        if h % div or w % div:
            raise ValueError(f"H and W ({h}x{w}) must be divisible by {div} for {self.cloth_levels} downsamplings")
        if skip not in {"body", "dual", "none"}:
            raise ValueError("skip must be body|dual|none")
        self.cfg = dict(cloth_channels=cloth_channels, body_channels=body_channels, out_channels=out_channels,
                        image_size=[h, w], base=base, max_channels=max_channels, cloth_bottleneck=cloth_bottleneck,
                        res_blocks=res_blocks, skip=skip, norm=norm, dropout=dropout)
        self.skip = skip
        ch = lambda lvl: min(base * 2 ** (lvl - 1), max_channels)  # channels at resolution H / 2**lvl
        nb, nc = self.BODY_LEVELS, self.cloth_levels

        self.body_enc = nn.ModuleList(
            [Down(body_channels if l == 1 else ch(l - 1), ch(l), norm, use_norm=l != 1) for l in range(1, nb + 1)])
        self.cloth_enc = nn.ModuleList(
            [Down(cloth_channels if l == 1 else ch(l - 1), ch(l), norm, use_norm=l not in (1, nc),
                  dropout=dropout if l >= nc - 1 else 0.0) for l in range(1, nc + 1)])
        # up-sample the narrow clothes code back to the body resolution, without skips [P]
        self.cloth_up = nn.ModuleList([ConvBlock(ch(l + 1), ch(l), norm) for l in range(nc - 1, nb - 1, -1)])
        self.fuse = nn.Sequential(nn.Conv2d(2 * ch(nb), ch(nb), 1, bias=norm == "none"), make_norm(norm, ch(nb)),
                                  nn.ReLU(inplace=True))
        self.res = nn.Sequential(*[ResBlock(ch(nb), norm, dropout) for _ in range(res_blocks)])
        dec = []
        for l in range(nb - 1, 0, -1):
            skip_ch = {"body": ch(l), "dual": 2 * ch(l), "none": 0}[skip]
            dec.append(ConvBlock(ch(l + 1) + skip_ch, ch(l), norm, dropout if l >= nb - 2 else 0.0))
        self.dec = nn.ModuleList(dec)
        self.head = nn.Sequential(ConvBlock(ch(1), base, norm), nn.Conv2d(base, out_channels, 1))

    def encode(self, cloth: torch.Tensor, body: torch.Tensor) -> Dict[str, object]:
        b_feats: List[torch.Tensor] = []
        x = body
        for layer in self.body_enc:
            x = layer(x)
            b_feats.append(x)
        c_feats: List[torch.Tensor] = []
        y = cloth
        for layer in self.cloth_enc:
            y = layer(y)
            c_feats.append(y)
        code = y  # [B, 1024, bottleneck, bottleneck]
        for layer in self.cloth_up:
            y = layer(F.interpolate(y, scale_factor=2, mode="nearest"))
        return {"body": b_feats, "cloth": c_feats, "cloth_code": code, "cloth_up": y}

    def forward(self, cloth: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        if cloth.shape[-2:] != body.shape[-2:]:
            raise ValueError(f"cloth {tuple(cloth.shape)} and body {tuple(body.shape)} differ in spatial size")
        if cloth.shape[1] != self.cfg["cloth_channels"] or body.shape[1] != self.cfg["body_channels"]:
            raise ValueError(f"expected {self.cfg['cloth_channels']} cloth / {self.cfg['body_channels']} body channels, "
                             f"got {cloth.shape[1]} / {body.shape[1]}. Check labels.body_label_set vs the checkpoint.")
        if list(cloth.shape[-2:]) != self.cfg["image_size"]:
            raise ValueError(f"input size {tuple(cloth.shape[-2:])} != model image_size {self.cfg['image_size']}")
        f = self.encode(cloth, body)
        x = self.res(self.fuse(torch.cat([f["body"][-1], f["cloth_up"]], dim=1)))
        nb = self.BODY_LEVELS
        for i, layer in enumerate(self.dec):
            l = nb - 1 - i  # target level (resolution H / 2**l)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if self.skip in ("body", "dual"):
                parts = [x, f["body"][l - 1]] + ([f["cloth"][l - 1]] if self.skip == "dual" else [])
                x = torch.cat(parts, dim=1)
            x = layer(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.head(x)

    def feature_shapes(self, batch: int = 1) -> Dict[str, Tuple[int, ...]]:
        h, w = self.cfg["image_size"]
        dev = next(self.parameters()).device
        with torch.no_grad():
            f = self.encode(torch.zeros(batch, self.cfg["cloth_channels"], h, w, device=dev),
                            torch.zeros(batch, self.cfg["body_channels"], h, w, device=dev))
        return {"body_top": tuple(f["body"][-1].shape), "cloth_code": tuple(f["cloth_code"].shape),
                "cloth_up": tuple(f["cloth_up"].shape)}


EXTRA_BODY_SETS = {"none": None, "densepose": "body15"}


def warp_channels(cfg):
    """(clothes-encoder channels, body-encoder channels, garment classes) for ``cfg`` (see model.warp_extra_body /
    model.warp_ref_body)."""
    from src.data.labels import get_label_set

    kg = get_label_set(cfg.labels.garment_label_set).num_classes
    kb = get_label_set(cfg.labels.body_label_set).num_classes
    extra = EXTRA_BODY_SETS[cfg.model.warp_extra_body]
    if extra:
        kb += get_label_set(extra).num_classes
    return kg + (kb if cfg.model.warp_ref_body else 0), kb, kg


def build_warp_generator(cfg, cloth_channels: int, body_channels: int, out_channels: int) -> DualPathUNet:
    m = cfg.model
    return DualPathUNet(cloth_channels, body_channels, out_channels, cfg.data.image_size, base=m.warp_base_channels,
                        max_channels=m.warp_max_channels, cloth_bottleneck=m.warp_cloth_bottleneck,
                        res_blocks=m.warp_res_blocks, skip=m.warp_skip, norm=m.warp_norm, dropout=m.warp_dropout)
