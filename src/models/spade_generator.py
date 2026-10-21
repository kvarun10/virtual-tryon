"""Stage C, part 2: SPADE translation network (paper Eq. 7, CoCosNet Sec. 3.2) and the full texture generator.

SPADE in plain words: a normal network normalises its activations and then rescales them with one
learned number per channel. SPADE instead predicts a **per-pixel** scale gamma(h, w) and shift beta(h, w)
from a guidance image, so the guidance decides *where* each colour or texture goes:

    out = norm(F)[c, h, w] * (1 + gamma[c, h, w]) + beta[c, h, w]                       (Eq. 7)

Guidance here = concat(warped exemplar I_S, target layout one-hot) [E]. Eq. 7 conditions on I_S only;
adding the layout gives sharp garment boundaries that the 32 x 32 I_S cannot provide.
Normalisation: positional normalisation (per position, over channels), as in CoCosNet App. E (Eq. 7 has
per-position mean/std). Instance norm is selectable.

Generator: learned constant [16nf, 4, 4] -> one SPADE ResBlock per resolution, doubling the size up to
H x W -> LeakyReLU -> 3x3 conv -> tanh (RGB in [-1, 1]).
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.models.alignment_network import AlignmentNetwork


class PositionalNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)


class SPADE(nn.Module):
    def __init__(self, channels: int, cond_channels: int, norm: str = "positional", hidden: int = 128):
        super().__init__()
        self.norm = PositionalNorm() if norm == "positional" else nn.InstanceNorm2d(channels, affine=False)
        self.shared = nn.Sequential(nn.Conv2d(cond_channels, hidden, 3, padding=1), nn.ReLU(inplace=True))
        self.gamma = nn.Conv2d(hidden, channels, 3, padding=1)
        self.beta = nn.Conv2d(hidden, channels, 3, padding=1)

    def forward(self, x, cond):
        cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
        h = self.shared(cond)
        return self.norm(x) * (1 + self.gamma(h)) + self.beta(h)


class SPADEResBlock(nn.Module):
    def __init__(self, fin: int, fout: int, cond_channels: int, norm: str, use_sn: bool):
        super().__init__()
        wrap = spectral_norm if use_sn else (lambda m: m)
        mid = min(fin, fout)
        self.learned_shortcut = fin != fout
        self.conv0 = wrap(nn.Conv2d(fin, mid, 3, padding=1))
        self.conv1 = wrap(nn.Conv2d(mid, fout, 3, padding=1))
        self.norm0 = SPADE(fin, cond_channels, norm)
        self.norm1 = SPADE(mid, cond_channels, norm)
        if self.learned_shortcut:
            self.conv_s = wrap(nn.Conv2d(fin, fout, 1, bias=False))
            self.norm_s = SPADE(fin, cond_channels, norm)

    def forward(self, x, cond):
        s = self.conv_s(self.norm_s(x, cond)) if self.learned_shortcut else x
        dx = self.conv0(F.leaky_relu(self.norm0(x, cond), 0.2))
        dx = self.conv1(F.leaky_relu(self.norm1(dx, cond), 0.2))
        return s + dx


class SPADEGenerator(nn.Module):
    def __init__(self, cond_channels: int, image_size, nf: int = 64, norm: str = "positional", use_sn: bool = True):
        super().__init__()
        h, w = int(image_size[0]), int(image_size[1])
        if h != w or h < 8 or h & (h - 1):
            raise ValueError(f"SPADE generator needs a square power-of-two size, got {h}x{w}")
        n_up = int(math.log2(h // 4))
        mult = [min(16, 2 ** (n_up - i)) for i in range(n_up + 1)]  # e.g. 128 px: 16,16,8,4,2,1
        self.const = nn.Parameter(torch.randn(1, mult[0] * nf, 4, 4))  # learned constant input (CoCosNet's constant code z)
        self.head = SPADEResBlock(mult[0] * nf, mult[0] * nf, cond_channels, norm, use_sn)
        self.ups = nn.ModuleList([SPADEResBlock(mult[i] * nf, mult[i + 1] * nf, cond_channels, norm, use_sn) for i in range(n_up)])
        self.out = nn.Conv2d(mult[-1] * nf, 3, 3, padding=1)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        x = self.head(self.const.expand(cond.shape[0], -1, -1, -1), cond)
        for block in self.ups:
            x = block(F.interpolate(x, scale_factor=2, mode="nearest"), cond)
        return torch.tanh(self.out(F.leaky_relu(x, 0.2)))


class TextureGenerator(nn.Module):
    """Alignment network + SPADE generator as one module (one optimiser, one checkpoint entry).

    ``forward(layout [B,Kg,H,W], exemplar [B,3,H,W], exemplar_parsing [B,Kg,H,W] or None) -> dict``
    with ``fake`` [B,3,H,W] plus the alignment outputs (``warped`` I_S, ``cycle``, features).
    """

    def __init__(self, layout_channels: int, image_size, corr_size: int = 32, feat_channels: int = 256,
                 alpha: float = 100.0, nf: int = 64, norm: str = "positional", use_sn: bool = True,
                 exemplar_parsing: bool = True):
        super().__init__()
        self.exemplar_parsing = exemplar_parsing
        ex_ch = 3 + (layout_channels if exemplar_parsing else 0)
        self.align = AlignmentNetwork(layout_channels, ex_ch, image_size, corr_size, feat_channels, alpha)
        self.spade = SPADEGenerator(3 + layout_channels, image_size, nf, norm, use_sn)
        self.cfg = dict(layout_channels=layout_channels, image_size=list(image_size), corr_size=corr_size,
                        feat_channels=feat_channels, alpha=alpha, nf=nf, norm=norm, use_sn=use_sn,
                        exemplar_parsing=exemplar_parsing)

    def exemplar_input(self, exemplar: torch.Tensor, exemplar_parsing) -> torch.Tensor:
        if not self.exemplar_parsing:
            return exemplar
        if exemplar_parsing is None:
            raise ValueError("model.texture_exemplar_parsing=true needs the exemplar's garment parsing")
        return torch.cat([exemplar, exemplar_parsing.to(exemplar.dtype)], 1)

    def forward(self, layout: torch.Tensor, exemplar: torch.Tensor, exemplar_parsing=None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        if layout.shape[-2:] != exemplar.shape[-2:]:
            raise ValueError(f"layout {tuple(layout.shape)} and exemplar {tuple(exemplar.shape)} differ in size")
        a = self.align(layout, self.exemplar_input(exemplar, exemplar_parsing), exemplar, return_attention)
        warped_full = F.interpolate(a["warped"].to(layout.dtype), size=layout.shape[-2:], mode="bilinear", align_corners=False)
        cond = torch.cat([warped_full, layout], 1)  # I_S (upsampled from the correspondence size) + layout
        a["fake"] = self.spade(cond)
        return a


def build_texture_generator(cfg, layout_channels: int) -> TextureGenerator:
    m = cfg.model
    return TextureGenerator(layout_channels, cfg.data.image_size, m.texture_corr_size, m.texture_feat_channels,
                            m.texture_alpha, m.generator_base_channels, m.texture_norm, m.spectral_norm,
                            m.texture_exemplar_parsing)
