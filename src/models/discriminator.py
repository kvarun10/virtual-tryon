"""Conditional PatchGAN discriminator for the warp stage (paper Eq. 3 uses a discriminator D;
its architecture is not given [E]).

Input: concat(garment map probabilities [B, Kg, H, W], B_seg one-hot [B, Kb, H, W]). Output: a grid of
real/fake logits (one per overlapping image patch). Conditioning on B_seg lets D judge whether the
garment *fits the body*, not only whether it looks like a garment. Spectral normalisation is used
instead of batch statistics (paper Sec. 5.1 applies it to the texture-stage networks).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int, base: int = 64, n_layers: int = 3, use_spectral_norm: bool = True):
        super().__init__()
        wrap = sn if use_spectral_norm else (lambda m: m)
        layers = [wrap(nn.Conv2d(in_channels, base, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
        ch = base
        for i in range(1, n_layers):
            nxt = min(base * 2 ** i, 512)
            layers += [wrap(nn.Conv2d(ch, nxt, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
            ch = nxt
        nxt = min(ch * 2, 512)
        layers += [wrap(nn.Conv2d(ch, nxt, 4, 1, 1)), nn.LeakyReLU(0.2, inplace=True), wrap(nn.Conv2d(nxt, 1, 4, 1, 1))]
        self.net = nn.Sequential(*layers)
        self.cfg = dict(in_channels=in_channels, base=base, n_layers=n_layers, use_spectral_norm=use_spectral_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
