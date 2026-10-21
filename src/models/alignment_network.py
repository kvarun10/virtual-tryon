"""Stage C, part 1: alignment (correspondence) network, paper Eqs. 5-6 / CoCosNet Sec. 3.1.

Two domain adaptors map the target layout (F_x) and the exemplar (F_y) into a shared feature space at
the correspondence resolution. Then (in plain words):

1. every target position u and every exemplar position v get a feature vector T_B(u), T_C(v);
2. similarity M(u, v) = cosine of the channel-centred vectors (Eq. 5);
3. for each u, softmax over all v of alpha * M(u, v) gives weights that sum to 1 (Eq. 6);
4. the warped exemplar I_S(u) is the weighted average of the exemplar colours at those v.

Large alpha (100) makes the softmax almost an argmax: each target pixel copies its best match.

Numerics: steps 2-4 always run in FP32, even under autocast. alpha * M reaches 100, and fp16 softmax
over 1,024-4,096 entries loses precision. With M in [-1, 1] and alpha = 100, exp(100) overflows fp16
unless max-subtraction is perfect.

Memory: M is [B, HW, HW]. At 32 x 32 that is 1,048,576 floats (4 MiB) per sample; at 64 x 64, 16.8 M
(64 MiB). :func:`correlation_bytes` estimates this.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def correlation_bytes(corr_size: int, batch: int, factor: float = 4.0) -> int:
    """Rough FP32 memory for M, softmax and their gradients (``factor`` copies)."""
    hw = corr_size * corr_size
    return int(batch * hw * hw * 4 * factor)


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.InstanceNorm2d(ch, affine=True), nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(ch, ch, 3, padding=1), nn.InstanceNorm2d(ch, affine=True))

    def forward(self, x):
        return x + self.body(x)


class DomainAdaptor(nn.Module):
    """Conv-InstanceNorm-LeakyReLU blocks down to the correspondence resolution (CoCosNet App. E)."""

    def __init__(self, in_ch: int, n_down: int, feat_ch: int = 256, base: int = 64):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base, 3, padding=1), nn.InstanceNorm2d(base, affine=True), nn.LeakyReLU(0.2, True)]
        ch = base
        for i in range(n_down):
            nxt = min(base * 2 ** (i + 1), feat_ch)
            layers += [nn.Conv2d(ch, nxt, 4, stride=2, padding=1), nn.InstanceNorm2d(nxt, affine=True), nn.LeakyReLU(0.2, True)]
            ch = nxt
        layers += [nn.Conv2d(ch, feat_ch, 3, padding=1), nn.InstanceNorm2d(feat_ch, affine=True), nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AlignmentNetwork(nn.Module):
    def __init__(self, layout_channels: int, exemplar_channels: int, image_size, corr_size: int = 32,
                 feat_channels: int = 256, alpha: float = 100.0):
        super().__init__()
        h, w = int(image_size[0]), int(image_size[1])
        if h != w or h % corr_size or (h // corr_size) & (h // corr_size - 1):
            raise ValueError(f"image {h}x{w} must be square and a power-of-two multiple of corr_size={corr_size}")
        n_down = int(math.log2(h // corr_size))
        self.corr_size, self.alpha, self.image_size = corr_size, float(alpha), (h, w)
        self.fx = DomainAdaptor(layout_channels, n_down, feat_channels)
        self.fy = DomainAdaptor(exemplar_channels, n_down, feat_channels)
        # shared block that further adapts both branches (CoCosNet "shared adaptive feature block"),
        # ending with a parameter-free channel normalisation so the domain loss cannot shrink features to 0
        self.shared = nn.Sequential(_ResBlock(feat_channels), _ResBlock(feat_channels),
                                    nn.Conv2d(feat_channels, feat_channels, 1), nn.InstanceNorm2d(feat_channels, affine=False))
        self.cfg = dict(layout_channels=layout_channels, exemplar_channels=exemplar_channels, image_size=[h, w],
                        corr_size=corr_size, feat_channels=feat_channels, alpha=alpha)

    def encode_x(self, layout: torch.Tensor) -> torch.Tensor:
        return self.shared(self.fx(layout))

    def encode_y(self, exemplar_input: torch.Tensor) -> torch.Tensor:
        return self.shared(self.fy(exemplar_input))

    def forward(self, layout: torch.Tensor, exemplar_input: torch.Tensor, exemplar_rgb: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """``exemplar_input`` = exemplar RGB (+ parsing); ``exemplar_rgb`` [B,3,H,W] is what gets warped."""
        tx, ty = self.encode_x(layout), self.encode_y(exemplar_input)
        b, c, h, w = tx.shape
        with torch.autocast(device_type=tx.device.type, enabled=False):
            fx = tx.float().flatten(2)  # [B, C, HW]
            fy = ty.float().flatten(2)
            fx = F.normalize(fx - fx.mean(dim=1, keepdim=True), dim=1)  # channel-centred, unit length (Eq. 5)
            fy = F.normalize(fy - fy.mean(dim=1, keepdim=True), dim=1)
            m = torch.bmm(fx.transpose(1, 2), fy)  # [B, HW_u, HW_v], cosine in [-1, 1]
            attn = torch.softmax(self.alpha * m, dim=-1)  # over v (Eq. 6)
            y_small = F.adaptive_avg_pool2d(exemplar_rgb.float(), (h, w))  # exemplar at correspondence resolution
            y_flat = y_small.flatten(2).transpose(1, 2)  # [B, HW_v, 3]
            warped = torch.bmm(attn, y_flat).transpose(1, 2).reshape(b, 3, h, w)  # I_S
            # cycle (CoCosNet L_reg): warp back with softmax over u and compare with the exemplar
            attn_back = torch.softmax(self.alpha * m, dim=1)  # [B, HW_u, HW_v], columns sum to 1
            back = torch.bmm(attn_back.transpose(1, 2), warped.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, 3, h, w)
        out = {"warped": warped, "cycle": back, "exemplar_small": y_small, "feat_x": tx, "feat_y": ty}
        if return_attention:
            out["attention"] = attn
        return out
