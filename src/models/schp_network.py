"""SCHP (Self-Correction Human Parsing) network, re-implemented without CUDA extensions.

Source: ``networks/AugmentCE2P.py`` of https://github.com/PeikeLi/Self-Correction-Human-Parsing
(MIT licence, Copyright (c) 2020 Peike Li). Module names and shapes are kept identical so the
official checkpoints load with ``strict=True``.

Why a re-implementation?
------------------------
The original uses ``InPlaceABNSync`` (Mapillary inplace_abn), a CUDA/C++ extension that is
JIT-compiled at import time and frequently fails on new Colab images. We replace it with
:class:`ABNCompat`, a pure-PyTorch layer with the same parameters **and the same maths**.

Checkpoint-compatibility caveat (verified in the SCHP sources, ``modules/src/inplace_abn*.c*``):
InPlaceABN does NOT scale by ``weight`` like ``nn.BatchNorm2d``; it scales by
``abs(weight) + eps`` (needed for its invertibility trick)::

    y = (x - mean) / sqrt(var + eps) * (|gamma| + eps) + beta,   then activation

Loading the checkpoint into ``nn.BatchNorm2d`` would succeed without any error but silently
produce wrong activations wherever a stored ``gamma`` is negative. :class:`ABNCompat` reproduces
the ``|gamma| + eps`` rule. The default activation is leaky ReLU with slope 0.01; ``BatchNorm2d``
in the backbone is ``InPlaceABNSync(activation='none')`` exactly as in the original.

What has / has not been verified is recorded in docs/implementation_status.md.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
from torch.nn import functional as F


class ABNCompat(nn.Module):
    """Pure-PyTorch equivalent of InPlaceABN(Sync) in inference *and* training mode.

    Parameters/buffers: ``weight``, ``bias``, ``running_mean``, ``running_var`` (no
    ``num_batches_tracked``, matching the checkpoint).
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True,
                 activation: str = "leaky_relu", slope: float = 0.01):
        super().__init__()
        if activation not in {"leaky_relu", "elu", "relu", "none"}:
            raise ValueError(f"Unsupported activation {activation}")
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.affine, self.activation, self.slope = affine, activation, slope
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.abs() + self.eps if self.affine else None  # InPlaceABN rule
        x = F.batch_norm(x, self.running_mean, self.running_var, weight, self.bias,
                         self.training, self.momentum, self.eps)
        if self.activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=self.slope)
        if self.activation == "elu":
            return F.elu(x)
        if self.activation == "relu":
            return F.relu(x)
        return x

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}, activation={self.activation}, gamma=|w|+eps"


InPlaceABNSync = ABNCompat
BatchNorm2d = functools.partial(ABNCompat, activation="none")


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu_inplace(out + residual)


class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_features),
        )

    @staticmethod
    def _make_stage(features, out_features, size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(features, out_features, kernel_size=1, bias=False),
            InPlaceABNSync(out_features),
        )

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(stage(feats), size=(h, w), mode="bilinear", align_corners=True) for stage in self.stages]
        return self.bottleneck(torch.cat(priors + [feats], 1))


class Edge_Module(nn.Module):  # noqa: N801 - keep original name for readability vs. source
    def __init__(self, in_fea=(256, 512, 1024), mid_fea=256, out_fea=2):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_fea[0], mid_fea, 1, bias=False), InPlaceABNSync(mid_fea))
        self.conv2 = nn.Sequential(nn.Conv2d(in_fea[1], mid_fea, 1, bias=False), InPlaceABNSync(mid_fea))
        self.conv3 = nn.Sequential(nn.Conv2d(in_fea[2], mid_fea, 1, bias=False), InPlaceABNSync(mid_fea))
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()
        e1f, e2f, e3f = self.conv1(x1), self.conv2(x2), self.conv3(x3)
        e1, e2, e3 = self.conv4(e1f), self.conv4(e2f), self.conv4(e3f)
        up = functools.partial(F.interpolate, size=(h, w), mode="bilinear", align_corners=True)
        e2f, e3f, e2, e3 = up(e2f), up(e3f), up(e2), up(e3)
        edge = self.conv5(torch.cat([e1, e2, e3], dim=1))
        return edge, torch.cat([e1f, e2f, e3f], dim=1)


class Decoder_Module(nn.Module):  # noqa: N801
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False), InPlaceABNSync(256))
        self.conv2 = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False), InPlaceABNSync(48))
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, 1, bias=False), InPlaceABNSync(256),
            nn.Conv2d(256, 256, 1, bias=False), InPlaceABNSync(256),
        )
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode="bilinear", align_corners=True)
        x = self.conv3(torch.cat([xt, self.conv2(xl)], dim=1))
        return self.conv4(x), x


class SCHPNet(nn.Module):
    """ResNet-101 + PSP context + edge branch + fusion head (CE2P with self-correction).

    Input: ``B x 3 x H x W`` **BGR** image normalised with mean (0.406, 0.456, 0.485) and std
    (0.225, 0.224, 0.229). Output (same nesting as the original):
    ``[[parsing_logits, fusion_logits], [edge_logits]]`` at stride 4. Inference uses
    ``fusion_logits`` (``output[0][-1]``), as SCHP's ``simple_extractor.py`` does.
    """

    def __init__(self, num_classes: int, layers=(3, 4, 23, 3)):
        super().__init__()
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))
        self.context_encoding = PSPModule(2048, 512)
        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)
        self.fushion = nn.Sequential(  # (sic) - original spelling, needed for checkpoint keys
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True),
        )

    def _make_layer(self, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * Bottleneck.expansion, affine=True),
            )

        def grid(i):
            return multi_grid[i % len(multi_grid)] if isinstance(multi_grid, tuple) else 1

        layers = [Bottleneck(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=grid(0))]
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation, multi_grid=grid(i)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.context_encoding(x5)
        parsing_result, parsing_fea = self.decoder(x, x2)
        edge_result, edge_fea = self.edge(x2, x3, x4)
        fusion_result = self.fushion(torch.cat([parsing_fea, edge_fea], dim=1))
        return [[parsing_result, fusion_result], [edge_result]]
