"""Segmentation adapters. No pretrained weights are downloaded: SCHP is tested with random
weights for shapes / checkpoint handling only; real-weight checks are in
scripts/verify_schp_checkpoint.py and must run on Colab."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.config import load_config
from src.data.labels import LabelError
from src.data.preprocessing import load_image
from src.models.schp_network import ABNCompat, SCHPNet
from src.models.segmentation import (
    AnnotationBackend,
    BackendError,
    CachedBackend,
    CheckpointMismatchError,
    SCHPBackend,
    SegmentationBackend,
    SegmentationResult,
    build_backend,
    load_state_dict_checked,
    to_project_labels,
)
from src.models.weights import WeightsUnavailableError


def test_abncompat_matches_inplace_abn_formula():
    torch.manual_seed(0)
    layer = ABNCompat(4, activation="leaky_relu").eval()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1.5, -2.0, 0.0, 0.7]))  # negative and zero gammas
        layer.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
        layer.running_mean.copy_(torch.tensor([0.5, -0.5, 0.0, 1.0]))
        layer.running_var.copy_(torch.tensor([2.0, 0.5, 1.0, 3.0]))
    x = torch.randn(2, 4, 3, 3)
    eps = layer.eps
    gamma = layer.weight.abs() + eps  # InPlaceABN kernel: abs(weight) + eps
    ref = (x - layer.running_mean.view(1, -1, 1, 1)) / torch.sqrt(layer.running_var.view(1, -1, 1, 1) + eps)
    ref = F.leaky_relu(ref * gamma.view(1, -1, 1, 1) + layer.bias.view(1, -1, 1, 1), 0.01)
    torch.testing.assert_close(layer(x), ref)
    # plain BatchNorm with the same tensors gives a different answer -> the substitution matters
    bn = torch.nn.BatchNorm2d(4).eval()
    bn.load_state_dict({**layer.state_dict(), "num_batches_tracked": torch.tensor(0)})
    assert not torch.allclose(F.leaky_relu(bn(x), 0.01), layer(x))
    assert set(layer.state_dict()) == {"weight", "bias", "running_mean", "running_var"}


@pytest.fixture(scope="module")
def tiny_schp_ckpt(tmp_path_factory):
    """Random-weight checkpoint in the official format ({'state_dict': {'module.' + key: tensor}})."""
    torch.manual_seed(0)
    net = SCHPNet(num_classes=20)
    path = tmp_path_factory.mktemp("ckpt") / "fake_lip.pth"
    torch.save({"state_dict": {"module." + k: v for k, v in net.state_dict().items()}, "epoch": 1}, path)
    return path


def test_schp_state_dict_layout():
    net = SCHPNet(num_classes=20)
    sd = net.state_dict()
    assert len(sd) == 603  # identical to the original AugmentCE2P (checked against the source)
    assert sd["fushion.3.weight"].shape == (20, 256, 1, 1)
    assert sd["decoder.conv4.weight"].shape == (20, 256, 1, 1)
    assert not any("num_batches_tracked" in k for k in sd)


def test_schp_backend_shapes_with_random_weights(tiny_schp_ckpt):
    backend = SCHPBackend("schp_lip", tiny_schp_ckpt, torch.device("cpu"), "clothing")
    backend.input_size = (65, 65)  # tiny input to keep the CPU test fast; real size is 473
    img = np.random.default_rng(0).integers(0, 255, (90, 61, 3)).astype(np.uint8)
    res = backend.predict(img)
    assert res.labels.shape == (90, 61) and res.labels.dtype == np.uint8 and res.labels.max() < 20
    assert backend.load_info["extra_keys"] == ["epoch"]


def test_checkpoint_mismatch_is_reported(tiny_schp_ckpt, tmp_path):
    with pytest.raises(CheckpointMismatchError, match="shape mismatches"):
        load_state_dict_checked(SCHPNet(num_classes=18), tiny_schp_ckpt)  # LIP weights, ATR head
    (tmp_path / "html.pth").write_text("<html>Google Drive quota exceeded</html>")
    with pytest.raises(CheckpointMismatchError, match="Could not read"):
        load_state_dict_checked(SCHPNet(num_classes=20), tmp_path / "html.pth")


def test_missing_weights_message(smoke_cfg):
    with pytest.raises(WeightsUnavailableError, match="allow_weight_download"):
        build_backend("schp_lip", smoke_cfg)  # downloads are off by default


def test_backend_role_and_name_checks(smoke_cfg):
    with pytest.raises(KeyError):
        build_backend("jppnet", smoke_cfg)
    with pytest.raises(BackendError, match="body"):
        build_backend("annotation_densepose", smoke_cfg, role="clothing")


def test_annotation_backends(fixture_root, smoke_cfg):
    cfg = load_config(overrides=[f"data.root={fixture_root.as_posix()}"])
    name = sorted(p.stem for p in (fixture_root / "images").glob("*.jpg"))[0]
    img = load_image(fixture_root / "images" / f"{name}.jpg")
    body = build_backend("annotation_densepose", cfg)
    res = body.predict(img, key=name)
    assert res.labels.shape == img.shape[:2] and res.label_set == "densepose"
    assert to_project_labels(res, "body15").max() < 15
    with pytest.raises(BackendError, match="needs the image name"):
        body.predict(img)
    with pytest.raises(LabelError, match="cannot produce body15|Cannot convert"):
        to_project_labels(SegmentationResult(np.zeros((2, 2), np.uint8), "pascal_person_part", "schp_pascal", "body"), "body15")


class _CountingBackend(SegmentationBackend):
    name, role, label_set = "counting", "clothing", "lip"

    def __init__(self, tag="v1"):
        self.calls, self.tag = 0, tag

    def predict(self, image, key=None):
        self.calls += 1
        return SegmentationResult(np.full(image.shape[:2], 5, np.uint8), "lip", self.name, self.role)

    def describe(self):
        return {**super().describe(), "tag": self.tag}


def test_cache_reuses_and_guards(tmp_path):
    inner = _CountingBackend()
    cached = CachedBackend(inner, tmp_path)
    img = np.zeros((6, 4, 3), np.uint8)
    a = cached.predict(img)
    b = cached.predict(img)
    c = cached.predict(img, key="named")
    assert inner.calls == 2 and cached.hits == 1
    np.testing.assert_array_equal(a.labels, b.labels)
    assert c.labels.shape == (6, 4)
    with pytest.raises(BackendError, match="different backend configuration"):
        CachedBackend(_CountingBackend(tag="v2"), tmp_path)
