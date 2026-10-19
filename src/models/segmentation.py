"""Stage A (pre-treatment) segmentation backends behind one interface.

Paper: Dress-up uses LIP-JppNet for clothing segmentation and the 2-D part segmentation of
Neural Body Fitting (NBF) for the body (Sec. 4.3.1). Neither runs on current Colab (TensorFlow
1.x); see docs/paper_analysis.md Sec. 4. Substitutes provided here:

=====================  ========  ======================  ======================================
backend name           role      native label set        status
=====================  ========  ======================  ======================================
schp_lip               clothing  lip (20)                pretrained external (LIP-JppNet substitute)
schp_atr               clothing  atr (18)                pretrained external (no coat class)
segformer_clothes      clothing  atr (18)                pretrained external, optional, non-commercial
schp_pascal            body      pascal_person_part (7)  pretrained external (coarse NBF substitute)
annotation_parsing     clothing  dfmm (24)               dataset ground truth (training / evaluation)
annotation_densepose   body      densepose (25)          dataset DensePose (NBF substitute, 14 parts)
=====================  ========  ======================  ======================================

All backends take an ``H x W x 3`` uint8 RGB image and return a :class:`SegmentationResult`
whose ``labels`` has the *same height and width as the input image*. Conversion to the project
label sets (garment12 / body15) is done by :func:`to_project_labels`, never inside a backend, so
backends stay swappable.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ProjectConfig
from src.data.labels import LabelError, check_label_map, convert_labels, get_label_set, lut_for
from src.data.preprocessing import load_image, load_label_map
from src.models.weights import WEIGHTS, ensure_weights
from src.utils.device import select_device


class BackendError(RuntimeError):
    """Base class for segmentation backend failures."""


class CheckpointMismatchError(BackendError):
    """The checkpoint does not match the architecture (wrong file / label set / corruption)."""


@dataclass
class SegmentationResult:
    labels: np.ndarray  # H x W uint8, same size as the input image
    label_set: str
    backend: str
    role: str
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        check_label_map(self.labels, get_label_set(self.label_set), f"output of backend '{self.backend}'")


class SegmentationBackend(ABC):
    name: str = "abstract"
    role: str = "clothing"  # clothing | body
    label_set: str = ""

    @abstractmethod
    def predict(self, image: np.ndarray, key: Optional[str] = None) -> SegmentationResult:
        """Segment an RGB uint8 image. ``key`` identifies dataset images (needed by annotation backends)."""

    def describe(self) -> Dict:
        return {"name": self.name, "role": self.role, "label_set": self.label_set}


# --------------------------------------------------------------------------------------
# SCHP
# --------------------------------------------------------------------------------------

SCHP_INPUT_SIZE = {"lip": (473, 473), "atr": (512, 512), "pascal_person_part": (512, 512)}  # (H, W), simple_extractor.py
SCHP_MEAN_BGR = (0.406, 0.456, 0.485)
SCHP_STD_BGR = (0.225, 0.224, 0.229)


def _schp_affine(center: np.ndarray, scale: np.ndarray, output_size: Tuple[int, int], inv: bool = False) -> np.ndarray:
    """``get_affine_transform`` from SCHP utils/transforms.py (rotation 0, shift 0).

    Original: Copyright (c) Microsoft, MIT licence (Bin Xiao), as vendored in SCHP.
    """
    import cv2

    src_w = scale[0]
    dst_h, dst_w = output_size
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src_dir = np.array([0, src_w * -0.5], np.float32)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = dst[0, :] + dst_dir

    def third(a, b):
        d = a - b
        return b + np.array([-d[1], d[0]], dtype=np.float32)

    src[2, :] = third(src[0, :], src[1, :])
    dst[2, :] = third(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(dst, src) if inv else cv2.getAffineTransform(src, dst)


def _schp_center_scale(w: int, h: int, aspect_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """``_box2cs([0, 0, w - 1, h - 1])`` from SCHP datasets/simple_extractor_dataset.py."""
    bw, bh = w - 1, h - 1
    center = np.array([bw * 0.5, bh * 0.5], dtype=np.float32)
    if bw > aspect_ratio * bh:
        bh = bw / aspect_ratio
    elif bw < aspect_ratio * bh:
        bw = bh * aspect_ratio
    return center, np.array([bw, bh], dtype=np.float32)


def load_state_dict_checked(model: torch.nn.Module, checkpoint_path: Path) -> Dict:
    """Load a checkpoint with ``strict`` semantics and a readable error report."""
    try:
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointMismatchError(
            f"Could not read checkpoint {checkpoint_path}: {type(exc).__name__}: {exc}. The file may be "
            "truncated (interrupted download) or not a PyTorch checkpoint (e.g. a saved HTML page)."
        ) from exc
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else None
    if not isinstance(state, dict) or not state:
        raise CheckpointMismatchError(f"{checkpoint_path} does not contain a state_dict")
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}  # strip DataParallel prefix
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = sorted(
        f"{k}: checkpoint {tuple(state[k].shape)} vs model {tuple(expected[k].shape)}"
        for k in set(state) & set(expected)
        if tuple(state[k].shape) != tuple(expected[k].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise CheckpointMismatchError(
            f"Checkpoint {checkpoint_path.name} does not match {type(model).__name__}:\n"
            f"  missing ({len(missing)}): {missing[:5]}\n  unexpected ({len(unexpected)}): {unexpected[:5]}\n"
            f"  shape mismatches ({len(shape_mismatch)}): {shape_mismatch[:3]}\n"
            "A classifier shape mismatch usually means the wrong label set (e.g. LIP weights with ATR config)."
        )
    model.load_state_dict(state, strict=True)
    return {"n_tensors": len(state), "extra_keys": [k for k in obj if k != "state_dict"] if isinstance(obj, dict) else []}


class SCHPBackend(SegmentationBackend):
    """Faithful port of SCHP ``simple_extractor.py`` inference (no flip test-time augmentation)."""

    def __init__(self, weights_key: str, checkpoint_path: Path, device: torch.device, role: str):
        from src.models.schp_network import SCHPNet

        spec = WEIGHTS[weights_key]
        self.name = weights_key
        self.role = role
        self.label_set = spec.label_set
        self.input_size = SCHP_INPUT_SIZE[spec.label_set]
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.model = SCHPNet(num_classes=get_label_set(spec.label_set).num_classes)
        self.load_info = load_state_dict_checked(self.model, self.checkpoint_path)
        self.model.eval().to(device)
        self._mean = torch.tensor(SCHP_MEAN_BGR, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(SCHP_STD_BGR, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def predict(self, image: np.ndarray, key: Optional[str] = None) -> SegmentationResult:
        import cv2

        image = load_image(image)
        h, w = image.shape[:2]
        in_h, in_w = self.input_size
        center, scale = _schp_center_scale(w, h, in_w / in_h)
        trans = _schp_affine(center, scale, self.input_size)
        bgr = np.ascontiguousarray(image[:, :, ::-1])  # SCHP reads images with cv2 (BGR)
        warped = cv2.warpAffine(bgr, trans, (in_w, in_h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        x = torch.from_numpy(warped).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        x = (x - self._mean) / self._std
        out = self.model(x)
        logits = out[0][-1]  # fusion branch, as simple_extractor.py
        logits = F.interpolate(logits, size=(in_h, in_w), mode="bilinear", align_corners=True)
        logits = logits[0].permute(1, 2, 0).float().cpu().numpy()  # H x W x C
        inv = _schp_affine(center, scale, self.input_size, inv=True)
        back = np.stack(
            [cv2.warpAffine(logits[:, :, c], inv, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in range(logits.shape[2])],
            axis=2,
        )
        labels = back.argmax(axis=2).astype(np.uint8)
        return SegmentationResult(labels, self.label_set, self.name, self.role, {"input_size": self.input_size})

    def describe(self) -> Dict:
        return {**super().describe(), "checkpoint": str(self.checkpoint_path), "input_size": self.input_size,
                "architecture": "SCHP ResNet-101 (CE2P), pure-PyTorch ABN"}


# --------------------------------------------------------------------------------------
# SegFormer (optional)
# --------------------------------------------------------------------------------------


class SegFormerClothesBackend(SegmentationBackend):
    """``mattmdjaga/segformer_b2_clothes`` via Hugging Face transformers (18 ATR labels).

    Optional fallback; its weights inherit the NVIDIA SegFormer licence (non-commercial). Not
    exercised by the test-suite (needs a download). Label order was checked against the model
    card; it is re-checked at load time against the model's ``id2label``.
    """

    MODEL_ID = "mattmdjaga/segformer_b2_clothes"
    name = "segformer_clothes"
    role = "clothing"
    label_set = "atr"

    def __init__(self, device: torch.device, allow_download: bool):
        try:
            from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
        except ImportError as exc:
            raise BackendError("segformer_clothes needs `pip install transformers`") from exc
        try:
            self.processor = SegformerImageProcessor.from_pretrained(self.MODEL_ID, local_files_only=not allow_download)
            self.model = AutoModelForSemanticSegmentation.from_pretrained(self.MODEL_ID, local_files_only=not allow_download)
        except OSError as exc:
            raise BackendError(
                f"{self.MODEL_ID} is not in the local Hugging Face cache. Set "
                "segmentation.allow_weight_download=true to download it."
            ) from exc
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        expected = get_label_set("atr").names
        got = tuple(id2label[i].lower().replace("-", "_") for i in range(len(id2label)))
        if got != expected:
            raise CheckpointMismatchError(f"Unexpected SegFormer label order {got}; expected {expected}")
        self.device = device
        self.model.eval().to(device)

    @torch.no_grad()
    def predict(self, image: np.ndarray, key: Optional[str] = None) -> SegmentationResult:
        image = load_image(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        logits = F.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
        labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return SegmentationResult(labels, self.label_set, self.name, self.role)


# --------------------------------------------------------------------------------------
# Dataset annotations (ground truth / precomputed)
# --------------------------------------------------------------------------------------


class AnnotationBackend(SegmentationBackend):
    """Reads DFMM-layout annotation files instead of running a network.

    Used (1) to train later stages on manual labels and (2) as the reference when evaluating the
    pretrained parsers. Only works for images that belong to the dataset (``key`` = image stem).
    """

    def __init__(self, kind: str, data_root: Path, layout):
        if kind not in {"parsing", "densepose"}:
            raise ValueError(kind)
        self.kind = kind
        self.root = Path(data_root)
        self.layout = layout
        self.name = f"annotation_{kind}"
        self.role = "clothing" if kind == "parsing" else "body"
        self.label_set = "dfmm" if kind == "parsing" else "densepose"

    def predict(self, image: np.ndarray, key: Optional[str] = None) -> SegmentationResult:
        if not key:
            raise BackendError(
                f"Backend '{self.name}' reads dataset annotation files and needs the image name (key). "
                "For arbitrary photos choose a network backend such as schp_lip / schp_pascal."
            )
        if self.kind == "parsing":
            path = self.root / self.layout.parsing_dir / f"{key}{self.layout.parsing_suffix}"
            labels = load_label_map(path)
        else:
            path = self.root / self.layout.densepose_dir / f"{key}{self.layout.densepose_suffix}"
            labels = load_label_map(path, channel=self.layout.densepose_part_channel)
        if image is not None and labels.shape != np.asarray(image).shape[:2]:
            raise LabelError(f"{path.name}: annotation size {labels.shape} != image size {np.asarray(image).shape[:2]}")
        return SegmentationResult(labels, self.label_set, self.name, self.role, {"path": str(path)})


# --------------------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------------------


class CachedBackend(SegmentationBackend):
    """Stores each prediction as a PNG under ``cache_dir/<backend>/``; re-uses it on later calls.

    Cache key: the dataset image name if given, else the SHA-1 of the image pixels. A
    ``backend.json`` file records the wrapped backend's description; if it differs from the
    current one (e.g. a different checkpoint) the cache is refused instead of silently mixing.
    """

    def __init__(self, backend: SegmentationBackend, cache_dir: Path):
        self.inner = backend
        self.name, self.role, self.label_set = backend.name, backend.role, backend.label_set
        self.dir = Path(cache_dir) / backend.name
        self.dir.mkdir(parents=True, exist_ok=True)
        desc_path = self.dir / "backend.json"
        desc = json.dumps(backend.describe(), sort_keys=True)
        if desc_path.is_file():
            if desc_path.read_text(encoding="utf-8") != desc:
                raise BackendError(
                    f"Cache {self.dir} was produced by a different backend configuration:\n"
                    f"  cached : {desc_path.read_text(encoding='utf-8')}\n  current: {desc}\n"
                    "Delete that folder to rebuild the cache."
                )
        else:
            desc_path.write_text(desc, encoding="utf-8")
        self.hits = self.misses = 0

    def _key(self, image: np.ndarray, key: Optional[str]) -> str:
        if key:
            return key
        arr = np.ascontiguousarray(load_image(image))
        return "sha1_" + hashlib.sha1(arr.tobytes() + str(arr.shape).encode()).hexdigest()

    def predict(self, image: np.ndarray, key: Optional[str] = None) -> SegmentationResult:
        from PIL import Image

        path = self.dir / f"{self._key(image, key)}.png"
        if path.is_file():
            self.hits += 1
            return SegmentationResult(load_label_map(path), self.label_set, self.name, self.role, {"cache": str(path)})
        self.misses += 1
        result = self.inner.predict(image, key)
        tmp = path.with_suffix(".tmp.png")
        Image.fromarray(result.labels).save(tmp)
        tmp.replace(path)  # atomic: a Colab disconnect never leaves half-written cache files
        return result

    def describe(self) -> Dict:
        return {**self.inner.describe(), "cached_in": str(self.dir)}


# --------------------------------------------------------------------------------------
# Factory and label conversion
# --------------------------------------------------------------------------------------

BACKENDS = {
    "schp_lip": "clothing",
    "schp_atr": "clothing",
    "segformer_clothes": "clothing",
    "annotation_parsing": "clothing",
    "schp_pascal": "body",
    "annotation_densepose": "body",
}


def build_backend(name: str, cfg: ProjectConfig, role: Optional[str] = None, use_cache: Optional[bool] = None) -> SegmentationBackend:
    if name not in BACKENDS:
        raise KeyError(f"Unknown segmentation backend '{name}'. Available: {sorted(BACKENDS)}")
    if role and BACKENDS[name] != role:
        raise BackendError(f"Backend '{name}' produces {BACKENDS[name]} maps but was configured as the {role} backend")
    device = select_device(cfg.segmentation.device)
    if name.startswith("schp_"):
        path = ensure_weights(name, Path(cfg.paths.pretrained_dir), cfg.segmentation.allow_weight_download)
        backend: SegmentationBackend = SCHPBackend(name, path, device, BACKENDS[name])
    elif name == "segformer_clothes":
        backend = SegFormerClothesBackend(device, cfg.segmentation.allow_weight_download)
    else:
        backend = AnnotationBackend(name.split("_", 1)[1], Path(cfg.data.root), cfg.data.layout)
    cache = cfg.segmentation.cache_outputs if use_cache is None else use_cache
    if cache and not name.startswith("annotation_"):
        backend = CachedBackend(backend, Path(cfg.paths.cache_dir) / "segmentation")
    return backend


def to_project_labels(result: SegmentationResult, target_set: str) -> np.ndarray:
    """Convert a backend result into a project label set, with an explanation if impossible."""
    if result.label_set == target_set:
        return result.labels.copy()
    if lut_for(result.label_set, target_set) is None:
        hint = ""
        if result.label_set == "pascal_person_part" and target_set == "body15":
            hint = (" schp_pascal has only 7 coarse parts and cannot produce body15. Either use "
                    "annotation_densepose, or set labels.body_label_set=pascal_person_part so that "
                    "training data (DensePose) is converted to the same 7 parts.")
        raise LabelError(f"Cannot convert '{result.label_set}' maps from '{result.backend}' to '{target_set}'.{hint}")
    return convert_labels(result.labels, result.label_set, target_set)
