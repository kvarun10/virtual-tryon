"""Reference-image handling for the two reference modes.

Mode "person_worn"  (paper setting, Dress-up Sec. 4.3.1)
    The reference photo shows a person wearing the garment. A human parser segments it, the
    result is mapped to ``garment12`` and only the configured ``transfer_classes`` are kept -> C_seg.

Mode "garment_only"  (NOT in the paper; experimental extension)
    The reference is a product photo of an isolated garment. Human parsers are trained on people
    and are not expected to work here, so a separate branch is used:
      1. background removal for plain (near-uniform) studio backgrounds,
      2. the garment category is supplied by the user (no classifier is trained),
      3. the mask is labelled with that category -> a *flat* C_seg in product-photo geometry.
    A flat product layout is NOT the same representation the person-worn warp network is trained
    on (a garment worn by a body). Converting it requires a warp model trained on product/worn
    pairs (VITON-HD, DressCode), which DeepFashion-MultiModal does not contain. Hence
    ``PreparedReference.validated`` is False for this branch and downstream stages must refuse
    it until such a model exists. See docs/clothing_only_mode.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.config import ProjectConfig
from src.data.labels import get_label_set, keep_classes
from src.data.preprocessing import load_image
from src.models.segmentation import SegmentationBackend, to_project_labels

REFERENCE_MODES = ("person_worn", "garment_only")
GARMENT_ONLY_CATEGORIES = ("top", "outer", "dress", "skirt", "pants")


class ReferenceError(ValueError):
    """Raised when a reference image cannot be turned into a usable C_seg."""


@dataclass
class PreparedReference:
    mode: str
    image: np.ndarray  # H x W x 3 RGB uint8 (original resolution)
    garment_map: np.ndarray  # H x W garment12 labels, transfer classes only (C_seg)
    full_parsing: Optional[np.ndarray]  # H x W garment12 labels incl. face/hair (person_worn only)
    garment_mask: np.ndarray  # H x W bool
    label_set: str = "garment12"
    validated: bool = True  # False = experimental branch, not validated with downstream stages
    warnings: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)


class ReferencePreprocessor(ABC):
    mode: str

    @abstractmethod
    def prepare(self, image, key: Optional[str] = None, garment_category: Optional[str] = None) -> PreparedReference:
        ...


class PersonWornPreprocessor(ReferencePreprocessor):
    mode = "person_worn"

    def __init__(self, clothing_backend: SegmentationBackend, cfg: ProjectConfig):
        if clothing_backend.role != "clothing":
            raise ReferenceError(f"Backend '{clothing_backend.name}' is a {clothing_backend.role} backend")
        self.backend = clothing_backend
        self.cfg = cfg
        self.label_set = get_label_set(cfg.labels.garment_label_set)

    def prepare(self, image, key: Optional[str] = None, garment_category: Optional[str] = None) -> PreparedReference:
        image = load_image(image)
        result = self.backend.predict(image, key)
        parsing = to_project_labels(result, self.label_set.name)
        garment = keep_classes(parsing, self.label_set, self.cfg.labels.transfer_classes)
        mask = garment > 0
        warnings = []
        if garment_category is not None:
            warnings.append("garment_category is ignored in person_worn mode (classes come from the parser)")
        frac = float(mask.mean())
        if frac < self.cfg.reference.min_garment_fraction:
            raise ReferenceError(
                f"Only {frac:.1%} of the reference image was recognised as transferable clothing "
                f"({self.cfg.labels.transfer_classes}). Is there a clearly visible person wearing the garment? "
                "For product photos use reference.mode=garment_only."
            )
        return PreparedReference(self.mode, image, garment, parsing, mask, self.label_set.name, True, warnings,
                                 {"backend": result.backend, "native_label_set": result.label_set})


def remove_plain_background(image: np.ndarray, tolerance: float = 30.0, border: int = 4) -> np.ndarray:
    """Foreground mask for product photos on a near-uniform background.

    Method (deterministic, no learned model): estimate the background colour as the per-channel
    median of a ``border``-pixel frame, mark pixels whose max-channel distance exceeds
    ``tolerance`` as foreground, keep the largest connected component, fill interior holes (so a
    white logo inside a white-background shirt is kept), and smooth with a 3x3 opening/closing.

    Limitations: fails on textured / gradient backgrounds, shadows and garments whose colour is
    close to the background. A learned matting model (e.g. U^2-Net/rembg) is a possible upgrade
    and is listed as an experiment, not used here.
    """
    from scipy import ndimage

    image = load_image(image)
    h, w = image.shape[:2]
    b = max(1, min(border, h // 4, w // 4))
    frame = np.concatenate([image[:b].reshape(-1, 3), image[-b:].reshape(-1, 3),
                            image[:, :b].reshape(-1, 3), image[:, -b:].reshape(-1, 3)])
    bg = np.median(frame, axis=0)
    spread = np.abs(frame.astype(np.float32) - bg).max(axis=1)
    if np.percentile(spread, 90) > tolerance:
        raise ReferenceError(
            "The image border is not a uniform background (90th percentile colour deviation "
            f"{np.percentile(spread, 90):.0f} > tolerance {tolerance}). garment_only mode currently "
            "supports plain studio backgrounds only."
        )
    dist = np.abs(image.astype(np.float32) - bg).max(axis=2)
    fg = dist > tolerance
    fg = ndimage.binary_opening(fg, structure=np.ones((3, 3)))
    labels, n = ndimage.label(fg)
    if n == 0:
        return np.zeros((h, w), dtype=bool)
    sizes = ndimage.sum(fg, labels, index=range(1, n + 1))
    largest = labels == (int(np.argmax(sizes)) + 1)
    largest = ndimage.binary_closing(largest, structure=np.ones((3, 3)))
    return ndimage.binary_fill_holes(largest)


class GarmentOnlyPreprocessor(ReferencePreprocessor):
    """EXPERIMENTAL. Produces a flat-layout C_seg from a product photo (see module docstring)."""

    mode = "garment_only"

    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self.label_set = get_label_set(cfg.labels.garment_label_set)

    def prepare(self, image, key: Optional[str] = None, garment_category: Optional[str] = None) -> PreparedReference:
        category = garment_category or self.cfg.reference.garment_category
        if category is None:
            raise ReferenceError(
                f"garment_only mode needs the garment category (one of {GARMENT_ONLY_CATEGORIES}); "
                "set reference.garment_category or pass garment_category=..."
            )
        if category not in GARMENT_ONLY_CATEGORIES:
            raise ReferenceError(f"Unsupported garment category '{category}'. Use one of {GARMENT_ONLY_CATEGORIES}")
        if category not in self.cfg.labels.transfer_classes:
            raise ReferenceError(f"Category '{category}' is not in labels.transfer_classes")
        image = load_image(image)
        mask = remove_plain_background(image, self.cfg.reference.background_tolerance)
        frac = float(mask.mean())
        if frac < self.cfg.reference.min_garment_fraction:
            raise ReferenceError(f"Background removal kept only {frac:.1%} of the image; no garment found")
        if frac > 0.95:
            raise ReferenceError("Background removal kept >95% of the image; the background is probably not plain")
        garment = np.zeros(mask.shape, dtype=np.uint8)
        garment[mask] = self.label_set.index(category)
        return PreparedReference(
            self.mode, image, garment, None, mask, self.label_set.name, validated=False,
            warnings=["garment_only is experimental: the flat product layout is not compatible with the "
                      "person-worn warp network; no garment-only warp/texture model exists yet."],
            meta={"category": category, "method": "plain-background threshold"},
        )


def build_reference_preprocessor(cfg: ProjectConfig, clothing_backend: Optional[SegmentationBackend] = None,
                                 mode: Optional[str] = None) -> ReferencePreprocessor:
    mode = mode or cfg.reference.mode
    if mode == "person_worn":
        if clothing_backend is None:
            raise ReferenceError("person_worn mode needs a clothing segmentation backend")
        return PersonWornPreprocessor(clothing_backend, cfg)
    if mode == "garment_only":
        return GarmentOnlyPreprocessor(cfg)
    raise ReferenceError(f"Unknown reference mode '{mode}'. Use one of {REFERENCE_MODES}")
