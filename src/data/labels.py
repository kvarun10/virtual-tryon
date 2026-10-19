"""Semantic label sets, conversions between them, and colour palettes.

Every segmentation map in this project is an ``H x W`` integer array whose values index a
:class:`LabelSet`. Source models / datasets each have their own label set; the warp network
works in two *project* label sets:

* ``garment12`` - clothing / identity layout (C_seg, C'_seg). 12 classes.
* ``body15``    - clothing-independent body parts (B_seg). 14 parts + background, derived from
                  DensePose. A substitute for NBF's 12-part segmentation (paper Sec. 4.3.1).
* ``pascal_person_part`` (7 classes) is the coarser body representation produced by SCHP.

Conversions are lookup tables (LUTs): ``converted = lut[label_map]``. Every source label must
be mapped explicitly; a missing entry is a programming error and raises at import time.

Sources for the native label lists (verified during Phase 3, Oct 2026):
* LIP / ATR / Pascal-Person-Part: SCHP ``simple_extractor.py`` (github.com/PeikeLi/Self-Correction-Human-Parsing)
* DeepFashion-MultiModal: dataset README (github.com/yumingj/DeepFashion-MultiModal)
* DensePose fine part index I (1..24): DensePose documentation (facebookresearch/DensePose).
  The DFMM PNG channel that stores I is taken from the Text2Human loader and must be checked on
  real files (``scripts/prepare_dataset.py --inspect`` reports the values found).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LabelSet:
    name: str
    names: Tuple[str, ...]
    palette: Tuple[Tuple[int, int, int], ...]
    source: str

    def __post_init__(self):
        if len(self.names) != len(self.palette):
            raise ValueError(f"{self.name}: {len(self.names)} names but {len(self.palette)} colours")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"{self.name}: duplicate class names")

    @property
    def num_classes(self) -> int:
        return len(self.names)

    def index(self, class_name: str) -> int:
        try:
            return self.names.index(class_name)
        except ValueError:
            raise KeyError(f"Class '{class_name}' not in label set '{self.name}': {self.names}") from None

    def indices(self, class_names: Iterable[str]) -> List[int]:
        return [self.index(n) for n in class_names]

    def palette_array(self) -> np.ndarray:
        return np.asarray(self.palette, dtype=np.uint8)


def voc_palette(n: int) -> Tuple[Tuple[int, int, int], ...]:
    """Pascal-VOC style palette (the one SCHP's simple_extractor attaches to its PNGs)."""
    out = []
    for j in range(n):
        lab, r, g, b, i = j, 0, 0, 0, 0
        while lab:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
        out.append((r, g, b))
    return tuple(out)


# --------------------------------------------------------------------------------------
# Native label sets
# --------------------------------------------------------------------------------------

LIP_NAMES = (
    "background", "hat", "hair", "glove", "sunglasses", "upper_clothes", "dress", "coat",
    "socks", "pants", "jumpsuits", "scarf", "skirt", "face", "left_arm", "right_arm",
    "left_leg", "right_leg", "left_shoe", "right_shoe",
)
ATR_NAMES = (
    "background", "hat", "hair", "sunglasses", "upper_clothes", "skirt", "pants", "dress",
    "belt", "left_shoe", "right_shoe", "face", "left_leg", "right_leg", "left_arm",
    "right_arm", "bag", "scarf",
)
PASCAL_PERSON_PART_NAMES = (
    "background", "head", "torso", "upper_arms", "lower_arms", "upper_legs", "lower_legs",
)
DFMM_NAMES = (
    "background", "top", "outer", "skirt", "dress", "pants", "leggings", "headwear",
    "eyeglass", "neckwear", "belt", "footwear", "bag", "hair", "face", "skin", "ring",
    "wrist_wearing", "socks", "gloves", "necklace", "rompers", "earrings", "tie",
)
# DensePose part index I (0 = background). Each body part has two indices (e.g. 1 and 2 are
# both torso); the docs do not name which is front/back, so they get neutral _1/_2 suffixes.
DENSEPOSE_NAMES = (
    "background", "torso_1", "torso_2", "right_hand", "left_hand", "left_foot",
    "right_foot", "upper_leg_right_1", "upper_leg_left_1", "upper_leg_right_2",
    "upper_leg_left_2", "lower_leg_right_1", "lower_leg_left_1",
    "lower_leg_right_2", "lower_leg_left_2", "upper_arm_left_1",
    "upper_arm_right_1", "upper_arm_left_2", "upper_arm_right_2",
    "lower_arm_left_1", "lower_arm_right_1", "lower_arm_left_2",
    "lower_arm_right_2", "head_1", "head_2",
)

# --------------------------------------------------------------------------------------
# Project label sets
# --------------------------------------------------------------------------------------

GARMENT12_NAMES = (
    "background", "hair", "face", "skin", "top", "outer", "dress", "skirt", "pants",
    "footwear", "headwear", "accessory",
)
GARMENT12_PALETTE = (
    (0, 0, 0), (128, 64, 0), (255, 200, 150), (230, 160, 120), (220, 40, 40),
    (40, 120, 220), (200, 60, 200), (250, 170, 30), (40, 160, 70), (90, 90, 90),
    (150, 230, 230), (255, 255, 0),
)

BODY15_NAMES = (
    "background", "head", "torso", "right_upper_arm", "left_upper_arm", "right_lower_arm",
    "left_lower_arm", "right_hand", "left_hand", "right_upper_leg", "left_upper_leg",
    "right_lower_leg", "left_lower_leg", "right_foot", "left_foot",
)
BODY15_PALETTE = (
    (0, 0, 0), (255, 200, 150), (220, 40, 40), (40, 120, 220), (120, 180, 255),
    (40, 160, 70), (140, 220, 140), (250, 170, 30), (255, 220, 120), (200, 60, 200),
    (240, 150, 240), (90, 60, 160), (170, 140, 230), (110, 110, 110), (190, 190, 190),
)

LABEL_SETS: Dict[str, LabelSet] = {
    "lip": LabelSet("lip", LIP_NAMES, voc_palette(len(LIP_NAMES)), "SCHP / LIP"),
    "atr": LabelSet("atr", ATR_NAMES, voc_palette(len(ATR_NAMES)), "SCHP / ATR"),
    "pascal_person_part": LabelSet(
        "pascal_person_part", PASCAL_PERSON_PART_NAMES, voc_palette(7), "SCHP / Pascal-Person-Part"
    ),
    "dfmm": LabelSet("dfmm", DFMM_NAMES, voc_palette(len(DFMM_NAMES)), "DeepFashion-MultiModal README"),
    "densepose": LabelSet("densepose", DENSEPOSE_NAMES, voc_palette(len(DENSEPOSE_NAMES)), "DensePose part index I"),
    "garment12": LabelSet("garment12", GARMENT12_NAMES, GARMENT12_PALETTE, "project"),
    "body15": LabelSet("body15", BODY15_NAMES, BODY15_PALETTE, "project (from DensePose)"),
}


def get_label_set(name: str) -> LabelSet:
    try:
        return LABEL_SETS[name]
    except KeyError:
        raise KeyError(f"Unknown label set '{name}'. Available: {sorted(LABEL_SETS)}") from None


# --------------------------------------------------------------------------------------
# Mappings (source class name -> target class name)
# --------------------------------------------------------------------------------------

_MAPPINGS: Dict[Tuple[str, str], Dict[str, str]] = {
    ("lip", "garment12"): {
        "background": "background", "hat": "headwear", "hair": "hair", "glove": "accessory",
        "sunglasses": "accessory", "upper_clothes": "top", "dress": "dress", "coat": "outer",
        "socks": "footwear", "pants": "pants", "jumpsuits": "dress", "scarf": "accessory",
        "skirt": "skirt", "face": "face", "left_arm": "skin", "right_arm": "skin",
        "left_leg": "skin", "right_leg": "skin", "left_shoe": "footwear", "right_shoe": "footwear",
    },
    # NOTE: ATR has no coat/outer class - jackets are labelled upper_clothes, so 'outer' never
    # appears in ATR-derived maps. Prefer LIP for C_seg.
    ("atr", "garment12"): {
        "background": "background", "hat": "headwear", "hair": "hair", "sunglasses": "accessory",
        "upper_clothes": "top", "skirt": "skirt", "pants": "pants", "dress": "dress",
        "belt": "accessory", "left_shoe": "footwear", "right_shoe": "footwear", "face": "face",
        "left_leg": "skin", "right_leg": "skin", "left_arm": "skin", "right_arm": "skin",
        "bag": "accessory", "scarf": "accessory",
    },
    ("dfmm", "garment12"): {
        "background": "background", "top": "top", "outer": "outer", "skirt": "skirt",
        "dress": "dress", "pants": "pants", "leggings": "pants", "headwear": "headwear",
        "eyeglass": "accessory", "neckwear": "accessory", "belt": "accessory",
        "footwear": "footwear", "bag": "accessory", "hair": "hair", "face": "face",
        "skin": "skin", "ring": "accessory", "wrist_wearing": "accessory", "socks": "footwear",
        "gloves": "accessory", "necklace": "accessory", "rompers": "dress",
        "earrings": "accessory", "tie": "accessory",
    },
    ("densepose", "body15"): {
        "background": "background", "torso_1": "torso", "torso_2": "torso",
        "right_hand": "right_hand", "left_hand": "left_hand", "left_foot": "left_foot",
        "right_foot": "right_foot", "upper_leg_right_1": "right_upper_leg",
        "upper_leg_left_1": "left_upper_leg", "upper_leg_right_2": "right_upper_leg",
        "upper_leg_left_2": "left_upper_leg", "lower_leg_right_1": "right_lower_leg",
        "lower_leg_left_1": "left_lower_leg", "lower_leg_right_2": "right_lower_leg",
        "lower_leg_left_2": "left_lower_leg", "upper_arm_left_1": "left_upper_arm",
        "upper_arm_right_1": "right_upper_arm", "upper_arm_left_2": "left_upper_arm",
        "upper_arm_right_2": "right_upper_arm", "lower_arm_left_1": "left_lower_arm",
        "lower_arm_right_1": "right_lower_arm", "lower_arm_left_2": "left_lower_arm",
        "lower_arm_right_2": "right_lower_arm", "head_1": "head", "head_2": "head",
    },
    # Lets a model trained on DensePose-derived parts be compared with / fed by SCHP-PPP.
    ("body15", "pascal_person_part"): {
        "background": "background", "head": "head", "torso": "torso",
        "right_upper_arm": "upper_arms", "left_upper_arm": "upper_arms",
        "right_lower_arm": "lower_arms", "left_lower_arm": "lower_arms",
        "right_hand": "lower_arms", "left_hand": "lower_arms",
        "right_upper_leg": "upper_legs", "left_upper_leg": "upper_legs",
        "right_lower_leg": "lower_legs", "left_lower_leg": "lower_legs",
        "right_foot": "lower_legs", "left_foot": "lower_legs",
    },
}


def _build_lut(src: LabelSet, dst: LabelSet, mapping: Dict[str, str]) -> np.ndarray:
    missing = [n for n in src.names if n not in mapping]
    extra = [n for n in mapping if n not in src.names]
    if missing or extra:
        raise ValueError(f"Mapping {src.name}->{dst.name}: missing {missing}, unknown {extra}")
    return np.asarray([dst.index(mapping[n]) for n in src.names], dtype=np.uint8)


_LUTS: Dict[Tuple[str, str], np.ndarray] = {}
for (_s, _d), _m in _MAPPINGS.items():
    _LUTS[(_s, _d)] = _build_lut(LABEL_SETS[_s], LABEL_SETS[_d], _m)
# Composite: densepose -> pascal_person_part
_LUTS[("densepose", "pascal_person_part")] = _LUTS[("body15", "pascal_person_part")][_LUTS[("densepose", "body15")]]


def available_conversions() -> List[Tuple[str, str]]:
    return sorted(_LUTS)


class LabelError(ValueError):
    """Raised when a label map is incompatible with its declared label set."""


def check_label_map(label_map: np.ndarray, label_set: LabelSet, what: str = "label map") -> None:
    if label_map.ndim != 2:
        raise LabelError(
            f"{what} must be a 2-D array of class indices, got shape {label_map.shape}. "
            "A 3-channel array usually means a colour-coded PNG was loaded instead of the raw labels."
        )
    if not np.issubdtype(label_map.dtype, np.integer):
        raise LabelError(f"{what} must have an integer dtype, got {label_map.dtype}")
    if label_map.size and (label_map.min() < 0 or label_map.max() >= label_set.num_classes):
        bad = np.unique(label_map[(label_map < 0) | (label_map >= label_set.num_classes)])
        raise LabelError(
            f"{what} contains values {bad.tolist()[:10]} outside label set '{label_set.name}' "
            f"(0..{label_set.num_classes - 1}). Check that the right label set / backend is configured."
        )


def convert_labels(label_map: np.ndarray, src: str, dst: str) -> np.ndarray:
    """Convert an integer label map between label sets via a LUT."""
    if src == dst:
        check_label_map(label_map, get_label_set(src))
        return label_map.astype(np.uint8, copy=True)
    key = (src, dst)
    if key not in _LUTS:
        raise KeyError(f"No conversion {src} -> {dst}. Available: {available_conversions()}")
    check_label_map(label_map, get_label_set(src), f"'{src}' label map")
    return _LUTS[key][label_map]


def keep_classes(label_map: np.ndarray, label_set: LabelSet, class_names: Sequence[str], fill: int = 0) -> np.ndarray:
    """Keep the listed classes; every other pixel becomes ``fill`` (background)."""
    keep = np.isin(label_map, label_set.indices(class_names))
    out = np.full_like(label_map, fill)
    out[keep] = label_map[keep]
    return out


def colorize(label_map: np.ndarray, label_set: LabelSet) -> np.ndarray:
    """Integer map -> ``H x W x 3`` uint8 RGB image using the label set's palette."""
    check_label_map(label_map, label_set)
    return label_set.palette_array()[label_map]


def decode_color_map(rgb: np.ndarray, label_set: LabelSet, tolerance: int = 0) -> np.ndarray:
    """Inverse of :func:`colorize` for colour-coded maps (e.g. the paper's 3-channel inputs).

    Raises if any pixel colour is not in the palette (within ``tolerance``), which happens with
    JPEG-compressed or anti-aliased colour maps.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise LabelError(f"Expected H x W x 3 colour map, got {rgb.shape}")
    pal = label_set.palette_array().astype(np.int32)
    if len({tuple(c) for c in pal.tolist()}) != len(pal):
        raise LabelError(f"Palette of '{label_set.name}' has duplicate colours; cannot decode")
    dist = np.abs(rgb.astype(np.int32)[:, :, None, :] - pal[None, None]).max(axis=-1)
    idx = dist.argmin(axis=-1)
    if (dist.min(axis=-1) > tolerance).any():
        n_bad = int((dist.min(axis=-1) > tolerance).sum())
        raise LabelError(f"{n_bad} pixels do not match any '{label_set.name}' palette colour")
    return idx.astype(np.uint8)


def one_hot(label_map: np.ndarray, num_classes: int) -> np.ndarray:
    """``H x W`` -> ``C x H x W`` float32 one-hot."""
    if label_map.max(initial=0) >= num_classes:
        raise LabelError(f"label value {label_map.max()} >= num_classes {num_classes}")
    return np.eye(num_classes, dtype=np.float32)[label_map].transpose(2, 0, 1)


def class_histogram(label_map: np.ndarray, label_set: LabelSet) -> Dict[str, int]:
    counts = np.bincount(label_map.ravel(), minlength=label_set.num_classes)
    return {name: int(c) for name, c in zip(label_set.names, counts) if c}


def present_classes(label_map: np.ndarray, label_set: LabelSet) -> List[str]:
    return [label_set.names[i] for i in np.unique(label_map)]


def inject_identity_labels(
    garment_map: np.ndarray,
    target_map: np.ndarray,
    label_set: LabelSet,
    preserve_classes: Sequence[str],
    clear_existing: bool = True,
) -> np.ndarray:
    """Paper Sec. 4.2: replace face/hair segments with those of the target person.

    ``garment_map`` is the (warped) clothing layout, ``target_map`` the target's parsing in the
    same label set. With ``clear_existing`` the reference person's own face/hair pixels are
    removed first (set to background) so two faces cannot coexist.
    """
    if garment_map.shape != target_map.shape:
        raise LabelError(f"Shape mismatch {garment_map.shape} vs {target_map.shape}")
    ids = label_set.indices(preserve_classes)
    out = garment_map.copy()
    if clear_existing:
        out[np.isin(out, ids)] = label_set.index("background")
    mask = np.isin(target_map, ids)
    out[mask] = target_map[mask]
    return out


def lut_for(src: str, dst: str) -> Optional[np.ndarray]:
    return None if src == dst else _LUTS.get((src, dst))
