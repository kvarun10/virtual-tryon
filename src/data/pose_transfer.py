"""Standard DeepFashion pose-transfer pairs (PATN, Zhu et al. CVPR 2019), the test set used by ADGAN, DiOr and the
Dress-up paper's Table 1 (which does not name its pairs; this is the common public protocol).

Files (from the Google Drive folder linked in github.com/tengteng95/Pose-Transfer, MIT licence; downloaded with the
user's approval on 2026-10-26, verified by SHA-256 below):

* ``fasion-resize-pairs-test.csv``: 8,570 pairs (``from,to``), 802,612 bytes
* ``fasion-resize-pairs-train.csv``: 101,966 pairs, 9,551,830 bytes

Image names look like ``fashionWOMENBlouses_Shirtsid0000337203_3back.jpg`` = gender WOMEN, category Blouses_Shirts,
item 00003372, photo set 03, view 3_back, i.e. DFMM ``WOMEN-Blouses_Shirts-id_00003372-03_3_back``.

Protocol images are 176 × 256 (W × H): the In-shop photo fitted to 256 px height and centre-cropped/padded to 176.
DFMM photos (750 × 1101) have the same aspect ratio (0.68), so :func:`to_protocol_frame` letterboxes into 176 × 256.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.data.preprocessing import resize_image

PROTOCOL_SIZE = (256, 176)  # (H, W)
PAIR_FILES = {
    "test": {"name": "fasion-resize-pairs-test.csv", "gdrive_id": "1T2hIJMFvDIuReXO1g8P8kzqdCg5DaS6m", "bytes": 802612,
             "sha256": "3da031d5b376d544a1cc6e24f2618d9909e8d3bd4f9f44d7f355c9e02be644ba", "pairs": 8570},
    "train": {"name": "fasion-resize-pairs-train.csv", "gdrive_id": "1f_HIII0nQVf0asAs-CiAjPL0bEow-aAt", "bytes": 9551830,
              "sha256": "733435ff454d2612aa1898c6887870aec547abb29a146ea64627d087e77bff6d", "pairs": 101966},
}
SOURCE = "https://github.com/tengteng95/Pose-Transfer (Google Drive folder 1YMsYXc41dR3k8YroXeWGh9zweNUQmZBw)"
_NAME = re.compile(r"^fashion(?P<gender>WOMEN|MEN)(?P<category>.+?)id(?P<item>\d{8})(?P<seq>\d{2})_(?P<idx>\d+)(?P<view>[A-Za-z]+)\.jpg$")


class PairFilesUnavailableError(FileNotFoundError):
    pass


def to_dfmm_stem(name: str) -> str:
    m = _NAME.match(name.strip())
    if not m:
        raise ValueError(f"not a pose-transfer image name: {name!r}")
    d = m.groupdict()
    return f"{d['gender']}-{d['category']}-id_{d['item']}-{d['seq']}_{d['idx']}_{d['view']}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_pair_files(directory: Path, allow_download: bool = False) -> Dict[str, Path]:
    """Return verified paths of both pair files; download them (only these two files) if allowed and missing."""
    directory = Path(directory)
    out = {}
    for split, spec in PAIR_FILES.items():
        path = directory / spec["name"]
        if not path.is_file():
            if not allow_download:
                raise PairFilesUnavailableError(
                    f"{path} missing ({spec['bytes'] / 1e6:.1f} MB, {SOURCE}). Pass allow_download=True "
                    "(notebook: ALLOW_PAIR_DOWNLOAD=True) to fetch it once.")
            import gdown

            directory.mkdir(parents=True, exist_ok=True)
            gdown.download(id=spec["gdrive_id"], output=str(path), quiet=True)
        if _sha256(path) != spec["sha256"]:
            raise ValueError(f"{path}: SHA-256 mismatch (expected {spec['sha256'][:12]}...). Delete it and download again.")
        out[split] = path
    return out


def read_pairs(path: Path) -> List[Tuple[str, str]]:
    """``(reference_stem, target_stem)`` in DFMM naming."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [(to_dfmm_stem(r["from"]), to_dfmm_stem(r["to"])) for r in rows]


def item_of(stem: str) -> str:
    return stem.split("-")[2].split("_")[-1]


CATEGORIES = ("clean_unseen", "clean_val_items", "our_test_items_LOCKED", "our_train_items_LEAKED", "image_missing")


def categorise_pairs(pairs: Iterable[Tuple[str, str]], split_names: Dict[str, List[str]],
                     available: Optional[set] = None) -> Dict[Tuple[str, str], str]:
    """Leakage category of every pair, by ITEM (product), the unit our splits are disjoint on.

    * ``our_train_items_LEAKED``: an item of either photo was used to train our networks -> never scored
    * ``our_test_items_LOCKED``: an item of our locked test split -> not scored (test is used once)
    * ``clean_val_items``: items of our validation split (used only for model selection) -> scored
    * ``clean_unseen``: items in none of our splits (no manual parsing, never used) -> scored
    * ``image_missing``: a photo is not in DFMM (``available``), cannot be loaded
    """
    item_split = {}
    for split, names in split_names.items():
        for n in names:
            item_split[item_of(n)] = split
    out = {}
    for ref, tgt in pairs:
        if available is not None and not (ref in available and tgt in available):
            out[(ref, tgt)] = "image_missing"
            continue
        splits = {item_split.get(item_of(ref)), item_split.get(item_of(tgt))}
        if "train" in splits:
            out[(ref, tgt)] = "our_train_items_LEAKED"
        elif "test" in splits:
            out[(ref, tgt)] = "our_test_items_LOCKED"
        elif "val" in splits:
            out[(ref, tgt)] = "clean_val_items"
        else:
            out[(ref, tgt)] = "clean_unseen"
    return out


def leakage_report(test_pairs, train_pairs, split_names: Dict[str, List[str]], available: Optional[set] = None) -> Dict:
    cats = categorise_pairs(test_pairs, split_names, available)
    counts = Counter(cats.values())
    test_items = {item_of(s) for p in test_pairs for s in p}
    train_items_std = {item_of(s) for p in train_pairs for s in p}
    ours = {s: {item_of(n) for n in names} for s, names in split_names.items()}
    return {
        "standard_test_pairs": len(test_pairs), "standard_test_items": len(test_items),
        "standard_train_pairs": len(train_pairs), "standard_train_items": len(train_items_std),
        "standard_train_test_item_overlap": len(test_items & train_items_std),
        "pairs_by_category": {c: counts.get(c, 0) for c in CATEGORIES},
        "standard_test_items_in_our_splits": {s: len(test_items & it) for s, it in ours.items()},
        "scorable_now": counts.get("clean_unseen", 0) + counts.get("clean_val_items", 0),
        "note": "Scorable = items never used for training and not in our locked test split. A full 8,570-pair benchmark "
                "needs networks retrained with the standard test items excluded from training.",
    }


def to_protocol_frame(image: np.ndarray) -> np.ndarray:
    """Full photo -> 176 × 256 protocol image (letterboxed, white pad)."""
    return resize_image(image, PROTOCOL_SIZE, "letterbox", 255)[0]


def model_output_to_protocol(output: np.ndarray, info) -> np.ndarray:
    """Crop a letterboxed model-size output to its content (``info`` from the target's letterbox) and fit it into
    the 176 × 256 protocol frame, so it aligns pixel-for-pixel with :func:`to_protocol_frame` of the real photo."""
    ch, cw = info.content_size
    crop = np.ascontiguousarray(output[info.pad_top:info.pad_top + ch, info.pad_left:info.pad_left + cw])
    return resize_image(crop, PROTOCOL_SIZE, "letterbox", 255)[0]


def load_split_names(split_file: Path) -> Dict[str, List[str]]:
    return json.loads(Path(split_file).read_text(encoding="utf-8"))["splits"]
