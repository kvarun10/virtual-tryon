"""DeepFashion-MultiModal (DFMM) indexing, validation and statistics.

Verified facts (README of github.com/yumingj/DeepFashion-MultiModal, Oct 2026)
--------------------------------------------------------------------------------
* 44,096 JPG images of size 750 x 1101 "from DeepFashion"; 12,701 are full-body.
* Manually annotated 24-class human parsing for the 12,701 full-body images (PNG).
* 21 manually annotated keypoints; ``keypoints_loc.txt`` / ``keypoints_vis.txt``.
* DensePose extracted for all 44,096 images (PNG).
* Licence: non-commercial research only, no redistribution.
* It contains people only - there are NO isolated garment/product images.

From related code (not the README)
----------------------------------
* Annotation files are named ``<image stem>_segm.png`` / ``<image stem>_densepose.png`` and the
  DensePose part index is read from channel 2 (Text2Human ``parsing_generation_segm_attr_dataset.py``).
* Image names follow the DeepFashion In-shop pattern
  ``<GENDER>-<Category>-id_<8 digits>-<NN>_<view#>_<view>.jpg``; DiOr's download script converts
  them to the In-shop pose-transfer naming, i.e. DFMM images are In-shop images.

Everything here tolerates deviations (missing files, other names) and *reports* them instead of
assuming; run :func:`inspect_root` on a fresh download first.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.config import DataLayoutConfig
from src.data.labels import get_label_set
from src.data.preprocessing import ImageLoadError, image_full_ok, image_header_ok, load_label_map

NAME_PATTERN = re.compile(
    r"^(?P<gender>MEN|WOMEN)-(?P<category>[^-]+)-id_(?P<item>\d{8})-(?P<seq>\d{2})_(?P<view_idx>\d+)_(?P<view>[A-Za-z]+)$"
)


@dataclass
class ImageRecord:
    name: str  # image stem (no extension)
    image: str  # path relative to data root
    parsing: Optional[str] = None
    densepose: Optional[str] = None
    gender: Optional[str] = None
    category: Optional[str] = None
    item_id: Optional[str] = None  # grouping key for leakage-free splits
    seq: Optional[str] = None
    view: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

    @property
    def group(self) -> str:
        """Split group. Unparseable names form their own group (conservative)."""
        return self.item_id if self.item_id else f"name:{self.name}"


def parse_name(stem: str) -> Optional[Dict[str, str]]:
    m = NAME_PATTERN.match(stem)
    if not m:
        return None
    d = m.groupdict()
    d["view"] = f"{d.pop('view_idx')}_{d['view']}"
    return d


# --------------------------------------------------------------------------------------
# Inspection of an unknown download
# --------------------------------------------------------------------------------------


def inspect_root(root: Path, max_examples: int = 3) -> Dict:
    """Describe the folders / file types under ``root`` (two levels) without reading pixels."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}. Download DeepFashion-MultiModal (see README, "
            "'Dataset') and set data.root."
        )
    report = {"root": str(root), "entries": []}
    for entry in sorted(root.iterdir()):
        if entry.is_file():
            report["entries"].append({"path": entry.name, "type": "file", "bytes": entry.stat().st_size})
            continue
        exts: Counter = Counter()
        examples: List[str] = []
        subdirs = 0
        for child in entry.iterdir():
            if child.is_dir():
                subdirs += 1
                continue
            exts[child.suffix.lower()] += 1
            if len(examples) < max_examples:
                examples.append(child.name)
        report["entries"].append(
            {"path": entry.name + "/", "type": "dir", "files_by_ext": dict(exts), "subdirs": subdirs, "examples": examples}
        )
    return report


def inspect_densepose_file(path: Path) -> Dict:
    """Report mode, shape and value range per channel of one DensePose PNG."""
    with Image.open(path) as img:
        mode = img.mode
        arr = np.asarray(img)
    info = {"mode": mode, "shape": list(arr.shape), "channels": []}
    if arr.ndim == 2:
        arr = arr[:, :, None]
    for c in range(arr.shape[2]):
        vals = np.unique(arr[:, :, c])
        info["channels"].append({"channel": c, "min": int(vals.min()), "max": int(vals.max()), "n_unique": int(vals.size)})
    return info


# --------------------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------------------


@dataclass
class ValidationReport:
    n_images: int = 0
    n_with_parsing: int = 0
    n_with_densepose: int = 0
    unparseable_names: List[str] = field(default_factory=list)
    corrupt_images: Dict[str, str] = field(default_factory=dict)
    orphan_parsing: List[str] = field(default_factory=list)  # annotation without image
    orphan_densepose: List[str] = field(default_factory=list)
    bad_parsing: Dict[str, str] = field(default_factory=dict)  # wrong size / labels
    bad_densepose: Dict[str, str] = field(default_factory=dict)
    size_counts: Dict[str, int] = field(default_factory=dict)
    checked_annotations: int = 0

    @property
    def ok(self) -> bool:
        return not (self.corrupt_images or self.bad_parsing or self.bad_densepose)

    def summary(self) -> str:
        lines = [
            f"images                : {self.n_images}",
            f"  with manual parsing : {self.n_with_parsing}",
            f"  with DensePose      : {self.n_with_densepose}",
            f"unparseable file names: {len(self.unparseable_names)}",
            f"corrupt images        : {len(self.corrupt_images)}",
            f"orphan parsing / DP   : {len(self.orphan_parsing)} / {len(self.orphan_densepose)}",
            f"bad parsing / DP maps : {len(self.bad_parsing)} / {len(self.bad_densepose)}"
            f"  (of {self.checked_annotations} annotation files checked)",
            f"image sizes (WxH)     : {self.size_counts}",
        ]
        return "\n".join(lines)


class DFMMIndex:
    """List of :class:`ImageRecord` built from a DFMM directory."""

    def __init__(self, root: Path, layout: DataLayoutConfig, records: List[ImageRecord]):
        self.root = Path(root)
        self.layout = layout
        self.records = records
        self._by_name = {r.name: r for r in records}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, name: str) -> ImageRecord:
        return self._by_name[name]

    # ---- construction ---------------------------------------------------------------

    @classmethod
    def build(cls, root: Path, layout: DataLayoutConfig) -> Tuple["DFMMIndex", ValidationReport]:
        root = Path(root)
        img_dir = root / layout.images_dir
        if not img_dir.is_dir():
            found = [p.name for p in root.iterdir()] if root.is_dir() else []
            raise FileNotFoundError(
                f"Image folder '{img_dir}' not found. Contents of {root}: {found}. "
                "Adjust data.layout.images_dir or check the download/unzip step."
            )
        parsing_dir = root / layout.parsing_dir
        dp_dir = root / layout.densepose_dir
        parsing_files = {p.name for p in parsing_dir.glob(f"*{layout.parsing_suffix}")} if parsing_dir.is_dir() else set()
        dp_files = {p.name for p in dp_dir.glob(f"*{layout.densepose_suffix}")} if dp_dir.is_dir() else set()

        report = ValidationReport()
        records: List[ImageRecord] = []
        for img_path in sorted(img_dir.glob(f"*{layout.image_ext}")):
            stem = img_path.stem
            meta = parse_name(stem)
            if meta is None:
                report.unparseable_names.append(img_path.name)
            seg_name = stem + layout.parsing_suffix
            dp_name = stem + layout.densepose_suffix
            rec = ImageRecord(
                name=stem,
                image=str(img_path.relative_to(root).as_posix()),
                parsing=f"{layout.parsing_dir}/{seg_name}" if seg_name in parsing_files else None,
                densepose=f"{layout.densepose_dir}/{dp_name}" if dp_name in dp_files else None,
                gender=meta and meta["gender"],
                category=meta and meta["category"],
                item_id=meta and meta["item"],
                seq=meta and meta["seq"],
                view=meta and meta["view"],
            )
            records.append(rec)
        names = {r.name for r in records}
        report.orphan_parsing = sorted(f for f in parsing_files if f[: -len(layout.parsing_suffix)] not in names)
        report.orphan_densepose = sorted(f for f in dp_files if f[: -len(layout.densepose_suffix)] not in names)
        report.n_images = len(records)
        report.n_with_parsing = sum(r.parsing is not None for r in records)
        report.n_with_densepose = sum(r.densepose is not None for r in records)
        return cls(root, layout, records), report

    # ---- validation -----------------------------------------------------------------

    def validate(
        self,
        report: ValidationReport,
        image_check: str = "header",
        annotation_sample: Optional[int] = 200,
        seed: int = 0,
        progress: bool = False,
    ) -> ValidationReport:
        """Check image integrity and (a sample of) annotation files.

        ``annotation_sample=None`` checks every annotation file (slow on 44k images).
        """
        iterator: Iterable[ImageRecord] = self.records
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(self.records, desc="images")
        sizes: Counter = Counter()
        for rec in iterator:
            path = self.root / rec.image
            if image_check != "none":
                ok, msg = (image_full_ok if image_check == "full" else image_header_ok)(path)
                if not ok:
                    report.corrupt_images[rec.name] = msg
                    continue
            try:
                with Image.open(path) as img:
                    rec.width, rec.height = img.size
                sizes[f"{rec.width}x{rec.height}"] += 1
            except Exception as exc:  # noqa: BLE001
                report.corrupt_images[rec.name] = f"{type(exc).__name__}: {exc}"
        report.size_counts = dict(sizes.most_common(10))

        annotated = [r for r in self.records if (r.parsing or r.densepose) and r.name not in report.corrupt_images]
        if annotation_sample is not None and len(annotated) > annotation_sample:
            rng = np.random.default_rng(seed)
            annotated = [annotated[i] for i in sorted(rng.choice(len(annotated), annotation_sample, replace=False))]
        dfmm = get_label_set("dfmm")
        dp = get_label_set("densepose")
        for rec in annotated:
            report.checked_annotations += 1
            if rec.parsing:
                msg = self._check_map(rec, rec.parsing, None, dfmm.num_classes)
                if msg:
                    report.bad_parsing[rec.name] = msg
            if rec.densepose:
                msg = self._check_map(rec, rec.densepose, self.layout.densepose_part_channel, dp.num_classes)
                if msg:
                    report.bad_densepose[rec.name] = msg
        return report

    def _check_map(self, rec: ImageRecord, rel: str, channel: Optional[int], num_classes: int) -> str:
        try:
            lab = load_label_map(self.root / rel, channel=channel)
        except ImageLoadError as exc:
            return str(exc)
        if rec.width and (lab.shape[1], lab.shape[0]) != (rec.width, rec.height):
            return f"size {lab.shape[1]}x{lab.shape[0]} != image {rec.width}x{rec.height}"
        if lab.max() >= num_classes:
            return f"label value {int(lab.max())} >= {num_classes}"
        return ""

    # ---- filtering / statistics --------------------------------------------------------

    def filter(self, require_parsing: bool = False, require_densepose: bool = False, exclude: Iterable[str] = ()) -> List[ImageRecord]:
        exclude = set(exclude)
        return [
            r
            for r in self.records
            if r.name not in exclude
            and (r.parsing or not require_parsing)
            and (r.densepose or not require_densepose)
        ]

    @staticmethod
    def statistics(records: List[ImageRecord]) -> Dict:
        groups: Dict[str, List[ImageRecord]] = defaultdict(list)
        for r in records:
            groups[r.group].append(r)
        per_item = Counter(len(v) for v in groups.values())
        return {
            "images": len(records),
            "items": len(groups),
            "items_with_2plus_images": sum(1 for v in groups.values() if len(v) >= 2),
            "images_per_item_histogram": dict(sorted(per_item.items())),
            "gender": dict(Counter(r.gender for r in records)),
            "category": dict(Counter(r.category for r in records).most_common()),
            "view": dict(Counter(r.view for r in records).most_common()),
            "with_parsing": sum(r.parsing is not None for r in records),
            "with_densepose": sum(r.densepose is not None for r in records),
        }

    # ---- persistence -------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"root": str(self.root), "layout": asdict(self.layout), "records": [asdict(r) for r in self.records]}, fh)
        return path

    @classmethod
    def load(cls, path: Path, root: Optional[Path] = None) -> "DFMMIndex":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        layout = DataLayoutConfig(**data["layout"])
        records = [ImageRecord(**r) for r in data["records"]]
        return cls(Path(root or data["root"]), layout, records)


def build_same_item_pairs(records: List[ImageRecord], max_pairs_per_item: Optional[int] = None, seed: int = 0) -> List[Tuple[str, str]]:
    """Ordered (source, target) pairs of different images of the same item.

    These are the "same garment, different pose" pairs needed for cross-view warp supervision
    and for texture-stage training (exemplar = source, ground truth = target). Pairs never cross
    item boundaries, so building them per split keeps splits leakage-free.
    """
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        if r.item_id:
            groups[r.item_id].append(r.name)
    pairs: List[Tuple[str, str]] = []
    for item in sorted(groups):
        names = sorted(groups[item])
        item_pairs = [(a, b) for a in names for b in names if a != b]
        if max_pairs_per_item is not None and len(item_pairs) > max_pairs_per_item:
            idx = rng.choice(len(item_pairs), max_pairs_per_item, replace=False)
            item_pairs = [item_pairs[i] for i in sorted(idx)]
        pairs.extend(item_pairs)
    return pairs
