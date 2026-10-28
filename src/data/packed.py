"""Packed training store: every DeepFashion In-shop photo at 256 x 256 with its label maps, in a few large files.

Why: the DFMM loaders decode a 750 x 1101 JPEG and resize it on every access, which starves a fast GPU. The In-shop
release already holds the same photos letterboxed to 256 x 256 (verified 2026-10-27: DFMM parses resized with our
letterbox overlay the In-shop photos exactly; all 44,096 DFMM photos are among the 52,712 In-shop photos). So the
store keeps the original In-shop JPEG bytes (no recompression) and label PNGs that are already at 256 x 256.

Layout of ``<packed_dir>`` (written by :mod:`src.data.packing`)::

    meta.json              version, size, label sets, counts, sources
    names.json             [{"name", "item_id", "seq", "view", "manual"}]   (index i = row i in every blob)
    images.bin/.idx.npy    JPEG bytes / int64 offsets (N + 1)
    garment_manual.*       garment12 PNGs from DFMM manual parsing (empty entry = no manual label)
    garment_auto.*         garment12 PNGs from SCHP-LIP (every photo)
    body.*                 body maps (labels.body_label_set) from SCHP-Pascal (every photo)
    body_dp.*              optional: DensePose body maps (body15), empty entry = no DensePose for that photo
                           (written by src.data.packing.add_densepose; used when data.body_source=densepose)
    splits.json            {"splits": {"train", "val", "test"}, "meta": {...}}  (see build_packed_splits)
    pairs.csv              same-item pair table (src/data/pair_quality.py format) for data.pair_filter

``data.label_source``: ``manual`` uses only photos with manual parsing (like the DFMM baseline);
``manual_then_auto`` uses the manual map where it exists and the SCHP map otherwise (all photos).
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from PIL import Image

from src.config import ProjectConfig

PACK_VERSION = 1
BLOBS = ("images", "garment_manual", "garment_auto", "body")


class PackedStoreError(FileNotFoundError):
    pass


class BlobWriter:
    """Append-only byte records: ``<name>.bin`` + ``<name>.idx.npy`` (offsets, N + 1)."""

    def __init__(self, path_stem: Path):
        self.stem = Path(path_stem)
        self.fh = open(self.stem.with_suffix(".bin"), "wb")
        self.offsets = [0]

    def append(self, data: bytes) -> None:
        self.fh.write(data)
        self.offsets.append(self.offsets[-1] + len(data))

    def close(self) -> int:
        self.fh.close()
        np.save(self.stem.with_suffix(".idx.npy"), np.asarray(self.offsets, np.int64))
        return len(self.offsets) - 1


class BlobReader:
    """Random access to one blob. The memmap is opened lazily per process, so DataLoader workers each get their own."""

    def __init__(self, path_stem: Path):
        self.stem = Path(path_stem)
        self.offsets = np.load(self.stem.with_suffix(".idx.npy"))
        self._mm, self._pid = None, None

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def get(self, i: int) -> bytes:
        if self._mm is None or self._pid != os.getpid():
            size = int(self.offsets[-1])
            self._mm = np.memmap(self.stem.with_suffix(".bin"), np.uint8, "r", shape=(max(size, 1),)) if size else np.zeros(1, np.uint8)
            self._pid = os.getpid()
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self._mm[a:b].tobytes()

    def __getstate__(self):  # never pickle the memmap into workers
        d = dict(self.__dict__)
        d["_mm"] = d["_pid"] = None
        return d


def encode_png(labels: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(labels.astype(np.uint8))).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def decode_png(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)))


def decode_jpeg(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data))
    return np.asarray(img.convert("RGB") if img.mode != "RGB" else img)


@dataclass
class PackedRecord:
    """Duck-typed stand-in for :class:`src.data.deepfashion_mm.ImageRecord` (the fields the datasets use)."""

    name: str
    item_id: Optional[str]
    seq: Optional[str]
    view: Optional[str]
    manual: bool
    label_source: str = "manual_then_auto"

    @property
    def parsing(self) -> Optional[str]:  # truthy = a garment map is available under the configured label source
        if self.manual:
            return "manual"
        return "auto" if self.label_source == "manual_then_auto" else None

    @property
    def image(self) -> str:
        return self.name

    @property
    def group(self) -> str:
        return self.item_id if self.item_id else f"name:{self.name}"


class PackedStore:
    def __init__(self, directory: Path, label_source: str = "manual_then_auto", body_source: str = "schp"):
        self.dir = Path(directory)
        meta_path = self.dir / "meta.json"
        if not meta_path.is_file():
            raise PackedStoreError(f"No packed store at {self.dir} (meta.json missing). Run the packing job first "
                                   "(modal_app.py::pack or src.data.packing.build_pack).")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("version") != PACK_VERSION:
            raise PackedStoreError(f"{self.dir}: pack version {self.meta.get('version')} != {PACK_VERSION}; rebuild it")
        self.label_source = label_source
        rows = json.loads((self.dir / "names.json").read_text(encoding="utf-8"))
        self.records = [PackedRecord(r["name"], r.get("item_id"), r.get("seq"), r.get("view"), bool(r["manual"]), label_source)
                        for r in rows]
        self.index = {r.name: i for i, r in enumerate(self.records)}
        self.blobs = {b: BlobReader(self.dir / b) for b in BLOBS}
        for b, r in self.blobs.items():
            if len(r) != len(self.records):
                raise PackedStoreError(f"{self.dir}: blob '{b}' has {len(r)} entries for {len(self.records)} names")
        self.size = tuple(self.meta["size"])
        self.body_source = body_source
        self.dp = None
        if (self.dir / "body_dp.idx.npy").is_file():
            self.dp = BlobReader(self.dir / "body_dp")
            if len(self.dp) != len(self.records):
                raise PackedStoreError(f"{self.dir}: body_dp has {len(self.dp)} entries for {len(self.records)} names")
            self._has_dp = np.diff(self.dp.offsets) > 0
        elif body_source == "densepose":
            raise PackedStoreError(f"{self.dir}: no DensePose maps (body_dp); run src.data.packing.add_densepose first")

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def image(self, name: str) -> np.ndarray:
        return decode_jpeg(self.blobs["images"].get(self.index[name]))

    def garment(self, name: str) -> np.ndarray:
        i = self.index[name]
        if self.records[i].manual:
            return decode_png(self.blobs["garment_manual"].get(i))
        if self.label_source != "manual_then_auto":
            raise KeyError(f"{name}: no manual garment map and data.label_source=manual")
        return decode_png(self.blobs["garment_auto"].get(i))

    def garment_auto(self, name: str) -> np.ndarray:
        return decode_png(self.blobs["garment_auto"].get(self.index[name]))

    def has_dp(self, name: str) -> bool:
        return self.dp is not None and bool(self._has_dp[self.index[name]])

    def body_dp(self, name: str) -> np.ndarray:
        """DensePose body15 map, whatever ``body_source`` is (used as an extra warp input)."""
        data = self.dp.get(self.index[name]) if self.dp is not None else b""
        if not data:
            raise KeyError(f"{name}: no DensePose map")
        return decode_png(data)

    def has_body(self, name: str) -> bool:
        """False only when DensePose maps are requested and this photo has none."""
        return self.body_source != "densepose" or bool(self._has_dp[self.index[name]])

    def body(self, name: str) -> np.ndarray:
        i = self.index[name]
        if self.body_source == "densepose":
            data = self.dp.get(i)
            if not data:
                raise KeyError(f"{name}: no DensePose map")
            return decode_png(data)
        return decode_png(self.blobs["body"].get(i))


class _PackedIndex:
    def __init__(self, store: PackedStore):
        self.root = store.dir
        self._store = store

    def __getitem__(self, name: str) -> PackedRecord:
        return self._store.records[self._store.index[name]]

    def __len__(self) -> int:
        return len(self._store)


@dataclass
class PackedPrepared:
    """Same interface as :class:`src.data.dataset.PreparedData` as far as the trainers and datasets use it."""

    store: PackedStore
    splits: Dict[str, List[str]]
    split_meta: Dict
    synthetic: bool = False

    @property
    def index(self) -> _PackedIndex:
        return _PackedIndex(self.store)

    def records(self, split: str) -> List[PackedRecord]:
        recs = [self.store.records[self.store.index[n]] for n in self.splits[split]]
        return [r for r in recs if r.parsing]


def load_packed(cfg: ProjectConfig) -> PackedPrepared:
    store = PackedStore(Path(cfg.data.packed_dir), cfg.data.label_source, cfg.data.body_source)
    if tuple(cfg.data.image_size) != store.size:
        print(f"note: packed store is {store.size}, model size {tuple(cfg.data.image_size)}; maps are resized per sample")
    split_path = Path(cfg.data.split_file)
    if not split_path.is_file():
        raise PackedStoreError(f"Split file {split_path} missing (the packing job writes {store.dir / 'splits.json'})")
    data = json.loads(split_path.read_text(encoding="utf-8"))
    splits = {k: [n for n in v if n in store] for k, v in data["splits"].items()}
    if cfg.data.max_samples:
        splits = {k: v[: cfg.data.max_samples] for k, v in splits.items()}
    return PackedPrepared(store, splits, data.get("meta", {}))


def prepare_any(cfg: ProjectConfig):
    """Packed store when ``data.packed_dir`` is set, otherwise the DFMM pipeline (unchanged)."""
    if cfg.data.packed_dir:
        return load_packed(cfg)
    from src.data.dataset import prepare_dataset

    return prepare_dataset(cfg)


def item_of_name(name: str) -> str:
    return name.split("-")[2].split("_")[-1]


def build_packed_splits(all_rows: List[Dict], old_splits: Dict[str, List[str]], protected_items: Set[str],
                        protected_label: str = "standard pose-transfer test items") -> Dict:
    """New splits over all packed photos, keeping the old evaluation splits intact.

    * ``val`` / ``test``: exactly the old (manually parsed) val / test photos, so metrics stay comparable and the
      locked test split is unchanged.
    * ``train``: every photo whose ITEM is in neither old val, old test nor ``protected_items`` (the standard
      benchmark test items), whether manually parsed or not.
    Everything else (unparsed photos of val / test / protected items) is unused. Items never cross splits.
    """
    item_split: Dict[str, str] = {}
    for split in ("val", "test"):
        for n in old_splits.get(split, []):
            item_split[item_of_name(n)] = split
    train, unused = [], {"val_or_test_items": 0, "protected_items": 0}
    old_train = set(old_splits.get("train", []))
    dropped_old_train = 0
    for r in all_rows:
        it = r["item_id"] or item_of_name(r["name"])
        if it in item_split:
            unused["val_or_test_items"] += 1
        elif it in protected_items:
            unused["protected_items"] += 1
            dropped_old_train += r["name"] in old_train
        else:
            train.append(r["name"])
    names = {r["name"] for r in all_rows}
    splits = {"train": sorted(train), "val": [n for n in old_splits.get("val", []) if n in names],
              "test": [n for n in old_splits.get("test", []) if n in names]}
    tr_items = {item_of_name(n) for n in splits["train"]}
    for s in ("val", "test"):
        clash = tr_items & {item_of_name(n) for n in splits[s]}
        if clash:
            raise ValueError(f"item leakage train/{s}: {sorted(clash)[:5]}")
    if tr_items & protected_items:
        raise ValueError("protected items in train")
    meta = {"rule": f"train = all photos whose item is not in old val/test or the {protected_label}",
            "counts": {k: len(v) for k, v in splits.items()},
            "train_manual": sum(1 for r in all_rows if r["manual"] and r["name"] in set(splits["train"])),
            "unused_photos": unused, "old_train_photos_dropped_as_protected": dropped_old_train,
            "protected_items": len(protected_items)}
    return {"splits": splits, "meta": meta}

