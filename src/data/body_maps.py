"""Body maps (B_seg) for training: read from dataset annotations or from precomputed network output.

Network backends (e.g. SCHP-Pascal) are never run inside a DataLoader worker. They are run once by
:func:`precompute_body_maps` (on the Colab GPU) and saved as label PNGs:

* working copy: ``segmentation.body_maps_local/<backend>/<image name>.png`` (Colab local disk)
* permanent copy: ``segmentation.body_maps_archive/<backend>.zip`` (Google Drive), written by
  :func:`archive_body_maps` and restored in later sessions by :func:`stage_body_maps`.

A ``backend.json`` next to the PNGs records which backend and checkpoint produced them;
:class:`BodyMapSource` refuses maps made by a different configuration.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from PIL import Image

from src.config import ProjectConfig
from src.data.deepfashion_mm import ImageRecord
from src.data.labels import convert_labels, get_label_set
from src.data.preprocessing import load_image, load_label_map

# native label set of every backend that can provide body maps
BODY_BACKEND_LABELS = {"annotation_densepose": "densepose", "schp_pascal": "pascal_person_part"}


class BodyMapsUnavailableError(FileNotFoundError):
    pass


class BodyMapSource:
    """Returns the B_seg label map (in ``labels.body_label_set``) for a dataset record."""

    def __init__(self, cfg: ProjectConfig, data_root: Path):
        self.backend = cfg.segmentation.body_backend
        if self.backend not in BODY_BACKEND_LABELS:
            raise ValueError(f"Body backend '{self.backend}' cannot provide training maps. Use one of {sorted(BODY_BACKEND_LABELS)}")
        self.native = BODY_BACKEND_LABELS[self.backend]
        self.target = cfg.labels.body_label_set
        self.num_classes = get_label_set(self.target).num_classes
        self.root = Path(data_root)
        self.layout = cfg.data.layout
        self.map_dir = Path(cfg.segmentation.body_maps_local) / self.backend
        if self.backend != "annotation_densepose" and not (self.map_dir / "backend.json").is_file():
            raise BodyMapsUnavailableError(
                f"No precomputed '{self.backend}' body maps in {self.map_dir}. On Colab run notebook Section 8a "
                f"(or `python scripts/precompute_body_maps.py --config <cfg>`); if they were computed in an earlier "
                f"session they are restored from {cfg.segmentation.body_maps_archive}."
            )

    def load(self, rec: ImageRecord) -> np.ndarray:
        if self.backend == "annotation_densepose":
            if not rec.densepose:
                raise BodyMapsUnavailableError(f"{rec.name}: no DensePose file")
            native = load_label_map(self.root / rec.densepose, channel=self.layout.densepose_part_channel)
        else:
            path = self.map_dir / f"{rec.name}.png"
            if not path.is_file():
                raise BodyMapsUnavailableError(f"{rec.name}: body map missing at {path}. Re-run the precompute step.")
            native = load_label_map(path)
        return convert_labels(native, self.native, self.target)


def precompute_body_maps(cfg: ProjectConfig, records: Iterable[ImageRecord], data_root: Path, backend=None,
                         progress: bool = True) -> Dict:
    """Run the configured network body backend over ``records`` (resumable: existing PNGs are skipped)."""
    from src.models.segmentation import build_backend

    name = cfg.segmentation.body_backend
    if name == "annotation_densepose":
        return {"skipped": "annotation_densepose reads dataset files; nothing to precompute"}
    backend = backend or build_backend(name, cfg, role="body", use_cache=False)
    out_dir = Path(cfg.segmentation.body_maps_local) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    desc = json.dumps(backend.describe(), sort_keys=True, default=str)
    meta = out_dir / "backend.json"
    if meta.is_file() and meta.read_text(encoding="utf-8") != desc:
        raise ValueError(f"{out_dir} holds maps from a different backend config; delete it first.\n{meta.read_text()}\nvs\n{desc}")
    meta.write_text(desc, encoding="utf-8")
    records = list(records)
    todo = [r for r in records if not (out_dir / f"{r.name}.png").is_file()]
    it = todo
    if progress:
        from tqdm.auto import tqdm

        it = tqdm(todo, desc=f"body maps ({name})")
    for rec in it:
        labels = backend.predict(load_image(Path(data_root) / rec.image)).labels
        tmp = out_dir / f"{rec.name}.tmp.png"
        Image.fromarray(labels).save(tmp)
        tmp.replace(out_dir / f"{rec.name}.png")
    return {"backend": name, "computed": len(todo), "already_present": len(records) - len(todo), "dir": str(out_dir)}


def archive_body_maps(cfg: ProjectConfig) -> Path:
    """Zip the local body maps into the permanent archive folder (Drive). PNGs are stored, not recompressed."""
    name = cfg.segmentation.body_backend
    src = Path(cfg.segmentation.body_maps_local) / name
    if not (src / "backend.json").is_file():
        raise BodyMapsUnavailableError(f"Nothing to archive in {src}")
    dst_dir = Path(cfg.segmentation.body_maps_archive)
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst_dir / f"{name}.zip.part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(src.iterdir()):
            if f.suffix in {".png", ".json"} and not f.name.endswith(".tmp.png"):
                zf.write(f, f.name)
    final = dst_dir / f"{name}.zip"
    tmp.replace(final)
    return final


def stage_body_maps(cfg: ProjectConfig) -> Optional[Dict]:
    """Restore local body maps from the Drive archive if the local copy is missing. Returns None if no archive."""
    name = cfg.segmentation.body_backend
    if name == "annotation_densepose":
        return None
    local = Path(cfg.segmentation.body_maps_local) / name
    archive = Path(cfg.segmentation.body_maps_archive) / f"{name}.zip"
    if (local / "backend.json").is_file():
        return {"status": "already local", "dir": str(local), "files": sum(1 for _ in local.glob("*.png"))}
    if not archive.is_file():
        return None
    tmp = local.parent / f".extract_{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(tmp)
    shutil.rmtree(local, ignore_errors=True)
    tmp.replace(local)
    return {"status": "restored", "dir": str(local), "files": sum(1 for _ in local.glob("*.png"))}
