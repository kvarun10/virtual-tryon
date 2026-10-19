"""Get DeepFashion-MultiModal archives into Google Drive once, then unpack them on Colab's local disk.

Storage plan (see docs/data.md)
--------------------------------
* **Google Drive (persistent, the only permanent copy):**
  ``<drive_root>/datasets/DeepFashion-MultiModal/archives/`` holds the archives exactly as
  published (zip files). They are never extracted inside Drive: tens of thousands of small files
  on the Drive mount are slow to list and read, and use more quota.
* **Colab local disk (temporary, rebuilt each session):** ``data.root`` (default
  ``/content/data/DeepFashion-MultiModal``). Extracting straight from the Drive archives takes a few
  minutes (not measured yet). Training then reads from the local disk.
* **Your PC:** nothing. Local development uses the synthetic fixture.

Archives are downloaded only when ``allow_download=True`` is passed. The download goes from the
authors' Google Drive straight into your Drive folder, so no intermediate copy is kept anywhere.
If you prefer, use "Make a copy" in the Drive web UI and put the file in the ``archives`` folder
yourself; :func:`find_archive` will detect it.

Facts from the DFMM README: file IDs, sizes and file counts. The archive *format* (zip) and the folder
names inside are not documented. :func:`stage_archive` therefore finds the folder that contains the
expected files and moves it to the canonical name used by ``DataLayoutConfig``.
"""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.models.weights import sha256


@dataclass(frozen=True)
class ArchiveSpec:
    part: str
    gdrive_id: str
    readme_size: str  # as stated in the DFMM README
    approx_bytes: int  # used for free-space checks
    n_files: int  # README file count
    pattern: str  # glob the extracted files must match
    target_dir: str  # canonical folder under data.root (matches DataLayoutConfig defaults)


DFMM_ARCHIVES: Dict[str, ArchiveSpec] = {
    "images": ArchiveSpec("images", "1U2PljA7NE57jcSSzPs21ZurdIPXdYZtN", "~5.4 GB", int(5.4e9), 44096, "*.jpg", "images"),
    "parsing": ArchiveSpec("parsing", "1r-5t-VgDaAQidZLVgWtguaG7DvMoyUv9", "~90 MB", int(90e6), 12701, "*_segm.png", "segm"),
    "keypoints": ArchiveSpec("keypoints", "1ZXdOQI-d4zNhqRJdUEWSQvPwAtLdjovo", "956 KB", int(1e6), 2, "keypoints_*.txt", "keypoints"),
    "densepose": ArchiveSpec("densepose", "14uyqBUDDcL1VLaXm7qmqghdcbkFuQa1s", "~5.6 GB", int(5.6e9), 44096, "*_densepose.png", "densepose"),
}
PRESETS = {"images_parsing": ["images", "parsing", "keypoints"], "all": ["images", "parsing", "keypoints", "densepose"]}
STAGED_MARKER = ".staged_{part}.json"


class FetchError(RuntimeError):
    pass


def resolve_parts(parts) -> List[str]:
    if isinstance(parts, str):
        parts = PRESETS.get(parts, [parts])
    unknown = [p for p in parts if p not in DFMM_ARCHIVES]
    if unknown:
        raise FetchError(f"Unknown archive part(s) {unknown}. Known: {sorted(DFMM_ARCHIVES)} or presets {sorted(PRESETS)}")
    return list(parts)


def storage_plan(parts, archive_dir: Path, data_root: Path) -> Dict:
    """What will be stored where. Nothing is touched."""
    parts = resolve_parts(parts)
    total = sum(DFMM_ARCHIVES[p].approx_bytes for p in parts)
    return {
        "parts": {p: {"readme_size": DFMM_ARCHIVES[p].readme_size, "files": DFMM_ARCHIVES[p].n_files} for p in parts},
        "drive_archives": str(archive_dir),
        "drive_bytes_approx": total,
        "colab_local_extracted": str(data_root),
        "colab_local_bytes_approx": total,
        "local_pc_bytes": 0,
    }


def find_archive(part: str, archive_dir: Path) -> Optional[Path]:
    """Return the archive for ``part`` if present (any file whose name starts with the part name)."""
    archive_dir = Path(archive_dir)
    if not archive_dir.is_dir():
        return None
    alias = {"images": ("images", "image"), "parsing": ("parsing", "segm"), "keypoints": ("keypoints",), "densepose": ("densepose",)}[part]
    hits = sorted(p for p in archive_dir.iterdir()
                  if p.is_file() and p.name.lower().startswith(alias) and not p.name.endswith((".part", ".json")))
    return hits[0] if hits else None


def download_archive(part: str, archive_dir: Path, allow_download: bool = False) -> Path:
    """Download one archive into ``archive_dir`` (Drive) unless it is already there."""
    existing = find_archive(part, archive_dir)
    if existing:
        return existing
    spec = DFMM_ARCHIVES[part]
    if not allow_download:
        raise FetchError(
            f"Archive '{part}' ({spec.readme_size}) is not in {archive_dir}. Either pass allow_download=True "
            f"or copy it there from https://drive.google.com/file/d/{spec.gdrive_id} ('Make a copy' in the Drive UI). "
            "Licence: non-commercial research only, no redistribution."
        )
    import gdown

    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    tmp = archive_dir / f"{part}.download.part"
    out = gdown.download(id=spec.gdrive_id, output=str(tmp), quiet=False)
    if out is None or not tmp.is_file() or tmp.stat().st_size < min(spec.approx_bytes // 2, 100_000):
        size = tmp.stat().st_size if tmp.is_file() else 0
        tmp.unlink(missing_ok=True)
        raise FetchError(
            f"Download of '{part}' failed or was cut short ({size} bytes; README size {spec.readme_size}). "
            "Google Drive often limits popular files ('too many users'). Retry later, or use 'Make a copy' in the Drive UI."
        )
    final = archive_dir / (f"{part}.zip" if zipfile.is_zipfile(tmp) else f"{part}.bin")
    tmp.rename(final)
    meta = {**asdict(spec), "file": final.name, "bytes": final.stat().st_size, "sha256": sha256(final),
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (archive_dir / f"{part}.provenance.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return final


def _best_dir(root: Path, pattern: str) -> Optional[Path]:
    """Directory under ``root`` containing the most files that match ``pattern``."""
    best, best_n = None, 0
    for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        n = sum(1 for _ in d.glob(pattern))
        if n > best_n:
            best, best_n = d, n
    return best


def stage_archive(part: str, archive: Path, data_root: Path, min_free_factor: float = 1.2) -> Dict:
    """Extract ``archive`` into ``data_root/<canonical dir>`` (skipped if already staged from the same file)."""
    spec = DFMM_ARCHIVES[part]
    archive, data_root = Path(archive), Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    marker = data_root / STAGED_MARKER.format(part=part)
    stamp = {"archive": archive.name, "bytes": archive.stat().st_size}
    target = data_root / spec.target_dir
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("stamp") == stamp and target.is_dir():
        return json.loads(marker.read_text(encoding="utf-8"))
    if not zipfile.is_zipfile(archive):
        raise FetchError(f"{archive} is not a zip archive; inspect it manually (format not documented by DFMM).")
    free = shutil.disk_usage(data_root).free
    if free < min_free_factor * spec.approx_bytes:
        raise FetchError(f"Only {free / 1e9:.1f} GB free at {data_root}; '{part}' needs about {spec.readme_size}.")
    tmp = data_root / f".extract_{part}"
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.time()
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(tmp)
    src = _best_dir(tmp, spec.pattern)
    if src is None:
        shutil.rmtree(tmp, ignore_errors=True)
        raise FetchError(f"No files matching '{spec.pattern}' inside {archive.name}; archive layout differs from expectations.")
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(src), str(target))
    shutil.rmtree(tmp, ignore_errors=True)
    n = sum(1 for _ in target.glob(spec.pattern))
    info = {"part": part, "stamp": stamp, "target": str(target), "files": n, "expected_files": spec.n_files,
            "seconds": round(time.time() - t0, 1), "source_folder_in_archive": str(src.relative_to(tmp)) if src != tmp else "."}
    if n != spec.n_files:
        info["warning"] = f"found {n} files, README says {spec.n_files}"
    marker.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def fetch_and_stage(cfg, parts="images_parsing", allow_download: bool = False, progress=print) -> List[Dict]:
    """Ensure each archive is in Drive (downloading only if allowed), then extract it to ``data.root``."""
    results = []
    for part in resolve_parts(parts):
        archive = download_archive(part, Path(cfg.data.archive_dir), allow_download)
        progress(f"[{part}] archive: {archive}")
        info = stage_archive(part, archive, Path(cfg.data.root))
        progress(f"[{part}] staged {info['files']} files -> {info['target']}" + (f"  WARNING: {info['warning']}" if "warning" in info else ""))
        results.append(info)
    return results
