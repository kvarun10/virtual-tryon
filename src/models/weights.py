"""Registry of third-party pretrained weights, with explicit-consent downloading.

Nothing is downloaded unless ``allow_download=True`` is passed (config:
``segmentation.allow_weight_download``). Each downloaded file gets a ``<file>.provenance.json``
recording source URL, licence, size and SHA-256 so results can be traced to exact weights.

Sizes: the official files' exact sizes / hashes are not published. Our re-implementation has
66.72 M parameters + 0.11 M buffers (measured), i.e. ~255 MiB of fp32 tensors per SCHP file.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class WeightSpec:
    key: str
    filename: str
    gdrive_id: str
    source: str
    licence: str
    label_set: str
    approx_size: str
    notes: str = ""


WEIGHTS: Dict[str, WeightSpec] = {
    "schp_lip": WeightSpec(
        key="schp_lip",
        filename="exp-schp-201908261155-lip.pth",
        gdrive_id="1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH",
        source="https://github.com/PeikeLi/Self-Correction-Human-Parsing (README, 'LIP' model link)",
        licence="MIT (code). Weights trained on LIP (research dataset).",
        label_set="lip",
        approx_size="~255 MiB",
        notes="Reported 59.36 mIoU on LIP val (SCHP README).",
    ),
    "schp_atr": WeightSpec(
        key="schp_atr",
        filename="exp-schp-201908301523-atr.pth",
        gdrive_id="1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP",
        source="https://github.com/PeikeLi/Self-Correction-Human-Parsing (README, 'ATR' model link)",
        licence="MIT (code). Weights trained on ATR (research dataset).",
        label_set="atr",
        approx_size="~255 MiB",
        notes="Reported 82.29 mIoU on ATR test. ATR has no coat class.",
    ),
    "schp_pascal": WeightSpec(
        key="schp_pascal",
        filename="exp-schp-201908270938-pascal-person-part.pth",
        gdrive_id="1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE",
        source="https://github.com/PeikeLi/Self-Correction-Human-Parsing (README, 'Pascal-Person-Part' link)",
        licence="MIT (code). Weights trained on Pascal-Person-Part.",
        label_set="pascal_person_part",
        approx_size="~255 MiB",
        notes="Reported 71.46 mIoU on Pascal-Person-Part val. 7 coarse body parts, no left/right.",
    ),
}


class WeightsUnavailableError(FileNotFoundError):
    pass


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def weight_path(key: str, pretrained_dir: Path) -> Path:
    return Path(pretrained_dir) / "schp" / WEIGHTS[key].filename


def ensure_weights(key: str, pretrained_dir: Path, allow_download: bool = False) -> Path:
    """Return the local path of a registered weight file, downloading only with consent."""
    if key not in WEIGHTS:
        raise KeyError(f"Unknown weights '{key}'. Known: {sorted(WEIGHTS)}")
    spec = WEIGHTS[key]
    path = weight_path(key, pretrained_dir)
    if path.is_file():
        return path
    if not allow_download:
        raise WeightsUnavailableError(
            f"Pretrained weights '{key}' not found at {path}.\n"
            f"  Source : {spec.source}\n  Licence: {spec.licence}\n  Size   : {spec.approx_size}\n"
            "Either place the file there manually, or set segmentation.allow_weight_download=true "
            "(downloads from the authors' Google Drive with gdown)."
        )
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise WeightsUnavailableError("gdown is not installed: pip install gdown") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    result = gdown.download(id=spec.gdrive_id, output=str(tmp), quiet=False)
    if result is None or not tmp.is_file() or tmp.stat().st_size < 1_000_000:
        size = tmp.stat().st_size if tmp.is_file() else 0
        tmp.unlink(missing_ok=True)
        raise WeightsUnavailableError(
            f"Download of '{key}' failed or returned an HTML page ({size} bytes). Google Drive often "
            "rate-limits popular files; retry later or download manually from the link in "
            f"{spec.source} and place it at {path}."
        )
    tmp.rename(path)
    provenance = {**asdict(spec), "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "bytes": path.stat().st_size, "sha256": sha256(path)}
    path.with_name(path.name + ".provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    return path


def provenance(key: str, pretrained_dir: Path) -> Optional[Dict]:
    p = weight_path(key, pretrained_dir)
    meta = p.with_name(p.name + ".provenance.json")
    return json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else None
