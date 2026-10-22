"""Try on your own photos with the trained models (works on CPU or GPU).

    python scripts/tryon.py --reference test_photos/1.jpeg --target test_photos/2.jpeg

``--reference``: a photo of a PERSON WEARING the garment (person-worn mode, as in the paper; product photos are not
supported). ``--target``: the person who should wear it. Either may be a single image or a folder; every
reference x target combination is run. Results go to ``--out/<reference>__on__<target>/`` (every stage as PNG plus
``overview.png``) and one ``summary.png`` with reference | target | result per row.

Weights:
* trained generators: ``--warp`` / ``--texture`` (default: ``checkpoints/<experiment>/<run>/last.pt`` under
  ``paths.drive_root``). The small ``generator_only.pt`` files written by notebook Section 15 work too.
* SCHP parsers: ``<paths.pretrained_dir>/schp/`` (already downloaded; add ``--allow-download`` only if missing).

Works best for: full-body, front-facing, one person, plain background, similar to DeepFashion studio photos.
Output size is the model resolution (128 x 128, letterboxed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import _bootstrap  # noqa: F401  (run as a script)
except ImportError:  # imported as scripts.tryon
    pass

import numpy as np
from PIL import Image

from src.config import load_config
from src.data.preprocessing import load_image
from src.inference.pipeline import PreTreatment, try_on
from src.inference.tryon import load_tryon_models

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _images(p: Path):
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if not files:
            raise SystemExit(f"No images in {p}")
        return files
    if not p.is_file():
        raise SystemExit(f"Not found: {p}")
    return [p]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, type=Path, help="image or folder: person wearing the garment")
    ap.add_argument("--target", required=True, type=Path, help="image or folder: person to dress")
    ap.add_argument("--out", type=Path, default=Path("outputs/my_tryon"))
    ap.add_argument("--config", default="configs/colab_pairfilter.yaml")
    ap.add_argument("--drive-root", default="outputs/local_drive",
                    help="folder with pretrained/ and checkpoints/ (Colab: /content/drive/MyDrive/VirtualTryOn)")
    ap.add_argument("--warp", default="last", help="last | best | path to a warp checkpoint")
    ap.add_argument("--texture", default="last", help="last | best | path to a texture checkpoint")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--allow-download", action="store_true", help="allow downloading missing SCHP weights")
    ap.add_argument("--dataset-keys", action="store_true",
                    help="file names are dataset image names (needed by annotation backends; tests only)")
    ap.add_argument("--override", action="append", default=[], help="extra config override key=value")
    a = ap.parse_args(argv)

    overrides = [f"paths.drive_root={Path(a.drive_root).as_posix()}", f"experiment.device={a.device}",
                 f"segmentation.device={a.device}", *a.override]
    if a.allow_download:
        overrides.append("segmentation.allow_weight_download=true")
    cfg = load_config(a.config, overrides)
    refs, tgts = _images(a.reference), _images(a.target)
    print(f"loading models (warp={a.warp}, texture={a.texture}) ...", flush=True)
    models = load_tryon_models(cfg, warp=a.warp, texture=a.texture)
    print("  ", models.info, "| device", models.device, flush=True)
    pre = PreTreatment(cfg)
    rows = []
    for r in refs:
        for t in tgts:
            if r.resolve() == t.resolve():
                continue
            out = a.out / f"{r.stem}__on__{t.stem}"
            keys = dict(reference_key=r.stem, target_key=t.stem) if a.dataset_keys else {}
            res = try_on(load_image(r), load_image(t), cfg, pretreatment=pre, models=models, save_dir=out, **keys)
            for note in res.notes:
                print(f"   note: {note}")
            e = res.extra
            rows.append(np.concatenate([e["reference_model_size"], e["target_model_size"], res.output], 1))
            print(f"{r.name} -> {t.name}: {out / 'overview.png'}", flush=True)
    if not rows:
        raise SystemExit("Nothing to do (reference and target are the same image).")
    a.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(a.out / "summary.png")
    print(f"summary (reference | target | result): {a.out / 'summary.png'}")
    print("overview columns: reference | C_seg | target | B_seg | C'_seg | layout | warped I_S | generated | result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
