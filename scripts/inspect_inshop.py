"""Inspect the DeepFashion In-shop release WITHOUT downloading images: what is new compared with our DFMM data.

    python scripts/inspect_inshop.py --inshop "<drive>/datasets/inshop" --split-file <drive>/cache/splits/<...>.json \
        [--dfmm-images /content/data/DeepFashion-MultiModal/images] [--out report.json]

``--inshop`` is the folder that contains ``Anno/`` and ``Eval/`` (e.g. a Drive shortcut to
"In-shop Clothes Retrieval Benchmark"). Reads only the text annotations (~10 MB) and, if present, the
segmentation JSON files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import _bootstrap  # noqa: F401  (run as a script)
except ImportError:
    pass

from src.data.inshop import inspect_inshop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inshop", required=True, type=Path)
    ap.add_argument("--split-file", required=True, type=Path)
    ap.add_argument("--dfmm-images", type=Path, help="DFMM image folder (all 44k names); optional")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    names = None
    if a.dfmm_images:
        names = [p.stem for p in a.dfmm_images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    rep = inspect_inshop(a.inshop / "Eval" / "list_eval_partition.txt", a.inshop / "Anno" / "list_bbox_inshop.txt",
                         a.split_file, names, a.inshop / "Anno" / "segmentation")
    text = json.dumps(rep, indent=1)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
