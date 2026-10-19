"""Check that an SCHP checkpoint loads into the pure-PyTorch re-implementation, and optionally
parse one or more images.

    python scripts/verify_schp_checkpoint.py --weights schp_lip --checkpoint /path/exp-schp-201908261155-lip.pth \
        --images person1.jpg person2.jpg --out outputs/schp_check

This is the compatibility check required before using a parser: strict key/shape match, finite
outputs, and a visual inspection of the predicted labels (saved as colour PNGs with a legend).
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.data.labels import class_histogram, get_label_set
from src.data.preprocessing import load_image
from src.inference.visualization import overlay, show_panels
from src.models.segmentation import SCHPBackend
from src.models.weights import WEIGHTS, sha256
from src.utils.device import select_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, choices=sorted(WEIGHTS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--out", default="outputs/schp_check")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    print(f"checkpoint: {ckpt} ({ckpt.stat().st_size / 2**20:.1f} MiB) sha256={sha256(ckpt)[:16]}...")
    backend = SCHPBackend(args.weights, ckpt, select_device(args.device), "clothing" if args.weights != "schp_pascal" else "body")
    print("strict load OK:", backend.load_info)
    ls = get_label_set(backend.label_set)
    out = Path(args.out)
    for img_path in args.images:
        img = load_image(img_path)
        res = backend.predict(img)
        assert res.labels.shape == img.shape[:2]
        hist = class_histogram(res.labels, ls)
        print(f"{img_path}: {img.shape[1]}x{img.shape[0]} -> classes {hist}")
        show_panels([("input", img), (f"{ls.name} labels", ls.palette_array()[res.labels]),
                     ("overlay", overlay(img, res.labels, ls))], legends=[(ls, res.labels)],
                    title=f"{args.weights} on {Path(img_path).name}", save_path=out / f"{Path(img_path).stem}_{args.weights}.png")
    if torch.cuda.is_available():
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
