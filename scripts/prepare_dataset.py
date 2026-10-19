"""Index, validate and split the dataset; print statistics.

Examples::

    python scripts/prepare_dataset.py --config configs/colab_smoke_test.yaml
    python scripts/prepare_dataset.py --config configs/colab_baseline.yaml --inspect
    python scripts/prepare_dataset.py --config configs/colab_baseline.yaml --annotation-sample 0  # check all
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.config import load_config
from src.data.dataset import prepare_dataset
from src.data.deepfashion_mm import inspect_densepose_file, inspect_root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="e.g. data.split_seed=1")
    ap.add_argument("--inspect", action="store_true", help="only describe the folder structure")
    ap.add_argument("--rebuild-splits", action="store_true")
    ap.add_argument("--annotation-sample", type=int, default=200, help="annotation files to check (0 = all)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    root = Path(cfg.data.root)
    if args.inspect:
        print(json.dumps(inspect_root(root), indent=1))
        dp_dir = root / cfg.data.layout.densepose_dir
        sample = next(dp_dir.glob(f"*{cfg.data.layout.densepose_suffix}"), None) if dp_dir.is_dir() else None
        if sample:
            print("DensePose sample", sample.name, json.dumps(inspect_densepose_file(sample)))
            print(f"-> data.layout.densepose_part_channel={cfg.data.layout.densepose_part_channel}; the part-index "
                  "channel should have max <= 24.")
        return
    prepared = prepare_dataset(cfg, rebuild_splits=args.rebuild_splits,
                               annotation_sample=None if args.annotation_sample == 0 else args.annotation_sample,
                               progress=True)
    print(("SYNTHETIC FIXTURE - not real data\n" if prepared.synthetic else "") + prepared.report.summary())
    for split, names in prepared.splits.items():
        s = prepared.stats[split]
        print(f"{split:5s}: {len(names):6d} images, {s['items']:5d} items, {s['items_with_2plus_images']} items with >=2 images")
    print("usable:", json.dumps(prepared.stats["usable"], indent=1)[:2000])
    print("split file:", cfg.data.split_file)


if __name__ == "__main__":
    main()
