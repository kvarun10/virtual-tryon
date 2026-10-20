"""Compute SCHP-Pascal body maps for all annotated images once (Colab GPU), then archive them in Drive.

    python scripts/precompute_body_maps.py --config configs/colab_baseline.yaml
    python scripts/precompute_body_maps.py --config configs/colab_baseline.yaml --limit 50     # quick test

Resumable: already computed maps are skipped. Later sessions restore the maps from the Drive zip
automatically (``--stage-only``), so the network does not have to run again.
"""

import argparse
import json

import _bootstrap  # noqa: F401

from src.config import load_config
from src.data.body_maps import archive_body_maps, precompute_body_maps, stage_body_maps
from src.data.dataset import prepare_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=None, help="only the first N images of each split")
    ap.add_argument("--stage-only", action="store_true", help="only restore maps from the Drive archive")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    restored = stage_body_maps(cfg)
    print("stage:", restored)
    if args.stage_only:
        return
    prepared = prepare_dataset(cfg)
    records = [r for split in ("train", "val", "test") for r in prepared.records(split)[: args.limit]]
    print(json.dumps(precompute_body_maps(cfg, records, prepared.index.root), indent=1))
    print("archived to", archive_body_maps(cfg))


if __name__ == "__main__":
    main()
