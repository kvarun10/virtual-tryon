"""Run a network parser over a dataset split once and cache the label PNGs (resumable).

    python scripts/cache_segmentations.py --config configs/colab_baseline.yaml --backend schp_lip --split train --limit 500

Already-cached images are skipped, so the command can be re-run after a Colab disconnect.
"""

import argparse

import _bootstrap  # noqa: F401
from tqdm.auto import tqdm

from src.config import load_config
from src.data.dataset import prepare_dataset
from src.data.preprocessing import load_image
from src.models.segmentation import CachedBackend, build_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.override + ["segmentation.cache_outputs=true"])
    prepared = prepare_dataset(cfg)
    backend = build_backend(args.backend, cfg)
    if not isinstance(backend, CachedBackend):
        raise SystemExit(f"Backend {args.backend} reads annotations directly; nothing to cache.")
    records = prepared.records(args.split)[: args.limit]
    for rec in tqdm(records, desc=f"{args.backend}/{args.split}"):
        backend.predict(load_image(prepared.index.root / rec.image), key=rec.name)
    print(f"done: {backend.misses} computed, {backend.hits} already cached -> {backend.dir}")


if __name__ == "__main__":
    main()
