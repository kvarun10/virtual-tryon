"""Leakage check and paper-style benchmark on the standard DeepFashion pose-transfer test pairs.

    # 1) leakage check only (CPU, seconds; needs our split file and the DFMM image folder):
    python scripts/pose_transfer_benchmark.py --leakage-only --config configs/colab_pairfilter.yaml \
        --drive-root /content/drive/MyDrive/VirtualTryOn --image-dir /content/data/DeepFashion-MultiModal/images

    # 2) benchmark on the scorable pairs (GPU recommended):
    python scripts/pose_transfer_benchmark.py --config configs/colab_pairfilter.yaml --drive-root ... --image-dir ...

Pair files are read from ``<drive>/datasets/pose_transfer_pairs`` (``--allow-pair-download`` fetches exactly the two
CSVs, SHA-256 checked). LPIPS/FID weights come from ``<drive>/pretrained/torch`` (already downloaded in Phase 7).
Results: ``<drive>/outputs/<experiment>/pose_transfer/<time>/`` (leakage.json, results.json, results.md, grid.png).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

try:
    import _bootstrap  # noqa: F401  (run as a script)
except ImportError:
    pass

from src.config import load_config
from src.data.pose_transfer import categorise_pairs, ensure_pair_files, leakage_report, load_split_names, read_pairs
from src.utils.paths import ProjectPaths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/colab_pairfilter.yaml")
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--image-dir", type=Path, help="folder with the DFMM .jpg photos")
    ap.add_argument("--packed-dir", type=Path, help="read the photos from the packed In-shop store instead (Modal runs)")
    ap.add_argument("--pairs-dir", type=Path, help="default <drive>/datasets/pose_transfer_pairs")
    ap.add_argument("--allow-pair-download", action="store_true")
    ap.add_argument("--leakage-only", action="store_true")
    ap.add_argument("--limit", type=int, help="score only the first N scorable pairs (quick check, not reported)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--override", action="append", default=[])
    a = ap.parse_args(argv)

    cfg = load_config(a.config, [f"paths.drive_root={a.drive_root}", *a.override])
    paths = ProjectPaths.from_config(cfg)
    files = ensure_pair_files(a.pairs_dir or paths.datasets / "pose_transfer_pairs", a.allow_pair_download)
    test_pairs, train_pairs = read_pairs(files["test"]), read_pairs(files["train"])
    if a.packed_dir:
        from src.data.packed import PackedStore
        from src.evaluation.pose_transfer_eval import inshop_loader

        store = PackedStore(a.packed_dir)
        available, load = set(store.index), inshop_loader(store)
    elif a.image_dir:
        available, load = {p.stem for p in a.image_dir.iterdir() if p.suffix.lower() == ".jpg"}, None
    else:
        ap.error("give --image-dir or --packed-dir")
    splits = load_split_names(Path(cfg.data.split_file))
    leak = leakage_report(test_pairs, train_pairs, splits, available)
    out = a.out or paths.outputs / cfg.experiment.name / "pose_transfer" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "leakage.json").write_text(json.dumps(leak, indent=1), encoding="utf-8")
    print(json.dumps(leak, indent=1))
    if a.leakage_only:
        print("saved:", out / "leakage.json")
        return 0

    from src.data.pose_transfer import SOURCE  # noqa: F401
    from src.evaluation.final_eval import metric_weights_dir
    from src.evaluation.image_metrics import InceptionFeatures, LPIPSMetric
    from src.evaluation.pose_transfer_eval import evaluate_pose_transfer, format_pose_transfer
    from src.inference.tryon import load_tryon_models

    models = load_tryon_models(cfg, warp=cfg.evaluation.warp_checkpoint, texture=cfg.evaluation.texture_checkpoint)
    print("models:", models.info, flush=True)
    w, allow = metric_weights_dir(cfg), cfg.evaluation.allow_metric_download
    res = evaluate_pose_transfer(models, cfg, test_pairs, categorise_pairs(test_pairs, splits, available), a.image_dir,
                                 LPIPSMetric(models.device, w, allow), InceptionFeatures(models.device, w, allow),
                                 limit=a.limit, grid_path=out / "grid.png", load=load)
    res.update({"checkpoints": models.info, "limit": a.limit, "official": a.limit is None,
                "image_source": "packed In-shop store" if a.packed_dir else "DFMM photos", "split_file": cfg.data.split_file})
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    text = format_pose_transfer(res, leak)
    (out / "results.md").write_text(text, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    print(text)
    print("saved:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
