"""Put DFMM archives in Google Drive (once) and extract them to Colab's local disk (each session).

    # 1. show the plan only (default, touches nothing)
    python scripts/fetch_dfmm.py --config configs/colab_baseline.yaml
    # 2. download missing archives into Drive, then extract locally
    python scripts/fetch_dfmm.py --config configs/colab_baseline.yaml --download
    # 3. later sessions: archives already in Drive -> only extract
    python scripts/fetch_dfmm.py --config configs/colab_baseline.yaml --stage

--parts: images_parsing (default: images + parsing + keypoints) | all | e.g. densepose
"""

import argparse
import json

import _bootstrap  # noqa: F401

from src.config import load_config
from src.data.fetch import fetch_and_stage, find_archive, resolve_parts, storage_plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--parts", default="images_parsing")
    ap.add_argument("--override", nargs="*", default=[])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="download missing archives into Drive, then extract")
    mode.add_argument("--stage", action="store_true", help="only extract archives already in Drive")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    plan = storage_plan(args.parts, cfg.data.archive_dir, cfg.data.root)
    for part in resolve_parts(args.parts):
        plan["parts"][part]["already_in_drive"] = str(find_archive(part, cfg.data.archive_dir) or "no")
    print(json.dumps(plan, indent=1))
    if not (args.download or args.stage):
        print("Plan only. Re-run with --download (first time) or --stage (later sessions).")
        return
    fetch_and_stage(cfg, args.parts, allow_download=args.download)


if __name__ == "__main__":
    main()
