"""Turn a training JSONL log into CSV files for the report (no copying numbers by hand).

    python scripts/export_run_logs.py --log /content/drive/MyDrive/VirtualTryOn/logs/baseline_128/warp_train.jsonl \
        --out report/data --name warp_baseline_128

Writes ``<name>_train.csv`` (one row per logged step) and ``<name>_val.csv`` (validation / evaluation
rows, per-class IoU flattened into columns). In Colab, write to ``<drive>/report_exports`` and download it.
"""

import argparse
import csv
import json
from pathlib import Path


def flatten(rec):
    out = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out


def write(rows, path):
    if not rows:
        return None
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="report/data")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    recs = [json.loads(l) for l in Path(args.log).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = [flatten(r) for r in recs if r.get("kind") == "train"]
    val = [flatten(r) for r in recs if r.get("kind", "").startswith(("val", "eval"))]
    for rows, suffix in ((train, "train"), (val, "val")):
        p = write(rows, out / f"{args.name}_{suffix}.csv")
        print(f"{suffix}: {len(rows)} rows -> {p}")


if __name__ == "__main__":
    main()
