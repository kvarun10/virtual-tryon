"""Measure outfit consistency of all same-item photo pairs and write the pair table used by data.pair_filter.

    python scripts/analyze_pairs.py --config configs/colab_baseline.yaml

Outputs: the pair table (data.pair_table, CSV), plus summary JSON and a ΔE histogram in
outputs/<experiment>/pair_quality/. Nothing is filtered unless data.pair_filter is changed afterwards.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.dataset import prepare_dataset  # noqa: E402
from src.data.pair_quality import build_pair_table, summarise, write_pair_table  # noqa: E402
from src.utils.paths import ProjectPaths  # noqa: E402


def run(cfg, progress=True):
    prepared = prepare_dataset(cfg)
    records = [r for s in ("train", "val", "test") for r in prepared.records(s)]
    rows = build_pair_table(records, prepared.index.root, cfg.labels.transfer_classes, progress=progress)
    if not rows:
        raise SystemExit("No same-item pairs found")
    table = write_pair_table(rows, Path(cfg.data.pair_table))
    out = ProjectPaths.from_config(cfg).experiment_dir("outputs", cfg.experiment.name) / "pair_quality"
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise(rows)
    (out / "pair_quality_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    de = np.array([r["max_delta_e"] for r in rows if not np.isnan(r["max_delta_e"])])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(np.clip(de, 0, 60), bins=60, color="#1f77b4")
    for t in (10, 15, 20):
        ax.axvline(t, color="k", ls=":", lw=1)
    ax.set_xlabel("max colour difference of shared garments (CIE76 ΔE, clipped at 60)"); ax.set_ylabel("pairs")
    ax.set_title(f"Same-item pairs: {summary['share_same_classes']:.0%} have identical garment classes")
    fig.tight_layout(); fig.savefig(out / "pair_delta_e_hist.png", dpi=130); plt.close(fig)
    return table, out, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    table, out, summary = run(load_config(args.config, args.override))
    print(json.dumps(summary, indent=1))
    print("pair table:", table, "| summary + histogram:", out)


if __name__ == "__main__":
    main()
