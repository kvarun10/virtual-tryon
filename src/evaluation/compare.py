"""Compare an experiment's final-eval results with the baseline B0 on the SAME validation pairs (docs/experiments.md).

    python -m src.evaluation.compare --baseline <B0 results.json> --run <experiment results.json> [--out comparison.md]

Paired metrics (SSIM, L1, LPIPS of every variant) are matched pair by pair (reference, target) and reported as the mean
difference with a 95 % paired-bootstrap interval. Warp mIoU / per-class IoU / swap metrics and FID/KID are reported as
plain differences (no per-pair data). Positive Δ means the experiment is higher; for L1, LPIPS, FID and KID lower is better.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.evaluation.final_eval import paired_bootstrap_ci

LOWER_IS_BETTER = {"l1", "lpips", "fid", "kid"}
VARIANTS = [("gt_layout", "T4 texture with true layout"), ("e2e_raw", "T5 end-to-end, manual Stage A (raw)"),
            ("auto", "T6 fully automatic (+paste)"), ("copy", "T7 copy (sanity: must not change)")]


def _verdict(metric: str, ci: Dict) -> str:
    better_sign = -1 if metric in LOWER_IS_BETTER else 1
    if ci["ci_low"] > 0 or ci["ci_high"] < 0:
        return "better" if np.sign(ci["mean_diff"]) == better_sign else "worse"
    return "no clear change"


def compare_results(base: Dict, run: Dict) -> Dict:
    if base.get("split") != run.get("split"):
        raise ValueError(f"different splits: {base.get('split')} vs {run.get('split')}")
    out: Dict = {"split": run["split"], "baseline_started": base.get("started"), "run_started": run.get("started"),
                 "run_checkpoints": run.get("checkpoints"), "paired": [], "warp": {}, "realism": {}}
    if "paired" in base and "paired" in run and "per_pair" in base["paired"] and "per_pair" in run["paired"]:
        bp, rp = base["paired"]["per_pair"], run["paired"]["per_pair"]
        bidx = {tuple(p): i for i, p in enumerate(bp["pairs"])}
        common = [(bidx[tuple(p)], j) for j, p in enumerate(rp["pairs"]) if tuple(p) in bidx]
        out["paired_common_pairs"] = len(common)
        if len(common) >= 2:
            bi, ri = [c[0] for c in common], [c[1] for c in common]
            for v, label in VARIANTS:
                for m in ("ssim", "l1", "lpips"):
                    if m in bp.get(v, {}) and m in rp.get(v, {}):
                        a = np.asarray(bp[v][m])[bi]
                        b = np.asarray(rp[v][m])[ri]
                        ci = paired_bootstrap_ci(a, b)
                        out["paired"].append({"variant": v, "label": label, "metric": m, "baseline": float(a.mean()),
                                              "run": float(b.mean()), **ci, "verdict": _verdict(m, ci)})
    if "warp" in base and "warp" in run:
        bw, rw = base["warp"], run["warp"]
        out["warp"] = {k: {"baseline": bw[k], "run": rw[k], "delta": rw[k] - bw[k]}
                       for k in ("mean_iou_7", "class_area_agreement", "class_fidelity") if k in bw and k in rw}
        out["warp"]["per_class"] = {c: {"baseline": bw["per_class_iou"][c], "run": rw["per_class_iou"].get(c),
                                        "delta": (rw["per_class_iou"].get(c) or np.nan) - bw["per_class_iou"][c]}
                                    for c in bw.get("per_class_iou", {}) if bw["per_class_iou"][c] is not None}
    if "realism" in base and "realism" in run:
        out["realism"] = {k: {"baseline": base["realism"][k], "run": run["realism"][k],
                              "delta": run["realism"][k] - base["realism"][k]} for k in ("fid", "kid")}
    return out


def format_comparison(c: Dict, run_name: str = "experiment") -> str:
    L: List[str] = [f"# {run_name} vs baseline B0 ({c['split']} split)", "",
                    f"baseline run {c['baseline_started']}, experiment run {c['run_started']}; checkpoints "
                    + ", ".join(f"{k} {v.get('path')} (step {v.get('step')})" for k, v in (c.get("run_checkpoints") or {}).items()), ""]
    if c["paired"]:
        L += [f"## Paired try-on ({c['paired_common_pairs']} identical pairs; Δ = experiment − B0, 95 % bootstrap CI)", "",
              "| variant | metric | B0 | experiment | Δ | 95 % CI | verdict |", "|---|---|---|---|---|---|---|"]
        for r in c["paired"]:
            L.append(f"| {r['label']} | {r['metric']} | {r['baseline']:.4f} | {r['run']:.4f} | {r['mean_diff']:+.4f} | "
                     f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['verdict']} |")
        L += ["", "For SSIM higher is better; for L1 and LPIPS lower is better. The copy row must show no change "
              "(it does not use any trained network); if it does, the pairs differ.", ""]
    if c["warp"]:
        w = c["warp"]
        L += ["## Warp (whole split)", "", "| metric | B0 | experiment | Δ |", "|---|---|---|---|"]
        for k in ("mean_iou_7", "class_area_agreement", "class_fidelity"):
            if k in w:
                L.append(f"| {k} | {w[k]['baseline']:.3f} | {w[k]['run']:.3f} | {w[k]['delta']:+.3f} |")
        for cl, v in w.get("per_class", {}).items():
            L.append(f"| IoU {cl} | {v['baseline']:.3f} | {v['run'] if v['run'] is not None else float('nan'):.3f} | {v['delta']:+.3f} |")
        L.append("")
    if c["realism"]:
        r = c["realism"]
        L += ["## Realism (different-item try-on vs real images; lower is better)", "",
              f"FID {r['fid']['baseline']:.2f} → {r['fid']['run']:.2f} ({r['fid']['delta']:+.2f}); "
              f"KID {r['kid']['baseline']:.4f} → {r['kid']['run']:.4f} ({r['kid']['delta']:+.4f})", ""]
    return "\n".join(L)


def find_latest_results(outputs_dir: Path, experiment: str, split: str = "val", official_only: bool = True) -> Path:
    runs = sorted((Path(outputs_dir) / experiment / "final_eval").glob(f"{split}_*/results.json"))
    if official_only:
        runs = [r for r in runs if json.loads(r.read_text(encoding="utf-8")).get("official")]
    if not runs:
        raise FileNotFoundError(f"No {'official ' if official_only else ''}{split} final-eval results under "
                                f"{Path(outputs_dir) / experiment / 'final_eval'}")
    return runs[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--name", default="experiment")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    c = compare_results(json.loads(a.baseline.read_text(encoding="utf-8")), json.loads(a.run.read_text(encoding="utf-8")))
    text = format_comparison(c, a.name)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    except (AttributeError, ValueError):
        pass
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
        a.out.with_suffix(".json").write_text(json.dumps(c, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
