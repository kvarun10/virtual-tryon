"""Recompute every reported number from the saved per-pair scores (report/data/results/*.json). No GPU, no data needed.

    python scripts/show_results.py            # all tables
    python scripts/show_results.py --section benchmark

Each JSON holds, per test pair, the scores computed when the model was run on Modal (whole 176 x 256 image): ``ssim``
(our implementation), ``ssim_common`` (the PATN/GFLA/NTED/PIDM metric code), ``l1``, ``lpips``, plus FID/KID of the
whole set. This script only re-aggregates them (means, paired bootstrap 95% CIs), so the tables in docs/modal.md can be
checked live.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.final_eval import paired_bootstrap_ci  # noqa: E402

R = ROOT / "report" / "data" / "results"
PUBLISHED = [  # PIDM (CVPR 2023) Table 1, DeepFashion 256 x 176, all methods re-scored with one protocol
    ("PATN (2019)", 20.751, 0.6709, 0.2562), ("ADGAN (2020)", 14.458, 0.6721, 0.2283),
    ("PISE (2021)", 13.610, 0.6629, 0.2059), ("GFLA (2020)", 10.573, 0.7074, 0.2341),
    ("DPTN (2022)", 11.387, 0.7112, 0.1931), ("CASD (2022)", 11.373, 0.7248, 0.1936),
    ("NTED (2022)", 8.6838, 0.7182, 0.1752), ("PIDM (2023, diffusion)", 6.3671, 0.7312, 0.1678)]


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def mean(r, k):
    v = r["per_pair"].get(k)
    return float(np.mean(v)) if v else float("nan")


def fid(r):
    fk = r.get("fid_kid_raw_vs_real_targets") or {}
    return fk.get("fid", float("nan"))


def paired(a, b, metric):
    ia = {tuple(p): i for i, p in enumerate(a["per_pair"]["pairs"])}
    ib = {tuple(p): i for i, p in enumerate(b["per_pair"]["pairs"])}
    common = [p for p in ia if p in ib]
    x = np.array([a["per_pair"][metric][ia[p]] for p in common])
    y = np.array([b["per_pair"][metric][ib[p]] for p in common])
    return len(common), x.mean(), y.mean(), paired_bootstrap_ci(x, y)


def benchmark():
    print("\n=== Standard DeepFashion pose-transfer benchmark (whole 176 x 256 image, fully automatic) ===")
    print(f"{'model':<34}{'pairs':>7}{'SSIM':>8}{'SSIM*':>8}{'LPIPS':>8}{'L1':>8}{'FID':>8}")
    for label, f in (("B0 baseline (128 px)", "benchmark_b0_128.json"), ("full_256 (256 px, all data)", "benchmark_full_256.json"),
                     ("v2 (+ texture fine-tune)", "benchmark_v2.json"), ("W1 FINAL (+ richer warp)", "benchmark_w1_final.json")):
        r = load(f)
        print(f"{label:<34}{len(r['per_pair']['pairs']):>7}{mean(r, 'ssim_raw'):>8.4f}{mean(r, 'ssim_common_raw'):>8.4f}"
              f"{mean(r, 'lpips_raw'):>8.4f}{mean(r, 'l1_raw'):>8.4f}{fid(r):>8.2f}")
    print("SSIM* = common protocol code (skimage, Gaussian sigma 1.5, data range of the generated image); nan = not computed")
    print("B0 could only be scored on 1,570 pairs: it was trained on the products of the other pairs.")
    print("\nPaired comparisons on identical pairs (difference, 95% bootstrap CI):")
    for (la, fa), (lb, fb) in ((("B0", "benchmark_b0_128.json"), ("full_256", "benchmark_full_256.json")),
                                (("full_256", "benchmark_full_256.json"), ("v2", "benchmark_v2.json")),
                                (("v2", "benchmark_v2.json"), ("W1", "benchmark_w1_final.json"))):
        a, b = load(fa), load(fb)
        for m in ("ssim_raw", "lpips_raw"):
            n, x, y, ci = paired(a, b, m)
            print(f"  {la:>8} -> {lb:<8} {m[:-4]:<6} {x:.4f} -> {y:.4f}  diff {ci['mean_diff']:+.4f} "
                  f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  (n={n})")


def published():
    w = load("benchmark_w1_final.json")
    rows = PUBLISHED + [("OURS: W1 final", fid(w), mean(w, "ssim_common_raw"), mean(w, "lpips_raw"))]
    print("\n=== Placement among methods re-scored on one protocol (PIDM, CVPR 2023, Table 1) ===")
    print(f"{'method':<26}{'FID (lower)':>12}{'SSIM (higher)':>15}{'LPIPS (lower)':>15}")
    for m, f, s, l in sorted(rows, key=lambda r: -r[1]):
        print(f"{m:<26}{f:>12.2f}{s:>15.4f}{l:>15.4f}")
    print("Dress-up paper (unpublished protocol): FID 8.60, SSIM 0.95, LPIPS 0.228 (marked 'higher is better').")
    print("Caveats: ours on 7,954 of 8,570 pairs (locked test items and pairs without DensePose maps excluded);")
    print("our model never saw a standard test item; FID sample definitions may differ slightly.")


def selection():
    print("\n=== Model selection on VALIDATION try-ons (318 pose-change pairs of validation products) ===")
    base = load("valtryon_full256.json")
    for label, f in (("DensePose-only warp", "valtryon_densepose_warp.json"), ("v2 texture fine-tune", "valtryon_v2.json"),
                     ("W1 warp + v2 texture", "valtryon_w1.json")):
        b = load(f)
        out = []
        for m in ("ssim_raw", "lpips_raw"):
            n, x, y, ci = paired(base, b, m)
            out.append(f"{m[:-4]} {ci['mean_diff']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
        print(f"  {label:<24} vs full_256: " + " | ".join(out))


def diagnostics():
    print("\n=== Where the points are lost (validation try-ons, v2 model) ===")
    for label, f in (("fully automatic", "valtryon_v2.json"), ("manual labels instead of the parser", "valtryon_oracle_labels.json"),
                     ("+ true layout instead of the warp", "valtryon_oracle_layout.json")):
        r = load(f)
        print(f"  {label:<38} SSIM {mean(r, 'ssim_raw'):.4f}  LPIPS {mean(r, 'lpips_raw'):.4f}")
    print("  -> a perfect warp adds ~0.038 SSIM, a perfect parser ~0.005: the warp is the bottleneck.")


SECTIONS = {"benchmark": benchmark, "published": published, "selection": selection, "diagnostics": diagnostics}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=sorted(SECTIONS), action="append")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    for name in a.section or ["benchmark", "published", "selection", "diagnostics"]:
        SECTIONS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
