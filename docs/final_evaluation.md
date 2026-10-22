# Final evaluation protocol (Phase 7)

Written on 2026-10-22, **before** the test split was evaluated for the first time. The protocol is fixed
here so that no choice can be tuned on test results.

## Rules

1. **The test split (1,270 images, 678 items, item-disjoint) is used only for final reporting.** All model
   choices so far used training and validation data only. Every test evaluation is appended to
   `cache/test_split_log.jsonl` (time, commit, checkpoints, settings), so the number of test runs is auditable.
2. Models: `warp_pf` (step 10,000) and `texture_pf` (step 13,000), the current baselines, config
   `configs/colab_pairfilter.yaml`, 128 × 128 letterboxed, `last.pt`.
3. Everything is run once; no re-runs to pick the best number. If a bug is found, the fix and the re-run are
   both logged in `docs/problems_log.md`.
4. Our numbers are reported next to, never mixed with, the paper's published values, whose protocol is
   undocumented (paper_analysis.md §2.6).

## What is measured

| # | Component | Data (test split) | Metrics | Notes |
|---|---|---|---|---|
| T1 | Stage A clothing parser (SCHP-LIP) | all 1,270 images with manual parsing | pixel accuracy, mIoU (garment12, background excluded), per class | parser vs **manual** labels |
| T2 | Stage B warp | all 1,270 images (mixed self/cross-view triplets, deterministic) | mIoU over the 7 trained classes, per class | same definition as validation (metric v2) |
| T3 | Stage B warp, swap test | 64 cross-item pairs | class-area agreement, class fidelity | no ground truth exists |
| T4 | Stage C texture with the true layout | real cross-view pairs (pair filter ΔE ≤ 15) | SSIM, L1, LPIPS on the garment + skin region | upper bound for the end-to-end result |
| T5 | End-to-end, Stage A from **annotations** | same pairs | SSIM, L1, LPIPS (garment + skin) | isolates Stages B + C |
| T6 | End-to-end, Stage A from **SCHP networks** | same pairs | SSIM, L1, LPIPS (garment + skin) | the fully automatic pipeline |
| T7 | Lower bound: copy the reference photo | same pairs | SSIM, L1, LPIPS | |
| T8 | Realism of real try-on outputs | each test image as target, reference from a **different** item (fully automatic Stage A) | FID and KID vs the real test images | distribution-level; no ground truth |
| T9 | Noise floor for T8 | real validation images vs real test images | FID, KID | tells what "good" means at this sample size |
| T10 | Cost | 20 single-image runs after warm-up, T4 | ms per stage (Stage A clothing, Stage A body, warp, texture), total; peak GPU memory | `torch.cuda.synchronize` around each stage |

## Metric definitions

* **SSIM**: Gaussian window 11 × 11, σ = 1.5 (Wang et al. 2004), on [0, 1] RGB, averaged over the garment + skin
  pixels of the ground-truth layout (face and hair excluded because the pipeline pastes the target's own face/hair).
* **L1**: mean absolute error in [0, 1] over the same region.
* **LPIPS** (Zhang et al. 2018, AlexNet, v0.1; lower is better, a *distance*): computed on images whose pixels outside
  the garment + skin region are replaced by white in **both** images, so background and face cannot affect it.
  (The Dress-up paper marks LPIPS as "higher is better" and calls it diversity; we use it as a paired distance.)
* **FID** (Heusel et al. 2017, pytorch-fid Inception weights): 2,048-d pool features of 128 px images (resized to
  299 by the network); generated try-on images vs real test images. With ~1,270 images FID is **biased upwards and
  noisy**, so it is reported with the real-vs-real noise floor (T9) and with **KID** (unbiased, 100 subsets of 500,
  mean ± std). Neither is comparable with the paper's FID (different images, resolution, sample size).
* **mIoU**: confusion-matrix based over the whole split (not an average of per-image scores).

## Weights needed (downloaded once, with permission, cached in `<drive>/pretrained/torch`)

| File | Source | Size | Used by |
|---|---|---|---|
| `alexnet-owt-7be5be79.pth` | download.pytorch.org (torchvision) | ~233 MB | LPIPS backbone |
| `pt_inception-2015-12-05-6726825d.pth` | github.com/mseitzer/pytorch-fid releases | ~92 MB | FID/KID |

LPIPS's own linear-layer weights ship with the `lpips` package.

## Implementation and how to run

* Code: `src/evaluation/final_eval.py` (runner, test-split guard, audit log, results table) and
  `src/evaluation/image_metrics.py` (LPIPS, FID, KID). Tests: `tests/test_final_eval.py`, `tests/test_image_metrics.py`
  (no downloads: random AlexNet and a stand-in feature extractor; FID/KID checked against known answers).
* T5 vs T6: both use the precomputed SCHP-Pascal body maps (already the network the automatic pipeline uses). T5 takes
  the reference and target garment parses from the manual annotations; T6 takes both from SCHP-LIP. The scoring region
  always comes from the **manual** target parse.
* `raw` variants are scored before the target's face/hair pixels are pasted back. The pasted variants can gain a little
  from pasted pixels at the edge of the skin region, so the raw numbers are the cleaner comparison.
* T8 uses each split image once as the target with a seeded random reference from a different item; the "real" set is
  the same split's photos at model size. Because target backgrounds and faces appear in both sets, T8 measures realism
  of the garment area more than of whole images; T9 gives the real-vs-real floor.
* Per-pair scores for T4-T7 are stored in `results.json`, so later experiments can be compared on the same pairs with
  `paired_bootstrap_ci` (docs/experiments.md).
* Notebook Section 12: `RUN_FINAL_EVAL=True`, `FINAL_SPLIT="val"` first (rehearsal; gives validation numbers for the
  new metrics), then `FINAL_SPLIT="test"` once. `ALLOW_METRIC_DOWNLOAD=True` the first time. Results:
  `<drive>/outputs/<experiment>/final_eval/<split>_<time>/` and the Section 14 export.
