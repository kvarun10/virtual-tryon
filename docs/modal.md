# Full-data 256 px run on Modal (experiment `full_256`)

Budget: $30 Modal Starter credit; deadline ~1.5 days (from 2026-10-27). B0 (`baseline_128`, tag baseline-v1) stays frozen
and is the comparison point. Code: `modal_app.py`, `scripts/modal_upload.py`, `src/data/packed.py`, `src/data/packing.py`,
`configs/modal_full_256.yaml`.

## What changes vs B0 (decided before training; no tuning on validation)

| | B0 | full_256 |
|---|---|---|
| photos | 10,160 manually parsed DFMM photos | all In-shop photos except val/test items and the standard benchmark test items |
| garment labels | DFMM manual | manual where it exists, SCHP-LIP otherwise (`data.label_source=manual_then_auto`) |
| resolution | 128 px | 256 px (paper) |
| texture widths | corr 32, generator 32 ch | corr 64, generator 64 ch (CoCosNet / SPADE defaults) |
| batch | 8 | 16 |
| weights used | live | EMA of the generator (decay 0.999) |
| losses, learning rates, pair filter, label sets | | unchanged |

Facts behind the data choice (measured 2026-10-27):
* The In-shop `img.zip` photos (256 x 256) are the DFMM photos letterboxed to 174 x 256 at column 41: DFMM manual
  parses resized with our letterbox overlay them exactly. All 44,096 DFMM photos are among the 52,712 In-shop photos;
  8,616 are In-shop only. So no 5.4 GB DFMM image download is needed, and training/benchmark use the benchmark's own images.
* The standard benchmark (PATN pairs) is built from these In-shop photos; the benchmark crops columns 41:215.
* `BatchedSCHP` (GPU batch + `grid_sample` inverse warp) agrees with the original one-image SCHP on 99.99 % of pixels
  (3 photos, both parsers, real weights).

The packing job reports how well SCHP-LIP agrees with the manual labels on the 12,701 parsed photos
(`report.json`, `parser_agreement_vs_manual_garment12`); this is the quality of the automatic labels.

## Napkin maths (before measuring; the probe replaces the assumptions)

Measured on T4 at 128 px, batch 8 (implementation_status.md): warp 0.22 s/step, texture 0.35 s/step.
At 256 px with doubled texture widths: warp ~4x, texture ~16x (4x pixels; 4x conv FLOPs from 2x channels; the
64 x 64 correspondence is 16x the 32 x 32 one). Assumed speed vs T4 for fp16 conv nets: L4 2x, A10 2.5x, L40S 5x,
A100 6x, H100 10x.

Planned length: warp 12k steps, texture 14k steps at batch 16 (B0: 10k / 13k at batch 8 on 1/4 of the data).

| GPU | $/h (+CPU/RAM ~0.4) | warp 12k | texture 14k | GPU $ total | wall time (stages in parallel) |
|---|---|---|---|---|---|
| T4 | 0.59 | 6 h | 43 h | 29 | 43 h: too slow |
| A100 40 GB | 2.10 | 1.0 h | 7.2 h | 17 (+3) | ~7 h |
| L40S | 1.95 | 1.2 h | 8.6 h | 19 (+4) | ~9 h |
| H100 | 3.95 | 0.6 h | 4.3 h | 19.5 (+2) | ~4.5 h |

All fast GPUs cost about the same per step, so the shortest wall time wins: H100 unless the probe shows it is less
than ~1.9x faster than A100 40 GB. Plan: setup (CPU, ~$0.1), pack (L4, ~$0.6), probe on H100/A100/L40S (~$1),
both stages in parallel (~$21), two benchmarks (L4, ~$1); ~$6 reserve. Step counts are fixed from the probe so the
total stays under $25.

## Measured (2026-10-27): GPU probe, batch 16, 256 px, 40 steps, median s/step after warm-up

Measured on the 200-photo smoke-test store (speed per step does not depend on dataset size; means are higher because
the tiny store restarts its DataLoader every few steps).

| GPU | warp s/step | texture s/step | texture peak memory |
|---|---|---|---|
| H100 80 GB | 0.024 | 0.228 | 21 GB |
| A100 40 GB | 0.048 | 0.439 | 30 GB |
| L40S | 0.053 | 0.600 | 21 GB |

The napkin estimate (texture ~1.1 s/step on H100) was ~4x too pessimistic. H100 is 1.9x the A100 for 1.9x the price,
so it costs the same per step and halves the wall time: texture on H100. Warp at 0.024 s/step needs ~670 samples/s,
which the CPU workers (per-class affine, body-part jitter) cannot deliver, so warp is data-bound: it runs on a cheaper
GPU with more CPU cores. With the same budget the run is lengthened (decided before training, from cost only):
texture ~50k steps (~3.5 h, ~$15), warp ~40k steps.

Smoke-test packing (200 photos, L4): 11 s; SCHP-LIP vs manual on 55 photos: pixel accuracy 0.965, mIoU 0.594 (small sample).

## Order of work

1. `python scripts/modal_upload.py` (2.7 GB: In-shop photos as one zip, partition list, cache + pretrained Drive zips,
   B0 generator_only files).
2. `modal run modal_app.py::setup` - unpacks; downloads the two pair lists (approved 2026-10-26) and the DFMM
   parsing archive (90 MB, official link, as in Colab).
3. `modal run modal_app.py::pack` - store + report; `modal run modal_app.py::pack --limit 200` first as a smoke test.
4. `modal run modal_app.py::probe` - seconds/step and memory on three GPUs; choose GPU and steps.
5. `modal run --detach modal_app.py::train --stage warp ...` and `--stage texture ...` in parallel. A short first
   call (e.g. 1,500 steps) is checked (losses, validation grid) before continuing the same run to full length;
   resume is exact, so nothing is wasted.
6. `modal run modal_app.py::benchmark --which b0` and `--which full_256`: same pairs rule, same images.

## Honest limits

* The paper's evaluation protocol is unpublished (paper_comparison.md); matching 0.95 SSIM is not guaranteed.
* Our locked test items stay unscored in the benchmark; the count is in the report.
* Automatic labels carry SCHP errors (e.g. top/outer confusion, P11) into training targets.

## Results (2026-10-27)

Training: warp 60,000 steps (L40S, 12 CPU cores; 40k planned, extended to 60k because validation mIoU was still
rising, val CE still falling); texture 50,000 steps (H100, 210.7 min). For both, the final checkpoint was also the best
validation checkpoint; validation never declined (no sign of overfitting). Total Modal cost $22.99 (all from the $30
Starter credit; billed $0). Final generator weights (EMA): `checkpoints/full_256/{warp,texture}/generator_only.pt`
on the volume, copied to `report/private/models/full_256/` (not in git).

### Paper-style benchmark (standard PATN test pairs, whole 176 x 256 image, In-shop photos, fully automatic)

| | pairs | SSIM | L1 | LPIPS (distance) | FID |
|---|---|---|---|---|---|
| **full_256, raw output** | 7,990 | **0.699** | 0.068 | **0.205** | **13.64** (7,990 vs 2,932 real) |
| full_256 + target face/hair pixels (uses GT pixels) | 7,990 | 0.739 | 0.061 | 0.193 | |
| copy the reference photo | 7,990 | 0.514 | 0.158 | 0.464 | |
| ADGAN / DiOr / Dress-up (published Table 1) | 8,570? | 0.77 / 0.80 / 0.95 | | 0.226 / 0.176 / 0.228 | 16.69 / 14.34 / 8.60 |

Same 1,570 pairs as B0 (paired bootstrap, 95 % CI of new - B0):

| | B0 (128 px) | full_256 | difference |
|---|---|---|---|
| SSIM raw | 0.631 | 0.676 | +0.045 [+0.043, +0.047] |
| LPIPS raw | 0.396 | 0.218 | -0.178 [-0.181, -0.175] |
| L1 raw | 0.091 | 0.072 | -0.019 [-0.021, -0.018] |
| SSIM with face/hair paste | 0.659 | 0.718 | +0.059 [+0.057, +0.061] |

7,990 of 8,570 pairs were scorable: no standard test item was trained on; 580 pairs of our locked test items stay
unscored (test split used once). B0 could only be scored on 1,570 pairs (it trained on the items of the other 6,420).

How it compares with the published numbers (protocol differences: the paper does not publish its pairs, resolution,
region or FID sample size; ADGAN/DiOr are scored on all 8,570 pairs; FID depends on sample size and Inception code):
* **FID 13.64 is lower than ADGAN (16.69) and DiOr (14.34)**, but higher than Dress-up's 8.60.
* **LPIPS 0.205** (as a distance, lower is better) is lower than ADGAN 0.226 and Dress-up 0.228, higher than DiOr
  0.176. The paper labels LPIPS "higher is better" ("diversity"), so its LPIPS column is not directly comparable.
* **SSIM 0.699 is below** ADGAN 0.77, DiOr 0.80 and Dress-up 0.95.

### Garment layout (warp stage), all 1,271 validation photos, same pairs for both models

| definition | B0 (128 px) | full_256 | paper Table 2 |
|---|---|---|---|
| mIoU excl. background (training metric) | 70.8 | **77.9** | 87.6 (definition not published) |
| mIoU incl. background | 74.3 | **80.5** | |
| pixel accuracy | 97.2 | **98.0** | 98.9 |
| final layout (warp + automatic face/hair), all 12 classes incl. background | 62.1 | 66.2 | |

**Correction (2026-10-27):** the paper's Table 2 (mIoU 87.6 / Acc 98.9) is a *label-to-image* score (SPADE protocol: a parser is run on images generated from a given layout and compared with that layout; docs/paper_analysis.md 2.6). It does not measure the warp stage, so the warp mIoU above must not be compared with it. The comparable test would run SCHP on texture outputs generated from ground-truth layouts.

The final-layout definition counts headwear and accessories, which the method never transfers (IoU 0), so it is lower.
Per-class IoU (full_256, training metric): pants 0.85, dress 0.84, top 0.82, skirt 0.80, skin 0.75, outer 0.71,
footwear 0.67. Files: report/data/layout_eval_*_20261027.md, pose_transfer_*_20261027.md, pack256_report_20261027.json.

## v2 (2026-10-28): what was tried, what was kept

All selection on validation try-ons (318 pose-change pairs of validation products, `modal_app.py::val_tryon`); the
benchmark was run once on the chosen combination. Friend's workspace (lordjack265) for the texture work; cost there
$10.09, own workspace $27.63 total.

| change | validation try-on SSIM / LPIPS (paired vs current) | decision |
|---|---|---|
| DensePose body maps for the warp (retrained, 60k) | -0.016 [-0.018, -0.014] / +0.013 [+0.010, +0.016] | rejected: the clothed outline of SCHP-Pascal is a correct hint in pose transfer (same outfit) |
| texture fine-tune (L1 5 + contextual 1 + LR decay), best val step 6k | +0.0014 [+0.0010, +0.0018] / -0.0009 (n.s.) | kept |
| same, final step 20k | +0.0002 (n.s.) / +0.0002 (n.s.) | not used |

Final model: warp `full_256/warp` (SCHP-Pascal, 60k) + texture `v2_256/texture_ft/best.pt` (step 6k, EMA).

### Final benchmark (standard pairs, 7,990 scorable, whole 176 x 256, fully automatic)

| | full_256 | **v2 final** | paired difference (95% CI) |
|---|---|---|---|
| SSIM (ours) | 0.6993 | **0.7014** | +0.0021 [+0.0020, +0.0022] |
| SSIM (common protocol code: skimage, sigma 1.5, data range of the generated image) | - | **0.7000** | |
| LPIPS (AlexNet, distance) | 0.2048 | **0.2039** | -0.0009 [-0.0011, -0.0007] |
| L1 | 0.0680 | 0.0667 | -0.0013 [-0.0014, -0.0013] |
| FID (vs 2,932 real target photos) | 13.64 | **13.14** | |

Placement among methods re-scored on a common protocol (PIDM, CVPR 2023, Table 1, DeepFashion 256 x 176; read from the
paper): ADGAN 14.46 / 0.672 / 0.228, PISE 13.61 / 0.663 / 0.206, GFLA 10.57 / 0.707 / 0.234, DPTN 11.39 / 0.711 /
0.193, CASD 11.37 / 0.725 / 0.194, NTED 8.68 / 0.718 / 0.175, PIDM 6.37 / 0.731 / 0.168 (FID / SSIM / LPIPS). Ours
13.14 / 0.700 / 0.204: better than ADGAN, PISE and PATN on all three; LPIPS also better than GFLA. Caveats: 7,990 of
8,570 pairs (our locked test items are not scored), FID sample definitions may differ, our model never saw a standard
test item while those methods trained on the standard split (439 test items also in its training pairs).

### Where the points are lost (oracle diagnostics, validation try-ons, v2 final model)

| setting | SSIM | LPIPS |
|---|---|---|
| fully automatic | 0.7735 | 0.167 |
| manual garment labels instead of SCHP-LIP | 0.7786 (+0.005) | 0.161 |
| + true target layout instead of the warp output | **0.8171 (+0.038)** | **0.122** |

The warp costs ~8x more than the parser; with a perfect layout the texture network reaches 0.817. Next upgrade worth
compute: the warp (capacity / body-garment attention / reference image as input), then the texture correspondence.
