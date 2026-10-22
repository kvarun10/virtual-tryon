# Results table (Phase 7 baseline)

Our numbers come from `src/evaluation/final_eval.py` (protocol: `docs/final_evaluation.md`). Each run prints and saves
its own `results_table.md`. The test column below is taken from the official test run's table
(`report/data/final_eval_test_20261022.md`). The validation final-eval column will be filled from that run's
`results.json` once its export is in `report/private/exports/`. Cells marked *pending* have not been recorded yet.
**Nothing here is comparable with the paper's numbers** (section C and `docs/paper_comparison.md`).

Models: warp `warp_pf` (step 10,000), texture `texture_pf` (step 13,000), config `configs/colab_pairfilter.yaml`,
128 × 128 px, SCHP-LIP clothing parser, SCHP-Pascal body parts, pair filter ΔE ≤ 15.

## A. Our baseline

| # | Measurement | Metric | Validation (earlier sessions) | Validation (final-eval run) | **Test (final, 2026-10-22)** |
|---|---|---|---|---|---|
| T1 | Clothing parser vs manual parsing | mIoU / pixel acc | not measured (P25) | *pending* | **0.616 / 0.949** (1,270 images) |
| T2 | Warp | mIoU, 7 trained classes | 0.715 (full val, session A) | *pending* | **0.708** (1,270 images) |
| T3 | Warp swap test | class-area agreement / class fidelity | 0.879 / 0.985 (64 pairs) | *pending* | **0.875 / 0.976** (64 pairs) |
| T4 | Texture with the true layout | SSIM / L1 / LPIPS (garment + skin) | 0.576 / 0.093 / – (260 pairs) | *pending* | **0.573 / 0.095 / 0.052** (357 pairs) |
| T5 | End-to-end, manual Stage A (raw) | SSIM / L1 / LPIPS | 0.442 / – / – | *pending* | **0.431 / 0.129 / 0.078** |
| T5 | End-to-end, manual Stage A (+face/hair paste) | SSIM / L1 / LPIPS | 0.447 / – / – | *pending* | **0.437 / 0.128 / 0.076** |
| T6 | Fully automatic (raw) | SSIM / L1 / LPIPS | not measured | *pending* | **0.422 / 0.135 / 0.081** |
| T6 | Fully automatic (+paste) | SSIM / L1 / LPIPS | not measured | *pending* | **0.427 / 0.134 / 0.080** |
| T7 | Copy the reference photo (lower bound) | SSIM / L1 / LPIPS | 0.195 / – / – | *pending* | **0.181 / 0.209 / 0.122** |
| – | Warp mIoU on the paired set | manual / automatic Stage A | 0.655 / – | *pending* | **0.684 / 0.474** |
| T8 | Realism, different-item try-on | FID / KID | not measured | *pending* | **23.99 / 0.0073 ± 0.0005** |
| T9 | Real-vs-real noise floor | FID / KID | not measured | *pending* | **14.47 / 0.0003 ± 0.0002** (real val vs real test) |
| T10 | Latency per try-on, Tesla T4 | median ms / peak GPU MiB | not measured | *pending* | **737 ms / 2,705 MiB** |

Test per-class warp IoU: skin 0.680, top 0.781, outer 0.548, dress 0.779, skirt 0.728, pants 0.817, footwear 0.627.
Test latency by stage (median ms): clothing parser 281 + 279 (reference + target), body parser 123, warp 11.6,
texture 30.8. Parsers are 93 % of the time.

How to read it:
* Expected order T4 > T5 ≥ T6 > T7 holds. **T4 − T5 = 0.142 SSIM is the cost of warp errors** (largest gap).
  **T5 − T6 = 0.009 SSIM is the cost of parser errors** (small), even though the automatic parse lowers warp mIoU
  from 0.684 to 0.474. Many parser errors are class confusions between garments of similar shape (e.g. top/outer,
  P11), which change the mIoU but hardly change the generated appearance.
* Test agrees with validation (warp 0.708 vs 0.715; T4 0.573 vs 0.576; swap 0.875 vs 0.879), so no sign that
  the model choices over-fitted the validation split.
* T8 vs T9: the try-on outputs are clearly distinguishable from real photos (KID 0.0073 vs 0.0003 for real vs real;
  FID 24.0 vs 14.5). Part of the T9 floor of 14.5 is the small sample size: FID is biased upwards with ~1,270 images.
* LPIPS is a distance (lower is better), computed only on garment + skin pixels (all other pixels white in both images).

## B. Test-split log

Test runs so far: **1** (2026-10-22 06:55, official, all six parts; `earlier test-split runs: 0` printed before it).
Audit log: `<drive>/cache/test_split_log.jsonl`. The test split is not used again until the final Phase 8 model.

## C. Published reference values (Dress-up paper, NOT our results)

| Task (paper) | Metric | Dress-up (published) |
|---|---|---|
| Pose transfer | SSIM / LPIPS / FID | 0.95 / 0.228 / 8.60 |
| Label-to-image | mIoU / Acc / FID | 87.6 / 98.9 / 8.60 |

These were measured on different data, resolution, and protocol (undocumented, `docs/paper_analysis.md` §2.6). The paper
labels LPIPS as "higher is better" (diversity); ours is a paired distance. Never put them in the same column as ours.
Why, and what can be compared: `docs/paper_comparison.md`.
