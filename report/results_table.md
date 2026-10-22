# Results table (Phase 7 baseline)

Our numbers come from `src/evaluation/final_eval.py` (protocol: `docs/final_evaluation.md`). Each run writes its own
`results_table.md`; the values below are copied from those files by code, never by hand, and every cell names its source.
Cells marked *pending* have not been measured yet. **Nothing here is comparable with the paper's numbers** (section C).

Models: warp `warp_pf` (step 10,000), texture `texture_pf` (step 13,000), config `configs/colab_pairfilter.yaml`,
128 × 128 px, SCHP-LIP clothing parser, SCHP-Pascal body parts, pair filter ΔE ≤ 15.

## A. Our baseline

| # | Measurement | Metric | Validation (earlier sessions) | Validation (final-eval run) | **Test (final, once)** |
|---|---|---|---|---|---|
| T1 | Clothing parser vs manual parsing | mIoU / pixel acc | not measured (P25) | *pending* | *pending* |
| T2 | Warp | mIoU, 7 trained classes | 0.715 (full val, session A) | *pending* | *pending* |
| T3 | Warp swap test | class-area agreement | 0.879 (64 pairs, session A) | *pending* | *pending* |
| T4 | Texture with the true layout | SSIM / L1 / LPIPS (garment + skin) | 0.576 / – / – (260 pairs, session B) | *pending* | *pending* |
| T5 | End-to-end, manual Stage A (raw) | SSIM / L1 / LPIPS | 0.442 / – / – (session B) | *pending* | *pending* |
| T5 | End-to-end, manual Stage A (+face/hair paste) | SSIM / L1 / LPIPS | 0.447 / – / – (session B) | *pending* | *pending* |
| T6 | Fully automatic (SCHP Stage A, +paste) | SSIM / L1 / LPIPS | not measured | *pending* | *pending* |
| T7 | Copy the reference photo (lower bound) | SSIM / L1 / LPIPS | 0.195 / – / – (session B) | *pending* | *pending* |
| T8 | Realism, different-item try-on | FID / KID | not measured | *pending* | *pending* |
| T9 | Real-vs-real noise floor | FID / KID | not measured | *pending* | *pending* |
| T10 | Latency per try-on, T4 | ms (median), peak GPU MiB | not measured | *pending* | *pending* |

How to read it:
* T4 > T5 > T7 is the expected order. T4 − T5 is the cost of warp errors; T5 − T6 is the cost of parser errors.
* T8 means little on its own: compare it with T9, which is what two sets of **real** photos score at the same sample size.
* LPIPS is a distance (lower is better), computed only on garment + skin pixels (everything else set to white in both images).

## B. Session log of the test split

The audit log `<drive>/cache/test_split_log.jsonl` records every test run. Copy its line count here after the final run:
test runs = *pending*.

## C. Published reference values (Dress-up paper, NOT our results)

| Task (paper) | Metric | Dress-up (published) |
|---|---|---|
| Pose transfer | SSIM / LPIPS / FID | 0.95 / 0.228 / 8.60 |
| Label-to-image | mIoU / Acc / FID | 87.6 / 98.9 / 8.60 |

These were measured on different data, resolution, and protocol (undocumented, `docs/paper_analysis.md` §2.6). The paper
labels LPIPS as "higher is better" (diversity); ours is a paired distance. Never put them in the same column as ours.
