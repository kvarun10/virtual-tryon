# Final evaluation: split `test` (official)

Source: printed `results_table.md` of the official test run in Colab (2026-10-22, Tesla T4), pasted by the user.
The matching `results.json` stays in `<drive>/outputs/baseline_128/final_eval/test_20261022_065558/`.
This was the first test-split run (`earlier test-split runs: 0`).

commit af0a2ca, started 20261022_065558, checkpoints warp step 10000, texture step 13000

## T1 Stage A clothing parser vs manual parsing

1270 images: pixel accuracy 0.949, mIoU 0.616

## T2/T3 Warp

1270 images {'self': 1095, 'cross_view': 175}: mIoU (7 classes) 0.708, pixel acc 0.973; swap (64 pairs): class-area agreement 0.875, class fidelity 0.976

| class | IoU |
|---|---|
| skin | 0.680 |
| top | 0.781 |
| outer | 0.548 |
| dress | 0.779 |
| skirt | 0.728 |
| pants | 0.817 |
| footwear | 0.627 |

## T4-T7 Paired try-on (357 real cross-view pairs, region: garment + skin of the manual target parse)

| variant | SSIM (higher better) | L1 (lower) | LPIPS (lower) |
|---|---|---|---|
| T4 texture with true layout | 0.573 | 0.095 | 0.052 |
| T5 end-to-end, manual Stage A (raw) | 0.431 | 0.129 | 0.078 |
| T5 end-to-end, manual Stage A (+face/hair paste) | 0.437 | 0.128 | 0.076 |
| T6 fully automatic (raw) | 0.422 | 0.135 | 0.081 |
| T6 fully automatic (+paste) | 0.427 | 0.134 | 0.080 |
| T7 copy reference photo | 0.181 | 0.209 | 0.122 |

warp mIoU on these pairs: manual Stage A 0.684, automatic 0.474

## T8 Realism of try-on results (different-item references)

FID 23.99, KID 0.0073 ± 0.0005 (1270 vs 1270 images, KID subsets of 500)

## T9 Real-vs-real noise floor

FID 14.47, KID 0.0003 ± 0.0002 (1271 vs 1270 images, KID subsets of 500)

## T10 Cost (20 runs, Tesla T4, [128, 128] px)

| stage | median ms | p90 ms |
|---|---|---|
| stage_a_clothing_ref | 280.7 | 304.7 |
| stage_a_clothing_target | 278.7 | 288.2 |
| stage_a_body | 123.0 | 130.9 |
| warp | 11.6 | 11.8 |
| texture | 30.8 | 31.7 |
| total | 736.5 | 776.3 |

peak GPU memory: 2705.2 MiB
