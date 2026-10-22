# Final evaluation: split `val` (official)

Source: `results_table.md` of the official validation run (Colab T4), exported to `report/private/exports/`; per-pair scores in its `results.json` (private).

commit af0a2ca, started 20261022_055810, checkpoints warp step 10000, texture step 13000

## T1 Stage A clothing parser vs manual parsing

1271 images: pixel accuracy 0.951, mIoU 0.628

## T2/T3 Warp

1271 images {'self': 1157, 'cross_view': 114}: mIoU (7 classes) 0.715, pixel acc 0.973; swap (64 pairs): class-area agreement 0.879, class fidelity 0.985

| class | IoU |
|---|---|
| skin | 0.690 |
| top | 0.772 |
| outer | 0.587 |
| dress | 0.785 |
| skirt | 0.720 |
| pants | 0.818 |
| footwear | 0.633 |

## T4-T7 Paired try-on (260 real cross-view pairs, region: garment + skin of the manual target parse)

| variant | SSIM (higher better) | L1 (lower) | LPIPS (lower) |
|---|---|---|---|
| T4 texture with true layout | 0.576 | 0.093 | 0.050 |
| T5 end-to-end, manual Stage A (raw) | 0.442 | 0.126 | 0.075 |
| T5 end-to-end, manual Stage A (+face/hair paste) | 0.447 | 0.125 | 0.074 |
| T6 fully automatic (raw) | 0.432 | 0.131 | 0.079 |
| T6 fully automatic (+paste) | 0.438 | 0.130 | 0.078 |
| T7 copy reference photo | 0.195 | 0.207 | 0.125 |

warp mIoU on these pairs: manual Stage A 0.655, automatic 0.491

## T8 Realism of try-on results (different-item references)

FID 23.67, KID 0.0074 ± 0.0005 (1271 vs 1271 images, KID subsets of 500)

## T9 Real-vs-real noise floor

FID 9.11, KID 0.0000 ± 0.0002 (10160 vs 1271 images, KID subsets of 500)

## T10 Cost (20 runs, Tesla T4, [128, 128] px)

| stage | median ms | p90 ms |
|---|---|---|
| stage_a_clothing_ref | 298.6 | 311.4 |
| stage_a_clothing_target | 295.7 | 302.1 |
| stage_a_body | 124.2 | 125.9 |
| warp | 11.5 | 11.8 |
| texture | 30.5 | 30.8 |
| total | 769.1 | 783.4 |

peak GPU memory: 2698.4 MiB
