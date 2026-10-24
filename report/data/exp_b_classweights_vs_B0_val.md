# exp_b_classweights vs baseline B0 (val split)

Source: notebook Section 13 output of the Colab run (2026-10-24), pasted by the user; the same file is saved as
`<drive>/outputs/exp_b_classweights/final_eval/val_20261024_064300/comparison.md`.
Training: 10,000 steps, 40.4 min, best validation mIoU (400-image check) 0.6947 at step 8,000; evaluation uses `last.pt`.

baseline run 20261022_055810, experiment run 20261024_064300; checkpoints warp .../exp_b_classweights/warp/last.pt (step 10000), texture .../baseline_128/texture_pf/last.pt (step 13000)

## Paired try-on (260 identical pairs; Δ = experiment − B0, 95 % bootstrap CI)

| variant | metric | B0 | experiment | Δ | 95 % CI | verdict |
|---|---|---|---|---|---|---|
| T4 texture with true layout | ssim | 0.5758 | 0.5758 | +0.0000 | [+0.0000, +0.0000] | no clear change |
| T4 texture with true layout | l1 | 0.0926 | 0.0926 | +0.0000 | [+0.0000, +0.0000] | no clear change |
| T4 texture with true layout | lpips | 0.0497 | 0.0497 | +0.0000 | [+0.0000, +0.0000] | no clear change |
| T5 end-to-end, manual Stage A (raw) | ssim | 0.4416 | 0.4030 | -0.0385 | [-0.0456, -0.0314] | worse |
| T5 end-to-end, manual Stage A (raw) | l1 | 0.1256 | 0.1175 | -0.0081 | [-0.0099, -0.0063] | better |
| T5 end-to-end, manual Stage A (raw) | lpips | 0.0751 | 0.0710 | -0.0041 | [-0.0057, -0.0025] | better |
| T6 fully automatic (+paste) | ssim | 0.4379 | 0.4108 | -0.0271 | [-0.0337, -0.0206] | worse |
| T6 fully automatic (+paste) | l1 | 0.1301 | 0.1208 | -0.0093 | [-0.0114, -0.0073] | better |
| T6 fully automatic (+paste) | lpips | 0.0775 | 0.0729 | -0.0046 | [-0.0063, -0.0029] | better |
| T7 copy (sanity: must not change) | ssim | 0.1950 | 0.1950 | +0.0000 | [+0.0000, +0.0000] | no clear change |
| T7 copy (sanity: must not change) | l1 | 0.2073 | 0.2073 | +0.0000 | [+0.0000, +0.0000] | no clear change |
| T7 copy (sanity: must not change) | lpips | 0.1253 | 0.1253 | +0.0000 | [+0.0000, +0.0000] | no clear change |

## Warp (whole split)

| metric | B0 | experiment | Δ |
|---|---|---|---|
| mean_iou_7 | 0.715 | 0.685 | -0.030 |
| class_area_agreement | 0.879 | 0.879 | +0.000 |
| class_fidelity | 0.985 | 0.992 | +0.007 |
| IoU skin | 0.690 | 0.675 | -0.015 |
| IoU top | 0.772 | 0.774 | +0.002 |
| IoU outer | 0.587 | 0.587 | -0.000 |
| IoU dress | 0.785 | 0.746 | -0.039 |
| IoU skirt | 0.720 | 0.630 | -0.089 |
| IoU pants | 0.818 | 0.810 | -0.008 |
| IoU footwear | 0.633 | 0.571 | -0.063 |

## Realism (different-item try-on vs real images; lower is better)

FID 23.67 → 25.26 (+1.60); KID 0.0074 → 0.0089 (+0.0014)

## Training observation

From step ~6,900 the warp discriminator loss fell to ~0.0001 and the generator's adversarial loss sat at ~0.50
(LSGAN: the discriminator separates real and predicted maps perfectly and gives no useful gradient). The earlier
baseline warp run ended with d ≈ 0.07-0.12 (`warp_train_log_10k.csv`). See problems log P31.
