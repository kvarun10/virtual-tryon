# exp_c_realpairs vs baseline B0 (val split)

Source: notebook Section 13 output of the Colab run (2026-10-24), pasted by the user; the same file is saved as
`<drive>/outputs/exp_c_realpairs/final_eval/val_20261024_082359/comparison.md`.
Training: 13,000 steps, 82.4 min, real-pair sampling p = 0.8. Validation person-region SSIM during training (260 real
pairs, true layout): 0.533 (1k), 0.577 (2k), 0.591 (4k), 0.602 (7k), 0.602 (10k, best), 0.596 (13k). Evaluation uses
`last.pt` (step 13,000). Discriminator hinge d stayed at 1.6-2.1 throughout (as in B0, P22).

baseline run 20261022_055810, experiment run 20261024_082359; checkpoints warp .../baseline_128/warp_pf/last.pt (step 10000), texture .../exp_c_realpairs/texture/last.pt (step 13000)

## Paired try-on (260 identical pairs; Δ = experiment − B0, 95 % bootstrap CI)

| variant | metric | B0 | experiment | Δ | 95 % CI | verdict |
|---|---|---|---|---|---|---|
| T4 texture with true layout | ssim | 0.5758 | 0.5826 | +0.0068 | [+0.0018, +0.0121] | better |
| T4 texture with true layout | l1 | 0.0926 | 0.0888 | -0.0038 | [-0.0053, -0.0024] | better |
| T4 texture with true layout | lpips | 0.0497 | 0.0471 | -0.0026 | [-0.0037, -0.0014] | better |
| T5 end-to-end, manual Stage A (raw) | ssim | 0.4416 | 0.4475 | +0.0060 | [+0.0018, +0.0105] | better |
| T5 end-to-end, manual Stage A (raw) | l1 | 0.1256 | 0.1222 | -0.0034 | [-0.0047, -0.0021] | better |
| T5 end-to-end, manual Stage A (raw) | lpips | 0.0751 | 0.0722 | -0.0029 | [-0.0041, -0.0017] | better |
| T6 fully automatic (+paste) | ssim | 0.4379 | 0.4402 | +0.0023 | [-0.0019, +0.0066] | no clear change |
| T6 fully automatic (+paste) | l1 | 0.1301 | 0.1283 | -0.0018 | [-0.0033, -0.0004] | better |
| T6 fully automatic (+paste) | lpips | 0.0775 | 0.0749 | -0.0026 | [-0.0038, -0.0012] | better |
| T7 copy (sanity: must not change) | ssim | 0.1950 | 0.1950 | +0.0000 | [+0.0000, +0.0000] | no clear change |
| T7 copy (sanity: must not change) | l1 | 0.2073 | 0.2073 | +0.0000 | [+0.0000, +0.0000] | no clear change |
| T7 copy (sanity: must not change) | lpips | 0.1253 | 0.1253 | +0.0000 | [+0.0000, +0.0000] | no clear change |

## Warp (whole split)

Unchanged by design (the frozen B0 warp is used): mIoU 0.715, area agreement 0.879, class fidelity 0.985.

## Realism (different-item try-on vs real images; lower is better)

FID 23.67 → 25.73 (+2.06); KID 0.0074 → 0.0103 (+0.0028; B0's KID std over subsets was 0.0005)
