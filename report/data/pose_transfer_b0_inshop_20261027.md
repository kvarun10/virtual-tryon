# Paper-style pose-transfer benchmark (standard DeepFashion test pairs, whole 176 x 256 images)

Standard test pairs: 8570; by leakage category: {'clean_unseen': 628, 'clean_val_items': 942, 'our_test_items_LOCKED': 580, 'our_train_items_LEAKED': 6420, 'image_missing': 0}.
Scored here: 1570 pairs ({'clean_unseen': 628, 'clean_val_items': 942}). Our networks run at [128, 128] and are upscaled to 176 x 256, which lowers sharpness-sensitive scores.

| variant | SSIM (higher better) | L1 (lower) | LPIPS (lower, distance) |
|---|---|---|---|
| ours, raw output (primary) | 0.6305 | 0.0910 | 0.3964 |
| ours + target face/hair pixels (uses GT pixels) | 0.6593 | 0.0854 | 0.3837 |
| copy the reference photo (lower bound) | 0.4832 | 0.1662 | 0.4761 |

FID 97.62, KID 0.0832 ± 0.0015 (1570 generated vs 621 real target photos; FID depends on sample size, see docs/paper_comparison.md)

Paper's Table 1 for reference (protocol unpublished; LPIPS there marked 'higher is better'): Dress-up 0.95 / 0.228 / FID 8.60; ADGAN 0.77 / 0.226 / 16.69; DiOr 0.80 / 0.176 / 14.34.