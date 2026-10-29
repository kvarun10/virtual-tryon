# Paper-style pose-transfer benchmark (standard DeepFashion test pairs, whole 176 x 256 images)

Standard test pairs: 8570; by leakage category: {'clean_unseen': 7012, 'clean_val_items': 942, 'our_test_items_LOCKED': 580, 'our_train_items_LEAKED': 0, 'image_missing': 36}.
Scored here: 7954 pairs ({'clean_unseen': 7012, 'clean_val_items': 942}). Our networks run at [256, 256] and are upscaled to 176 x 256, which lowers sharpness-sensitive scores.

| variant | SSIM (ours) | SSIM (common protocol code) | L1 (lower) | LPIPS (lower, distance) |
|---|---|---|---|---|
| ours, raw output (primary) | 0.7028 | 0.7015 | 0.0657 | 0.2010 |
| ours + target face/hair pixels (uses GT pixels) | 0.7426 | 0.7423 | 0.0588 | 0.1897 |
| copy the reference photo (lower bound) | 0.5144 | 0.5143 | 0.1582 | 0.4633 |

FID 12.83, KID 0.0051 ± 0.0006 (7954 generated vs 2927 real target photos; FID depends on sample size, see docs/paper_comparison.md)

Paper's Table 1 for reference (protocol unpublished; LPIPS there marked 'higher is better'): Dress-up 0.95 / 0.228 / FID 8.60; ADGAN 0.77 / 0.226 / 16.69; DiOr 0.80 / 0.176 / 14.34.