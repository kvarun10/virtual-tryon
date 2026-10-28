# Paper-style pose-transfer benchmark (standard DeepFashion test pairs, whole 176 x 256 images)

Standard test pairs: 8570; by leakage category: {'clean_unseen': 7048, 'clean_val_items': 942, 'our_test_items_LOCKED': 580, 'our_train_items_LEAKED': 0, 'image_missing': 0}.
Scored here: 7990 pairs ({'clean_unseen': 7048, 'clean_val_items': 942}). Our networks run at [256, 256] and are upscaled to 176 x 256, which lowers sharpness-sensitive scores.

| variant | SSIM (ours) | SSIM (common protocol code) | L1 (lower) | LPIPS (lower, distance) |
|---|---|---|---|---|
| ours, raw output (primary) | 0.7014 | 0.7000 | 0.0667 | 0.2039 |
| ours + target face/hair pixels (uses GT pixels) | 0.7412 | 0.7409 | 0.0597 | 0.1925 |
| copy the reference photo (lower bound) | 0.5138 | 0.5136 | 0.1584 | 0.4639 |

FID 13.14, KID 0.0055 ± 0.0006 (7990 generated vs 2932 real target photos; FID depends on sample size, see docs/paper_comparison.md)

Paper's Table 1 for reference (protocol unpublished; LPIPS there marked 'higher is better'): Dress-up 0.95 / 0.228 / FID 8.60; ADGAN 0.77 / 0.226 / 16.69; DiOr 0.80 / 0.176 / 14.34.