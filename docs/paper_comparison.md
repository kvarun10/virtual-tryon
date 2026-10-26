# How our system compares with the Dress-up paper

Written on 2026-10-22, after the first and only official test-split run (`report/data/final_eval_test_20261022.md`).
Paper values come from `docs/paper_analysis.md` §2.6. Our values come from our own runs. The two are never merged.

## 1. Short answer

We built a **working, end-to-end re-implementation of the Dress-up pipeline** with the same three stages:
pre-treatment, segmentation warp, and correspondence + SPADE texture. It produces try-on images for new photo pairs.
It is **not a reproduction of the paper's numbers**, and its numbers **cannot be ranked against the paper's**. The
reasons are listed in section 3. Within our own fixed protocol the system behaves as a reduced-scale baseline should:
* it is far better than the trivial copy baseline;
* the warp stage is the main source of error;
* its outputs are still clearly distinguishable from real photos.

## 2. What is the same and what differs

| Aspect | Paper | Ours | Effect |
|---|---|---|---|
| Pipeline | Pre-treatment → dual-path U-Net warp → correspondence + SPADE | same structure (deviations D1-D25 in `implementation_deviations.md`) | faithful in structure |
| Clothing parser | LIP-JppNet fine-tuned on DeepFashion | SCHP-LIP, not fine-tuned (P01) | our parser: 0.616 mIoU vs manual labels (test) |
| Body representation | NBF body parts | SCHP-Pascal, 7 coarse parts (P12) | traces the clothing outline (see section 5) |
| Data | DeepFashion In-shop, "52,712 training images" | DeepFashion-MultiModal subset with manual parsing: 10,160 train images, item-disjoint splits | ~5× less training data |
| Resolution | 256 × 256 | 128 × 128 | less texture detail |
| Training | not stated (CoCosNet-style texture nets need days on multiple V100s) | warp 10k steps (~45 min), texture 13k steps (~77 min) on one T4 | orders of magnitude less compute |
| Training pairs | same item | same item **and** same outfit (pair filter ΔE ≤ 15, D24) | cleaner but fewer real pairs |
| Identity | face/hair labels from the target | same, plus target face/hair pixels pasted (D23) | keeps the target's face |
| Evaluation | pose transfer and label-to-image tables, protocol not given; garment transfer qualitative only | try-on on held-out test pairs, fixed protocol (`final_evaluation.md`) | different tasks and protocols |

## 2b. What the paper states about its evaluation (re-read 2026-10-26, Sec. 5.1-5.2, pp. 23165-23170)

Stated:
* Data: DeepFashion In-shop ("52,712 training images"); training at 256 × 256; "at least 4 P40 GPUs".
* Quantitative evaluation uses **pose transfer** because it is "the only one that has reference images available"
  (Sec. 5.2.2), i.e. a generated image is compared with a real photo of the same clothes in the target pose.
* Metrics: SSIM, FID, and LPIPS described as measuring "generation diversity" (marked ↑).
* Baselines ADGAN, DiOr and Ivan et al. were run with their public code and models "on the DeepFashion dataset".
* Label-to-image (Table 2) uses mIoU, pixel accuracy and FID, compared with pix2pixHD, SPADE and SMIS.

Not stated: the test pairs or split and their number; the evaluation resolution and crop; whole image or a region;
SSIM settings; FID sample size; which inputs Dress-up receives for pose transfer (predicted layout from the warp
stage, or a ground-truth layout); how mIoU/Acc in Table 2 were computed.

**Correction of an earlier statement in our discussion:** I had suggested the 0.95 might come from giving the
generator the target's true layout. The paper does not say this; it is not supported by the text and is removed.
What *is* clear from Sec. 5.2.2 is that their pose-transfer test uses the same idea as our paired evaluation:
two photos of the same clothes, the second one is the ground truth.

## 2c. Paper-style benchmark on the standard pose-transfer pairs (set up 2026-10-26)

* **Pairs:** the public DeepFashion pose-transfer split of PATN (Zhu et al. 2019), used by ADGAN and DiOr:
  `fasion-resize-pairs-test.csv` (8,570 pairs, 811 items, 3,144 photos; 802,612 bytes) and
  `fasion-resize-pairs-train.csv` (101,966 pairs; 9,551,830 bytes), from the Google Drive folder linked in
  github.com/tengteng95/Pose-Transfer (MIT), SHA-256 pinned in `src/data/pose_transfer.py`. Only these two files were
  downloaded (the folder also holds keypoint annotations, a pose estimator and PATN weights, which this method does
  not use). The Dress-up paper does not name its pairs; this is the common public protocol for its Table 1 baselines.
* **Measured on the files:** every test pair is two photos of the same item; **439 of the 811 test items also occur
  in the standard training pairs** (by item number), i.e. the published methods trained on other photos of most test
  products. Our item-disjoint protocol is stricter than that.
* **Leakage against our splits:** pairs are grouped by item (`categorise_pairs`): items we trained on are never
  scored; our locked test items are skipped; validation items and items outside our splits are scored. The counts
  are produced by `scripts/pose_transfer_benchmark.py --leakage-only` (Colab, needs our split file).
* **Scoring (Table 1 style):** fully automatic pipeline, whole 176 × 256 image, SSIM / L1 / LPIPS (distance) and
  FID/KID vs the real target photos. The raw output is primary; the face/hair-pasted variant copies ground-truth pixels
  and is reported separately. Our networks run at 128 px and are upscaled, so this measures the *current* models at
  a disadvantage; the full 8,570-pair result needs retraining at 256 px with the standard test items excluded.

## 3. Why the numbers cannot be compared directly

1. **Same idea, different protocol.** Our paired evaluation (reference = one photo of an outfit, ground truth =
   another photo of the same outfit in another pose) *is* a pose-transfer test in substance, which is exactly why the
   paper chose pose transfer for its numbers (Sec. 5.2.2). The differences are the pairs, the scoring region, the
   resolution and the metric settings, not the task. Real try-on with a *different* garment has no ground truth in
   either work (the paper shows it only as pictures; we use FID/KID).
2. **Undocumented protocol.** The paper does not give its test pairs, crop, resolution, SSIM settings or FID sample size.
   It reports the same FID (8.60) for two different tasks. Its SSIM of 0.95 is far above every baseline in its own
   table (≤ 0.80).
3. **Opposite meaning of LPIPS.** The paper marks LPIPS "higher is better" (diversity). We use it as a paired distance,
   where lower is better.
4. **Region vs whole image.** Our SSIM / L1 / LPIPS count only garment + skin pixels. Whole-image scores on DeepFashion
   are inflated by the white studio background and our white letterbox padding (P20). Measured on the same
   260 real validation pairs with the same network (Experiment C's texture network, true layout): **whole image
   0.898**, person region (all non-background pixels) 0.596 (training-log validation), garment + skin 0.583
   (final-eval). The baseline B0 has region scores within 0.01 of these, but its whole-image score on these pairs
   was never measured. Since 2026-10-26 the final evaluation reports all three regions for every variant, so this
   is measured directly for B0 and for the end-to-end pipeline in the next validation run.
5. **FID depends on sample size and resolution.** Even two sets of *real* photos score FID 14.47 in our setting
   (1,270 vs 1,271 images at 128 px). That is already worse than the paper's 8.60, so our FID cannot be read on the
   paper's scale. Measured directly: real train (10,160) vs real validation (1,271) photos give FID 9.11, but real
   validation vs real test (1,271 vs 1,270) give 14.47. Only the sample size changed, so FID cannot be compared
   across different sample sizes.

## 4. Side by side, with the closest analogue of each number

| Paper metric (task) | Paper | Closest number of ours (test) | Comparable? |
|---|---|---|---|
| SSIM (pose transfer) | 0.95 | 0.573: texture stage given the true layout on real pose-change pairs (T4), garment + skin only. Full try-on: 0.431 (T5) / 0.427 (T6, fully automatic) | No: different task, region, resolution, protocol |
| LPIPS (pose transfer, "↑") | 0.228 | 0.052 (T4) / 0.080 (T6), masked distance, lower is better | No: opposite meaning, masked |
| FID (pose transfer) | 8.60 | 23.99 for try-on outputs vs a real-vs-real floor of 14.47 | No: our real-vs-real floor is already above 8.60 |
| mIoU / Acc (label-to-image) | 87.6 / 98.9 | not measured (would need a parser run on generated images). Our warp mIoU 0.708 measures a different thing (predicted layout vs true layout) | No |

The fairest summary uses relative statements inside our own protocol (section 5). These are valid because every
variant was scored on the same 357 test pairs.

## 5. How well it works (test split, our protocol)

* **Better than copying:** the full automatic try-on more than doubles SSIM over copying the reference photo
  (0.427 vs 0.181) and lowers LPIPS from 0.122 to 0.080.
* **The warp stage is the bottleneck:** with the true layout the texture stage reaches 0.573 SSIM. With the predicted
  layout it reaches 0.431. Warp errors cost 0.142 SSIM, much more than anything else.
* **Parser errors matter less than expected:** the automatic parser lowers warp mIoU from 0.684 to 0.474, but SSIM only
  from 0.431 to 0.422. Most parser mistakes swap labels between garments of similar shape (e.g. top/outer, P11).
* **Weak classes:** outer 0.548 and footwear 0.627 IoU; top 0.781, pants 0.817 and dress 0.779 are the strongest.
* **Realism gap:** KID 0.0073 for try-on outputs vs 0.0003 for real vs real photos. The outputs are noticeably blurrier
  and less detailed than real photos (128 px, short training).
* **Visible failure in the test grids:** a long-sleeved shirt put on a person wearing a tank top came out
  *sleeveless*. The body map follows the target's own clothing outline (P12). The swap metrics measure the same
  effect (class-area agreement 0.875).
* **Generalisation:** test numbers match validation within about 0.01 (warp 0.708 vs 0.715; texture 0.573 vs 0.576;
  full automatic try-on 0.427 vs 0.438; FID 23.99 vs 23.67), so the
  model choices were not tuned to the validation split.
* **Speed:** 0.74 s per try-on on a T4 at 128 px. 93 % of that time is the two SCHP parsers; warp + texture take 42 ms.
  The paper reports no speed figures. Peak GPU memory is 2.7 GB.

## 6. What would bring it closer to the paper's setting

Ranked by measured impact (`docs/experiments.md`):
1. Better warp body input (DensePose, Exp. A) and class balancing (Exp. B). The warp costs the most SSIM.
2. More real pose-change pairs and a stronger discriminator for the texture stage (Exp. C, D).
3. 256 px, the paper's resolution (Exp. E).

Even then, a numeric match with the paper would need its exact test protocol, which it does not publish.
