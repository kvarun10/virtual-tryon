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

## 3. Why the numbers cannot be compared directly

1. **Different tasks.** The paper's numeric tables are for *pose transfer* and *label-to-image*. Its garment-transfer
   (try-on) results are only shown as pictures. We measure try-on: reference garment → target person.
2. **Undocumented protocol.** The paper does not give its test pairs, crop, resolution, SSIM settings or FID sample size.
   It reports the same FID (8.60) for two different tasks. Its SSIM of 0.95 is far above every baseline in its own
   table (≤ 0.80).
3. **Opposite meaning of LPIPS.** The paper marks LPIPS "higher is better" (diversity). We use it as a paired distance,
   where lower is better.
4. **Region vs whole image.** Our SSIM / L1 / LPIPS count only garment + skin pixels. Whole-image scores on DeepFashion
   are inflated by the white studio background (P20). On the full validation split, the same texture outputs scored
   0.883 SSIM on the whole image but 0.540 on the person region.
5. **FID depends on sample size and resolution.** Even two sets of *real* photos score FID 14.47 in our setting
   (1,270 vs 1,271 images at 128 px). That is already worse than the paper's 8.60, so our FID cannot be read on the
   paper's scale.

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
* **Generalisation:** test numbers match validation within 0.01 (warp 0.708 vs 0.715; texture 0.573 vs 0.576), so the
  model choices were not tuned to the validation split.
* **Speed:** 0.74 s per try-on on a T4 at 128 px. 93 % of that time is the two SCHP parsers; warp + texture take 42 ms.
  The paper reports no speed figures. Peak GPU memory is 2.7 GB.

## 6. What would bring it closer to the paper's setting

Ranked by measured impact (`docs/experiments.md`):
1. Better warp body input (DensePose, Exp. A) and class balancing (Exp. B). The warp costs the most SSIM.
2. More real pose-change pairs and a stronger discriminator for the texture stage (Exp. C, D).
3. 256 px, the paper's resolution (Exp. E).

Even then, a numeric match with the paper would need its exact test protocol, which it does not publish.
