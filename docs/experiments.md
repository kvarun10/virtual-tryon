# Experiment roadmap (Phase 8)

Written on 2026-10-22, after the Phase 7 protocol was fixed and before any experiment was run. **No result in this file
exists yet.** Each experiment changes **one** variable relative to the Phase 7 baseline and is judged on the
**validation split only**. The test split is used again only at the very end, once, for the final chosen model,
and that run is logged like the first one (`docs/final_evaluation.md`).

## Common ground for every experiment

**Baseline B0** (Phase 7): warp `warp_pf` (10k steps), texture `texture_pf` (13k steps), `configs/colab_pairfilter.yaml`,
128 px, SCHP-LIP clothing, SCHP-Pascal body (7 parts), pair filter `classes_color` ΔE ≤ 15, seed 42.

**Fixed unless the experiment says otherwise:** dataset and item-disjoint splits (10,160 / 1,271 / 1,270), pair filter,
image size, model widths, optimiser settings, number of steps, seed, metric definitions (warp metric v2, texture
metric v3, Phase 7 region and LPIPS masking).

**How each experiment is run:** a config file `configs/exp_<letter>_<name>.yaml` with `_base_: colab_pairfilter.yaml` and
its own `training.warp_run_name` / `texture_run_name`, so B0's checkpoints are never overwritten. Evaluation is notebook
Section 12 with `FINAL_SPLIT="val"`, `FINAL_LIMIT=None`.

**Comparing with B0:** paired metrics are compared on the *same* validation pairs with
`src.evaluation.final_eval.paired_bootstrap_ci` (per-pair scores are stored in `results.json`). mIoU is compared per
class and overall. Unless an experiment states its own rule, it is **adopted** only if:
1. its primary metric improves and the 95 % bootstrap interval of the difference excludes 0 (for paired metrics), or
   mIoU improves by ≥ 0.01 with no trained class dropping by > 0.02 (for the warp);
2. no secondary metric gets clearly worse (swap area agreement −0.02, end-to-end SSIM CI entirely below 0);
3. it fits the compute budget of one Colab T4 session (~2-3 h).

A single seed is used because of compute limits. Differences that are smaller than the seed noise therefore cannot
be ruled out. Where an adopted change is borderline, a second seed is run before it becomes the new baseline.

**Why this order.** On validation, end-to-end SSIM is 0.442 while the texture stage given the true layout reaches
0.576 (session B), so **warp errors cost about 0.13 SSIM**: more than anything measured on the texture side. The warp
experiments therefore come first.

| ID | Experiment | Problem | Stage | Priority | Est. T4 cost |
|---|---|---|---|---|---|
| A | DensePose body maps instead of SCHP-Pascal | P12, P19 | warp | 1 | DFMM DensePose download (~5.6 GB, once) + ~45 min |
| B | Class-weighted cross-entropy | P19 | warp | 2 | ~45 min |
| C | More real pose-change pairs for texture | P27(a) | texture | 3 | ~80 min |
| D | Discriminator strength for texture | P22, P27(b) | texture | 4 | ~80 min per setting |
| E | 256 px resolution | P27(c) | both | 5 | ~3-4 h (several sessions) |
| F | Exemplar-parsing ablation | D18 | texture | 6 | ~80 min |
| G | Merge top/outer in parser output | P11 | Stage A | 7 | ~15 min (no training) |
| H | Garment-only references (product photos) | P04 | all | 8 (outside DFMM) | new dataset + full retraining |

---

## A. DensePose body representation

* **Research question:** does a body map that does not trace the clothing outline make the warp follow the *input*
  garment's shape better?
* **Hypothesis:** SCHP-Pascal maps include the clothing outline (IoU with the clothed silhouette 0.94-0.97; long dress
  0.39), which teaches a "copy the target's outline" shortcut. DensePose (body15, clothing-independent) removes it, so
  outer/skirt/dress IoU and swap area agreement rise.
* **Baseline:** B0 warp.
* **Changed variable:** `segmentation.body_backend=annotation_densepose`, `labels.body_label_set=body15`
  (`data.require_densepose=true`; images without DensePose drop out, so the split size changes and is recorded).
* **Fixed variables:** everything else, including 10k steps and body jitter.
* **Dataset / split:** DFMM, item-disjoint splits restricted to images with DensePose; validation scored on the
  intersection with B0's validation images so both see the same images.
* **Metrics:** primary warp mIoU (7 classes); per-class IoU of outer, skirt, dress; swap class-area agreement;
  end-to-end SSIM/LPIPS with B0's texture.
* **Compute:** DensePose archive download (~5.6 GB into Drive, needs the user's permission), ~45 min warp training.
* **Results to record:** n images with DensePose per split; full validation warp metrics; swap grid; end-to-end table.
* **Decision:** adopt if the common rule holds **and** swap area agreement does not drop. Caveat recorded either way:
  at inference on new photos DensePose needs its own network (Detectron2), which this project does not run yet. If A
  wins, a DensePose inference backend becomes a follow-up task.

## B. Class-weighted cross-entropy for the warp

* **Research question:** are outer (0.56) and skirt (0.63) weak because they are rare?
* **Hypothesis:** weighting CE by inverse class frequency (median-frequency balancing over the training targets) raises
  rare-class IoU without lowering common classes by more than 0.02.
* **Baseline:** B0 warp.
* **Changed variable:** CE class weights (new option `training.warp_class_weights: none | median_freq`).
* **Fixed variables:** everything else.
* **Dataset / split:** DFMM train for frequencies; validation for scoring.
* **Metrics:** per-class IoU, mIoU, swap area agreement.
* **Compute:** ~45 min.
* **Results to record:** class frequencies and weights used; full validation metrics; per-class change vs B0.
* **Decision:** common rule; additionally outer or skirt must gain ≥ 0.03.

## C. More real pose-change pairs for the texture stage

* **Research question:** is texture quality on real pose changes limited by how few real pairs it sees?
* **Hypothesis:** with the filter only 2,514 of 10,160 training targets have a real partner, so most texture steps are
  easy pseudo pairs. Sampling targets *with* a partner more often raises SSIM / lowers LPIPS on real pairs.
* **Baseline:** B0 texture (and B0 warp for end-to-end).
* **Changed variable:** training sampler that draws targets with a real partner with probability 0.8 (new option;
  currently about 20 %: a quarter of the targets have a partner, and a partner is used with probability 0.8).
* **Fixed variables:** pair filter (ΔE ≤ 15), steps (13k), losses and weights.
* **Dataset / split:** DFMM; the 260 real validation pairs.
* **Metrics:** T4 SSIM / L1 / LPIPS (gt_layout), end-to-end SSIM / LPIPS; FID/KID of different-item try-ons on val.
* **Compute:** ~80 min.
* **Results to record:** share of real pairs actually seen; validation table; paired bootstrap vs B0.
* **Decision:** common rule on T4 LPIPS (primary) and SSIM. Watch for over-fitting to the ~2.5k real pairs (training
  real-pair loss falling while validation gets worse); if seen, stop early and record it.

## D. Discriminator strength for the texture stage

* **Research question:** does a stronger adversarial signal make textures sharper?
* **Hypothesis:** the hinge `d` stayed at 1.7-1.9 ("cannot tell") for 13k steps. A stronger D (TTUR lr_D 4e-4 → 8e-4,
  or 2 D steps per G step) lowers `d` and improves LPIPS and FID without hurting SSIM.
* **Baseline:** B0 texture.
* **Changed variable:** one D setting per run (first lr_D, then D steps if needed).
* **Fixed variables:** everything else.
* **Dataset / split:** DFMM; validation.
* **Metrics:** primary LPIPS (T4, T5); FID/KID (T8 on val); SSIM as a guard; training `d` curve.
* **Compute:** ~80 min per setting.
* **Results to record:** `d` and `g_adv` curves; validation table; grids at the same pairs.
* **Decision:** adopt if LPIPS improves per the common rule and SSIM's CI is not entirely below 0. Training must stay
  stable (no `d` collapse to 0 with rising FID).

## E. 256 px resolution

* **Research question:** how much of the gap to the paper's setting is resolution?
* **Hypothesis:** 256 px (the paper's warp size, `configs/extended_training.yaml`) improves texture detail and LPIPS;
  warp mIoU changes little.
* **Baseline:** B0 (128 px). Metrics are computed at each model's own resolution, and **also** with the 256 px output
  downscaled to 128 px so both are compared at the same size.
* **Changed variable:** `data.image_size=[256,256]` and the resolution-dependent sizes the config ties to it.
* **Fixed variables:** data, filter, losses; steps scaled to the same number of epochs.
* **Compute:** largest experiment, ~3-4 h of T4 time over several resumed sessions (resume is exact).
* **Results to record:** time per step, peak memory, validation table at both sizes.
* **Decision:** adopt if the 128 px-downscaled comparison wins per the common rule; otherwise report it as a cost/quality
  trade-off. Run only after A-D, so it starts from the best 128 px settings.

## F. Exemplar-parsing ablation (deviation D18)

* **Research question:** does feeding the exemplar's garment parsing to F_y (our adaptation) help, compared with the
  paper's image-only input?
* **Hypothesis:** semantic labels make the correspondence easier to learn on a small budget; removing them lowers T4
  SSIM/LPIPS.
* **Baseline:** B0 texture.
* **Changed variable:** `model.texture_exemplar_parsing=false`.
* **Fixed variables:** everything else.
* **Metrics:** T4 and T5 SSIM / L1 / LPIPS; visual check of I_S (warped exemplar).
* **Compute:** ~80 min.
* **Results to record:** validation table and I_S grids.
* **Decision:** this is an ablation. It justifies or removes D18 and is reported either way. Keep D18 unless the
  paper's variant is at least as good (CI of the difference includes 0 or favours it).

## G. Merge top/outer in the parser output (P11)

* **Research question:** does a simple fix for SCHP-LIP splitting one garment into "top" and "outer" improve the fully
  automatic try-on?
* **Hypothesis:** relabelling connected top/outer regions to their majority class reduces wrong two-garment inputs;
  T6 (automatic Stage A) moves closer to T5 (manual Stage A).
* **Baseline:** B0 with the raw SCHP-LIP output.
* **Changed variable:** post-processing of the parser output (new option `segmentation.merge_upper_garments`).
* **Fixed variables:** all trained models (no retraining).
* **Metrics:** T1 parser mIoU (top, outer IoU) vs manual labels; T6 SSIM / LPIPS; T5 − T6 gap.
* **Compute:** ~15 min (evaluation only).
* **Decision:** adopt if T1 top/outer IoU and T6 both improve on validation; record the user-photo case (photo 3) as a
  qualitative check.

## H. Garment-only references (product photos)

* **Research question:** can the pipeline transfer a garment from a product photo, not only from a person wearing it?
* **Hypothesis:** with training pairs of (product photo, person wearing it), the warp learns product-to-body deformation;
  DFMM cannot test this (it has no product photos, P04).
* **Baseline:** none on DFMM; the person-worn mode (B0) on the new dataset's person images.
* **Changed variable:** dataset (VITON-HD or DressCode, both research-only licences needing a request) and reference
  mode `garment_only`.
* **Fixed variables:** architecture and losses where possible.
* **Metrics:** paired SSIM / LPIPS on the dataset's test pairs, FID/KID, and the same Phase 7 protocol.
* **Compute:** new dataset download (size reported before any download), full retraining of both stages (several
  sessions).
* **Decision:** a separate project phase. It starts only after the licence is granted and the user approves the download.

---

## Results log

One row per finished run, filled from its `results.json` by code. The paired-difference columns use
`paired_bootstrap_ci` against B0 on the same validation pairs.

| Exp | Run name | Date | Primary metric (val) | Δ vs B0 [95 % CI] | Secondary | Decision |
|---|---|---|---|---|---|---|
| B0 | warp_pf / texture_pf | pending (Phase 7 val run) | | | | baseline |
