# Implementation status and verification log

Updated at the end of Phase 3. "Verified" means a check was actually run. The environment is stated
for each check.

## Verified locally (Windows, Python 3.12.2, torch 2.6.0 CPU, no GPU)

| Check | How | Result |
|---|---|---|
| Unit tests | `python -m pytest` (60 tests) | 60 passed |
| SCHP architecture vs the original source | Built the original `AugmentCE2P.ResNet` with a stub ABN, compared `state_dict` names and shapes for 20/18/7 classes, copied weights, compared outputs | 603 identical tensor names and shapes; max relative output difference 2e-5 (consistent with eps in abs(γ)+eps) |
| SCHP preprocessing affine | Compared `_schp_affine` / `_schp_center_scale` with SCHP `get_affine_transform` / `_box2cs` on 5 image sizes × 2 input sizes × fwd/inv | Max difference 0 |
| InPlaceABN semantics | Read `modules/src/inplace_abn_cpu.cpp` and `inplace_abn_cuda.cu` in the SCHP repository | Both use `abs(weight) + eps`; `ABNCompat` is tested against that formula |
| Notebook | Executed every code cell with nbclient (Python 3 kernel, `DRY_RUN=True`, smoke config) | All cells ran without errors |
| Dataset pipeline | Synthetic fixture with injected faults (corrupt JPEG, missing DensePose, orphan annotations) | All faults reported; splits item-disjoint and reproducible |
| SCHP-LIP and SCHP-Pascal real checkpoints (downloaded with user approval, 2026-10-19) | `ensure_weights(..., allow_download=True)` then `scripts/verify_schp_checkpoint.py` | 255.1 / 255.0 MiB; sha256 prefixes `24fa3254ceeb74c8` (LIP) / `3b03d343c39fb069` (Pascal); **strict load OK, 603/603 tensors, no extra keys** |
| Is the abs(γ)+eps rule relevant for real weights? | Counted negative norm-layer gammas in the LIP checkpoint | 3,360 of 57,264 (5.87%) are negative, so plain `BatchNorm2d` would silently give wrong activations in those channels |

## NOT verified (needs your approval, data or a GPU)

| Item | Blocker | How to verify |
|---|---|---|
| Quantitative parser accuracy on real data | Needs DFMM manual parsing (dataset not downloaded) | Notebook Section 7, evaluation cell |
| SCHP-ATR checkpoint | Not downloaded (not approved) | same script with `--weights schp_atr` |
| DFMM folder names, file naming, DensePose channel | Dataset not downloaded | `scripts/prepare_dataset.py --config configs/colab_baseline.yaml --inspect` |
| SCHP-LIP accuracy on DFMM (garment12 mIoU) | Data + weights | Notebook Section 7, evaluation cell |
| SegFormer-clothes backend | Never run (needs a Hugging Face download) | Set `segmentation.clothing_backend=segformer_clothes` |
| Colab-specific code paths (Drive mount, git clone, pip install) | Not run outside Colab | Run the notebook in Colab |
| GPU execution | No GPU available locally | Colab |
| Garment-only preprocessing on real product photos | No real data | See `clothing_only_mode.md` §4 |

## Qualitative parser check on 5 user photos (2026-10-19, CPU)

The user supplied five full-body photos with plain backgrounds (`test_photos/`, gitignored). Both
parsers were run with `scripts/verify_schp_checkpoint.py`, about 6–8 s per image on CPU. Figures are in
`outputs/schp_check/` (gitignored). This was a **visual inspection with no ground truth, so no metrics are reported.**

| Photo | Clothing (SCHP-LIP → garment12) | Body (SCHP-Pascal) |
|---|---|---|
| 1. T-shirt + jeans, frontal | clean top / pants / footwear / face / hair | clean, all 6 parts |
| 2. T-shirt + jeans, side pose, hand on hip | clean | clean, including the bent arm |
| 3. Patterned long-sleeve shirt + trousers | **failure: one shirt split into `outer` (most of it) and a `top` patch** | clean |
| 4. Patterned maxi dress, hands on hips | clean `dress` | **legs almost absent** (hidden under the dress) |
| 5. Jacket over shirt + jeans | clean `outer` + `top` + `pants` (layering handled correctly) | clean |

What this means for Phase 4:

1. **Top/outer confusion** (photo 3): a single garment can get two classes. The warp network would see one
   shirt as two garments. Possible remedies (Experiment A): merge `top` and `outer` into one class for
   single-layer garments, or apply a per-connected-region majority vote.
2. **SCHP-Pascal labels visible parts only** (photo 4): under long dresses B_seg has no leg information,
   so the warp network gets no leg pose. DensePose (`body15`) estimates the body surface and may handle this
   better, but this is untested. It is evidence for the `body15` vs `pascal_person_part` decision (paper_analysis §10.3).
3. **Neck/torso skin:** LIP has no neck class, so the neck becomes background in garment12, while DFMM
   ground truth labels it `skin`. This only affects classes that are not transferred, but parser output and ground-truth maps differ there.
4. The left/right shoe imbalance in photo 2 has no effect, because both map to `footwear`.

## Phase 4: warp network (2026-10-20)

### Real-data Colab run (user-reported, before Phase 4)

T4 GPU (14.6 GB), Python 3.13, PyTorch 2.11+cu130. Section 6a staged exactly the README counts
(44,096 images / 12,701 parsing / 2 keypoint files). Section 6: 0 unparseable names, 0 corrupt
images, 0 bad maps in the 200-file sample, item-disjoint splits 10,160 / 1,271 / 1,270, and 2,189 train items
with ≥2 annotated views. Section 7: Stage A figure plausible. **The SCHP-LIP mIoU line was not
reported, so parser accuracy on DFMM is still unmeasured.** Odd image sizes (165 × 750×915, one
2574×3861) are handled by letterboxing. A full annotation check (`--annotation-sample 0`) has not been run.

### Verified locally (CPU, synthetic fixture)

| Check | Result |
|---|---|
| Paper Fig. 8 shapes at 256 px | body 512×16×16, clothes code 1024×2×2, upsampled clothes 512×16×16, 4 residual blocks |
| Output shapes | `[B, 12, H, W]` for 64/128 px, non-square 128×64, skip modes body/dual/none, 7- and 15-channel bodies |
| Input validation | wrong body-channel count, wrong size and impossible bottlenecks raise clear errors |
| Losses | CE (ignore-aware, fp32), LSGAN / hinge / vanilla GAN all finite; gradients finite for every generator parameter; all-ignore batch gives a differentiable zero |
| Triplets | self (per-class affine) and cross-view (same item, different photo); face/hair/headwear/accessory ignored; deterministic per (seed, epoch, index); swap pairs come from different items |
| Body maps | precompute → zip → delete → restore → load (with a stand-in backend), resumable; missing maps give a clear error |
| **Checkpoint resume** | an interrupted run (stopped mid-epoch) resumed in a new trainer ends with the **same generator weights (atol 1e-6)** as an uninterrupted run, with the adversarial loss active and across an epoch boundary |
| Checkpoint mismatch | resuming with a different body label set raises `CheckpointMismatchError` |
| **Overfit one batch** | CE 2.73 → 0.029 and pixel accuracy 3.6% → **100%** in 250 steps (small model, 64 px, 13 s on CPU) |
| Notebook | all 19 code cells ran in dry-run mode |
| Full test suite | 88 passed |

### NOT verified yet

* **No training on real data has run.** No real-data mIoU or class-fidelity numbers exist.
* SCHP-Pascal body-map precompute on the Colab GPU (speed, total time).
* Full-width model (67 M parameters at 128 px) on a T4: memory, speed, AMP behaviour.
* Data-loader throughput on Colab's 2 CPU cores (each sample decodes three full-resolution PNGs).

### First real-data warp run (Colab T4, 2026-10-20, user-reported)

| Item | Result |
|---|---|
| SCHP-Pascal body-map precompute | 12,701 maps in 31 min 28 s (6.83 img/s), archived to Drive |
| Model | generator 66.6 M, discriminator 2.78 M params; peak 2.01 GB in the 8b overfit check (batch 2) |
| Speed | 0.13–0.16 s/step at batch 8, 128 px, AMP; 1,270 steps/epoch (~3 min). Data loading is not a bottleneck |
| Run | 500 steps, **CE only** (adversarial starts at step 500), pairing `mixed` |
| Train CE | 2.52 → 0.245 |
| Val (1,271 images, first 50 batches) | CE 0.244, pixel acc 0.945, **mIoU 0.426**, swap class_fidelity 0.912 |

This is an early checkpoint, not a result. Qualitative swap-test findings: a top + **skirt** input
became top + **trousers** on a target with separated legs (the silhouette shortcut predicted in
`body_representation.md`), and an **outer** jacket became a **top**. Common classes (top, trousers,
footwear, skin) are placed plausibly. `class_fidelity` scored the skirt case as correct because the
input also contained a few trouser pixels, so the stricter `class_area_agreement` was added afterwards.
Per-class IoU was recorded but not printed in this run.

### Warp run to 10,000 steps (Colab T4, 2026-10-20, user-reported)

Resumed from step 500. GAN active from step 500 (LSGAN, λ=0.1), mixed pairing, 128 px, batch 8, AMP.
Speed 0.24 s/step with the GAN (~40 min for 9,500 steps). The GAN stayed balanced: `d` mostly 0.05–0.3,
`g_adv` 0.1–0.5, no collapse. Validation used the first 50 batches (400 of 1,271 images).

**Metric bug found in this run (fixed, metric v2):** the reported mIoU averaged every class that appeared
in the ground truth *or the prediction*. Classes that are never trained (hair, face, headwear, accessory)
entered with IoU 0 as soon as a stray pixel was predicted as one of them. The reported curve peaked at
step 3,000 and then "dropped", but that drop came from the definition, not the model. The values below
are recomputed from the logged per-class IoUs (mean over the 7 trained classes):

| Step | val CE | skin | top | outer | dress | skirt | pants | footwear | **mIoU (7, recomputed)** | reported (v1) | swap area agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 0.216 | 0.50 | 0.61 | 0.00 | 0.63 | 0.11 | 0.71 | 0.57 | 0.447 | 0.449 | 0.743 |
| 3,000 | 0.189 | 0.58 | 0.75 | 0.58 | 0.69 | 0.65 | 0.76 | 0.59 | 0.657 | 0.656 | 0.821 |
| 5,000 | 0.174 | 0.63 | 0.77 | 0.60 | 0.67 | 0.69 | 0.79 | 0.61 | 0.680 | 0.475 | 0.808 |
| 10,000 | **0.172** | 0.65 | 0.79 | 0.60 | 0.71 | 0.67 | 0.80 | 0.66 | **0.697** | 0.443 | **0.842** |

Consequences: `best.pt` was selected with the v1 metric and is the **step-3,000** model; `last.pt`
(step 10,000) is better on every trained class except outer (equal) and skirt (−0.02 vs step 5,000 only).
Checkpoints from v1 have their best/early-stop counters reset on resume. Official v2 numbers on the
**whole** val split still have to be produced with `evaluate_checkpoint` (notebook Section 10).

Qualitative (validation grid, step 10,000): common outfits closely match the ground truth. In the swap
test the jacket now stays a jacket and the dress stays a dress on a trousers-wearing target. The skirt
is now kept, but is drawn split along the target's legs, and long garments are clipped short:
a remaining partial silhouette leak from SCHP-Pascal body maps.

### Official warp baseline (full validation split, metric v2, 2026-10-21)

Run in Colab at commit `2c60c63` with `evaluate_checkpoint`, on all 1,271 validation images (item-disjoint split, seed 0).
128 px, SCHP-Pascal body maps, mixed pairing, LSGAN λ=0.1.

| Checkpoint | mIoU (7 trained classes) | skin | top | outer | dress | skirt | pants | footwear | swap area agreement | swap class fidelity | val CE | pixel acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step 3,000 (`best.pt`, chosen by the v1 metric) | 0.640 | 0.61 | 0.72 | 0.53 | 0.75 | 0.55 | 0.75 | 0.57 | 0.821 | 0.968 | 0.190 | 0.967 |
| **step 10,000 (`last.pt`) - baseline** | **0.689** | 0.67 | 0.76 | 0.56 | 0.78 | 0.63 | 0.79 | 0.63 | **0.842** | 0.958 | **0.174** | 0.972 |

These are our own measurements with our own protocol. They are **not comparable** with the paper's Table 2
(label-to-image mIoU), which measures something else. The test split has not been used.
Known weaknesses: outer and skirt (rare classes, different shape from the target's own clothing);
long garments are drawn too short or split along the legs on targets with visible legs (silhouette leak).

## Phase 5: texture stage (2026-10-21)

### Verified locally (CPU, synthetic fixture, untrained VGG)

| Check | Result |
|---|---|
| Correspondence maths (Eqs. 5–6) | with position-identifying features, I_S equals the permuted exemplar; attention rows sum to 1; a perfect bijection gives a perfect cycle |
| α behaviour | α = 0.001 → uniform average (blur); α = 1000 → exact self-match |
| FP32 correlation | attention and I_S stay float32 under bfloat16 autocast |
| Shapes | 64/128 px, exemplar parsing on/off; 128 px: G 27.4 M (nf 32), 256 px: G 97.2 M (nf 64); D 5.6 M (2 scales) |
| Correlation memory | 32×32: 4 MiB per sample for M (FP32); 64×64: 64 MiB |
| Losses | hinge (multi-scale), feature matching, VGG, warp L1, cycle, domain all finite; **every generator parameter receives a finite gradient** |
| VGG weights | download gated (`training.allow_vgg_download`), cached in `pretrained/torch` |
| SSIM | identical images = 1, symmetric, noise < 0.2 |
| Exact resume | interrupted + resumed texture run matches the uninterrupted run (atol 1e-5) across an epoch boundary |
| Overfit one batch | L1 0.95 → 0.008, SSIM 0.62 → **0.994** in 300 steps (64 px, CPU, 125 s) |
| Notebook | all 24 code cells run in dry-run mode (offline: untrained VGG) |

### NOT verified yet

* No texture training on real data; no real SSIM numbers; T4 speed and memory unknown for the 27 M generator.
* VGG-19 ImageNet weights not downloaded yet (needs `ALLOW_VGG_DOWNLOAD=True` once, ~548 MB).
* End-to-end try-on (warp output → texture) is Phase 6.

### Phase 5 follow-up (2026-10-21, after the first 2,000-step texture run)

Real-data texture run: 0.34–0.36 s/step on T4; val SSIM 0.868 (step 1,000) → 0.876 (step 2,000), whole image,
400 cross-view pairs. Two problems were found (docs/problems_log.md P20, P21) and addressed in code:
* validation now reports person-region SSIM/L1 and two baselines (copy exemplar; warped I_S only); `best.pt`
  is selected by person-region SSIM (texture metric v2). Real numbers need a Colab re-evaluation;
* pair-consistency analysis + optional `data.pair_filter` (off by default). The DFMM statistics need the Colab run (Section 6b).

## Phase 6: end-to-end try-on (2026-10-21)

Implemented: `src/inference/tryon.py` (load trained generators with architecture check; Stage A, then warp, then the
paper's Sec. 4.2 face/hair labels, then texture, then face/hair pixel paste), `try_on(..., models=...)` with
saved per-stage PNGs, `src/evaluation/end_to_end.py` (paired evaluation on garment + skin pixels with the
ground-truth-layout upper bound and the copy-exemplar lower bound, warp mIoU on the same pairs, swap grid),
notebook Section 11, time-budgeted training (`fit(time_limit_min=...)`), named runs (`training.*_run_name`),
one-folder export of session results (Section 14), and a CPU-runtime option (`REQUIRE_GPU=False`).

Verified locally (CPU, fixture, tiny models trained for 2 steps): models load in eval mode, the full `try_on`
writes all stage outputs, untrained classes never appear in C'_seg, the target's face pixels are copied exactly
into the result, missing/mismatched checkpoints give clear errors, end-to-end metrics are finite, and the
notebook runs all 28 code cells with training and Phase 6 switched on.

Not verified: any end-to-end result on real data (needs the next Colab session).

Real-data texture evaluation (step 2,000, full val, 1,271 pairs, commit a590a0d): person-region SSIM 0.540 vs
baselines 0.384 (warped I_S only) and 0.125 (copy exemplar); whole-image SSIM 0.883.
Pair analysis: 16,029 same-item pairs, 72.8% identical garment classes, median max ΔE 49 (problems_log P21).
