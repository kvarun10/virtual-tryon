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
