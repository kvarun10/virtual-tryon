# Problems log: issues found, causes, and what was done

A running record of every problem met so far, for the project report ("challenges and solutions").
Each entry gives the symptom, the evidence, the root cause, the action taken, the status, and where the
evidence lives. Status: **fixed** (verified), **mitigated** (reduced, still measurable), **open**
(known, planned), **by design** (accepted limitation, documented).

Evidence paths refer to `report/` (numbers, charts), `docs/`, and `tests/`. Dates are 2026.

## Summary

| ID | Date | Area | Problem (short) | Status |
|---|---|---|---|---|
| P01 | 10-19 | Paper | Original components (LIP-JppNet, NBF) cannot run on Colab | fixed (substitutes) |
| P02 | 10-19 | Paper | Paper is internally inconsistent / under-specified in several places | by design (documented) |
| P03 | 10-19 | Stage A | SCHP needs a CUDA extension; plain BatchNorm would silently be wrong | fixed |
| P04 | 10-19 | Data | DFMM has no product (garment-only) images | by design / open |
| P05 | 10-19 | Repo | Project folder sat inside a git repo of the whole home folder | fixed |
| P06 | 10-19 | Repo | `.gitignore` rule `data/` hid the whole `src/data/` package | fixed |
| P07 | 10-19 | Config | Default body-backend name did not exist (`dfmm_densepose`) | fixed |
| P08 | 10-19 | Data | Dataset prep rejected all images without DensePose | fixed |
| P09 | 10-19 | Storage | Reading 44k small files from the Drive mount is slow | fixed (design) |
| P10 | 10-19 | Workflow | Colab could not get the code; old notebook copies kept being opened | fixed |
| P11 | 10-20 | Stage A | SCHP-LIP split one patterned shirt into two garment classes | open |
| P12 | 10-20 | Stage B | SCHP-Pascal body maps trace the *clothed* outline (silhouette leak) | mitigated |
| P13 | 10-20 | Metrics | Lenient swap metric missed "skirt became trousers" | fixed |
| P14 | 10-20 | Training | Resume crashed on GPU ("RNG state must be a torch.ByteTensor") | fixed |
| P15 | 10-20 | Metrics | mIoU "dropped" 0.66 → 0.44 while every class improved | fixed |
| P16 | 10-21 | Workflow | Re-used Colab runtime kept running old code after `git pull` | fixed |
| P17 | 10-21 | Docs | Manual arithmetic errors in a recomputed results table | fixed |
| P18 | 10-21 | Repo | Word lock file nearly committed; doc locked while open in Word | fixed |
| P19 | 10-21 | Stage B | Warp weakest on outer (0.56) and skirt (0.63); long garments drawn short | open |
| P20 | 10-21 | Metrics | Texture SSIM 0.876 is hard to judge (white background inflates it) | fixed (needs re-run) |
| P21 | 10-21 | Data | "Same item ID" does not mean "same outfit" (82% of pairs cross photo sets) | fixed (filter; swap agreement 0.842 → 0.879) |
| P22 | 10-21 | Stage C | Texture discriminator barely discriminates yet (hinge d ≈ 2) | open (monitor) |
| P23 | 10-21 | Stage C | Generated faces belong to the exemplar person | by design (Phase 6) |
| P24 | 10-21 | Tooling | CPU dry run would have needed a 548 MB VGG download | fixed |
| P25 | 10-21 | Data | 165 images are 750×915, one is 2574×3861; full annotation check not run | open (check) |
| P26 | 10-21 | Metrics | Texture validation silently mixed easy "pseudo" pairs with real pose-change pairs | fixed (metric v3) |
| P27 | 10-21 | Stage C | 6.5× more texture training gave only +0.01 SSIM on real pose-change pairs | open (analysis) |

---

## Details

### P01 – Original pre-treatment networks unusable (Phase 1)
* **Symptom:** the paper uses LIP-JppNet (clothes) and Neural Body Fitting (body). Both are TensorFlow 1.x; NBF also needs SMPL (registration, non-commercial licence).
* **Evidence:** `docs/paper_analysis.md` §4 (repository checks).
* **Cause:** TF 1.x wheels do not exist for Colab's Python ≥ 3.10.
* **Action:** SCHP trained on LIP (same 20 labels) for clothes; SCHP Pascal-Person-Part (and DensePose as an option) for body.
* **Status:** fixed with documented substitutes (`docs/implementation_deviations.md` D1, D3).

### P02 – Paper inconsistencies and gaps (Phase 1)
* **Symptom / evidence (paper sections in brackets):** CE over "c = 1..3" classes (Eq. 2); Eq. 3 written without logs, λ not given; Fig. 4 caption appears swapped; Eq. 6 warps features but calls the result an image; translation network described both as "constant code z" and "VAE encoder"; LPIPS marked "higher is better"; the same FID (8.60) reported for two different tasks; no train/test split, no batch size or epochs.
* **Action:** every ambiguity documented with our interpretation and source tag ([P]/[R]/[E]) in `docs/paper_analysis.md` §2 and §8; CoCosNet used as the reference for missing texture-stage details.
* **Status:** by design: numbers are never compared with the paper's without a matching protocol (`report/README.md` §4).

### P03 – SCHP's InPlaceABN layer (Phase 3)
* **Symptom:** SCHP needs a JIT-compiled CUDA extension. Replacing it with `nn.BatchNorm2d` loads without error but changes outputs.
* **Evidence:** the InPlaceABN kernel scales by abs(γ) + eps; **5.87%** (3,360 / 57,264) of the LIP checkpoint's gammas are negative (`report/data/schp_verification.json`).
* **Action:** pure-PyTorch `ABNCompat` with the abs(γ) + eps rule; architecture checked against the original source (603/603 tensors, output difference 2e-5); real checkpoints load strictly.
* **Status:** fixed (`tests/test_segmentation.py`).

### P04 – No product images for garment-only references (Phase 2–3)
* **Symptom:** the user also wants try-on from product photos; DFMM only contains people.
* **Action:** experimental preprocessing only (plain-background removal + user-supplied category), marked `validated=False`; candidate datasets documented (VITON-HD: CC BY-NC 4.0, tops only; DressCode: institutional request).
* **Status:** by design for now / open (`docs/clothing_only_mode.md`).

### P05 – Home folder was a git repository (Phase 2)
* **Symptom:** `git rev-parse --show-toplevel` returned `C:\Users\HP ENVY`; `git add .` would have put the project (and pushing could have exposed personal files) into that repository.
* **Action:** separate `git init` in the project folder before the first commit.
* **Status:** fixed.

### P06 – `.gitignore` hid the data package (Phase 2)
* **Symptom:** the rule `data/` (meant for dataset folders) also matched `src/data/`; the first commit would have shipped without the data pipeline, and Colab would have failed on import.
* **Evidence:** caught by checking `git status --ignored` before committing.
* **Action:** root-anchored rules (`/data/`, `/datasets/`, ...); test of a clean clone after every commit.
* **Status:** fixed.

### P07 – Invalid default body backend name (Phase 3)
* **Symptom:** config default `dfmm_densepose` was not a registered backend; the smoke config overrode it, so tests did not catch it.
* **Action:** corrected to `annotation_densepose`; new test that every shipped config names valid backends.
* **Status:** fixed (`tests/test_config.py`).

### P08 – DensePose was mandatory (Phase 3)
* **Symptom:** with the approved "images + parsing" download, dataset preparation would have rejected every image.
* **Action:** `data.require_densepose` option; clear error where body maps are actually needed.
* **Status:** fixed (`tests/test_fetch.py`).

### P09 – Small files on Google Drive (Phase 3)
* **Symptom:** training directly from 44k files on the Drive mount would be I/O-bound.
* **Action:** zip archives stay in Drive (one copy); each session extracts them to Colab's local disk; body maps follow the same pattern.
* **Status:** fixed: data loading is not a bottleneck (0.13–0.36 s/step, `report/data/compute_and_timing.csv`).

### P10 – Getting code and notebook into Colab (Phase 3–4)
* **Symptom:** the code was not on GitHub; later, stale notebook copies in Drive kept being opened ("I don't see FETCH_DFMM").
* **Cause:** a notebook uploaded to Drive is a frozen copy; `git pull` updates only the code.
* **Action:** private GitHub repo + Colab secret `GITHUB_TOKEN`; open the notebook directly from GitHub.
* **Status:** fixed.

### P11 – Parser splits one garment into two classes (Phase 3)
* **Symptom:** SCHP-LIP labelled most of a patterned long-sleeve shirt "outer" and a patch "top".
* **Evidence:** `report/data/parser_qualitative_user_photos.csv` (photo 3), `report/private/user_photos/`.
* **Impact:** the warp network would treat one garment as two at try-on time.
* **Planned:** per-region majority vote or a merged top/outer class (Experiment A). Training uses manual DFMM labels, so it is unaffected.
* **Status:** open.

### P12 – Silhouette leak of SCHP-Pascal body maps (Phase 4)
* **Symptom:** body maps follow the clothed outline (jacket widens the torso; long dress removes the legs).
* **Evidence:** IoU with the clothed silhouette 0.94–0.97, long dress 0.39 (`report/data/body_silhouette_leak.csv`, `report/figures/body_map_silhouette_leak.png`); swap test: skirt drawn split along the target's legs.
* **Cause:** Pascal-Person-Part annotations label visible body regions including clothing; training targets always wear their own garment, which invites a "copy the outline" shortcut.
* **Action:** random per-part dilation/erosion of B_seg during training; swap-test metrics; DensePose comparison planned (Experiment A).
* **Status:** mitigated: swap area agreement rose 0.74 → 0.84 during training, but long garments are still clipped (`docs/body_representation.md`).

### P13 – Lenient swap metric (Phase 4)
* **Symptom:** `class_fidelity` scored a skirt-to-trousers failure as correct because the input also had a few trouser pixels.
* **Action:** added `class_area_agreement` (histogram intersection of garment-class areas), with a regression test built from that exact case.
* **Status:** fixed (`tests/test_warp.py::test_class_area_agreement_catches_skirt_to_pants`).

### P14 – Resume crash on GPU (Phase 4)
* **Symptom:** `TypeError: RNG state must be a torch.ByteTensor` when continuing training in Colab.
* **Cause:** checkpoints were loaded with `map_location=cuda`, which also moved the saved RNG state to the GPU; local CPU tests could not see this.
* **Action:** load checkpoints on CPU; RNG restore converts to CPU uint8 itself; regression tests.
* **Status:** fixed: verified by the 10k-step Colab run resuming from step 500.

### P15 – mIoU metric bug (Phase 4)
* **Symptom:** reported validation mIoU fell from 0.656 (step 3,000) to 0.443 (step 10,000) while every garment class improved.
* **Evidence:** sum of the 7 trained-class IoUs ÷ (7–11) reproduces every reported value within 0.001 (`report/data/warp_val_log_10k.csv`, `report/figures/warp_miou_metric_v1_vs_v2.png`).
* **Cause:** untrained classes (hair, face, headwear, accessory) entered the mean with IoU 0 once a stray pixel was predicted as one of them.
* **Action:** metric v2 (mean over the 7 trained classes), untrained classes masked out of predictions, `best.pt` counters reset for old checkpoints, `evaluate_checkpoint` on the full validation split.
* **Consequence:** `best.pt` (step 3,000, chosen by the buggy metric) scored 0.640; the real best is step 10,000 at **0.689** (full val, `report/data/warp_full_val_eval.csv`).
* **Status:** fixed.

### P16 – Stale code in a re-used Colab runtime (Phase 4)
* **Symptom:** `'WarpTrainer' object has no attribute 'evaluate_checkpoint'` although GitHub had the new code.
* **Cause:** Python keeps already-imported modules; `git pull` changed the files on disk but not the loaded classes.
* **Action:** Section 4 discards already-imported `src.*` modules after pulling and prints the commit in use.
* **Status:** fixed (verified: Colab printed `commit: 2c60c63`, then `bc51998`).

### P17 – Arithmetic errors in a hand-made table (report building)
* **Symptom:** two recomputed mIoU values (steps 2,000 and 6,000) were wrong when first typed in.
* **Action:** all derived numbers are now computed by code from the stored CSVs and checked against the logged values.
* **Status:** fixed (`report/data/warp_val_log_10k.csv`).

### P18 – Word lock file and locked document (Phase 5)
* **Symptom:** `docs/~$per_analysis.md` was staged for commit; a doc update failed with "used by another process".
* **Cause:** `paper_analysis.md` was open in Microsoft Word.
* **Action:** `~$*` added to `.gitignore`; the update was applied after Word released the file.
* **Status:** fixed.

### P19 – Weak warp classes and short long garments (Phase 4)
* **Evidence:** full-val IoU outer 0.56, skirt 0.63 (others 0.63–0.79); swap grids show dresses and skirts drawn too short.
* **Probable causes:** class rarity, the silhouette leak (P12), and inconsistent cross-view pairs (P21).
* **Planned (one change at a time, fixed protocol):** pair filter (P21), class-weighted CE, DensePose body maps, longer training.
* **Status:** open.

### P20 – Texture SSIM hard to interpret (Phase 5)
* **Symptom:** first texture run reported SSIM 0.876 without context.
* **Cause:** DeepFashion photos are mostly white background; matching blank areas scores high SSIM for free.
* **Evidence:** on the fixture, copying the exemplar unchanged scores 0.69 on the whole image but 0.43 on the person region; unit test `tests/test_texture.py::test_masked_ssim_ignores_background`.
* **Action:** validation now reports person-region SSIM/L1 plus two baselines on the same pairs (copy the exemplar unchanged; the warped exemplar I_S alone). `best.pt` is selected by person-region SSIM (metric v2).
* **Result (full val, 1,271 pairs, step 2,000):** whole image 0.883 vs **person region 0.540**. Baselines on the
  person region: copy exemplar 0.125, warped I_S alone 0.384. The generator adds +0.16 SSIM over I_S; L1 is
  barely better (0.157 vs 0.164), so it mainly improves structure, not colour, so far
  (`report/data/texture_eval_step2000_fullval.csv`).
* **Status:** fixed (measured).

### P21 – Same item ID ≠ same outfit (Phase 5)
* **Symptom:** a validation pair showed a plaid shirt + jeans as the exemplar and a green sweater + black jeans as the "same item" target (`report/private/colab/texture_step2000_val_grid_part1.png`, rows 2–3).
* **Cause:** a DeepFashion item ID names one product (here probably the beanie or bag); the rest of the outfit can change between photos.
* **Impact:** such pairs ask both networks to produce garments that are not in their input (noise in cross-view supervision for warp and texture).
* **Action:** pair-consistency analysis (identical garment classes + CIELAB colour difference per garment) for all same-item pairs; optional `data.pair_filter` (off by default so baselines stay comparable); tested on a fixture with deliberately changed outfits.
* **Measurement (16,029 pairs, 2,701 items):** 72.8% have identical garment classes; median max ΔE 49;
  only 15.8% pass "same classes and ΔE ≤ 15" (`report/data/pair_quality_summary_dfmm.json`). That is
  more inconsistency than outfit changes alone explain. Hypotheses under test: (a) the photo-set number in the
  file name (`-01_`, `-02_`) marks different shoots or colourways; (b) the max-over-garments ΔE is dominated
  by small regions such as shoes.
* **Breakdown (offline, from the downloaded pair table):** the photo-set number explains it. Same-set pairs
  (2,961): median max ΔE **3.8**. Different-set pairs (13,068, 82%): median **57.5**. 1,654 of 2,701 items have
  more than one photo set, i.e. one product in several colourways or outfits. A third of same-set pairs still differ
  in garment classes, mostly because shoes, top or trousers are cropped or hidden in close-up, back and side views
  (`report/data/pair_quality_breakdown.json`, `report/figures/pair_delta_e_by_photo_set.png`).
* **Decision:** filter `classes_color`, max ΔE ≤ **15**, which lies between the 95th (11.7) and 99th (17.1) percentile
  of trusted same-outfit pairs. It keeps 2,537 of 16,029 pairs (15.8%); images with a usable partner drop 8,500 → 3,131.
  The rest train with self-supervised (warp) or pseudo (texture) pairs. Config: `configs/colab_pairfilter.yaml`.
  Evaluated against the unfiltered baseline on the same filtered validation pairs.
* **Result (Colab session A, 2026-10-21, same 1,271 validation images, filter on for both):** validation mIoU unchanged
  (filtered 0.715 vs baseline 0.716, every class within ±0.01), but the **swap test improved: area agreement
  0.842 → 0.879, class fidelity 0.958 → 0.985**. The filtered warp follows the *input* garment more faithfully in
  the real try-on situation, which is the behaviour the contradictory pairs were damaging
  (`report/data/warp_pairfilter_vs_baseline_fullval.csv`).
* **Status:** fixed; `warp_pf` is the new warp baseline.

### P22 – Weak texture discriminator (Phase 5)
* **Evidence:** hinge `d` stayed at 1.7–2.1 over 2,000 steps (2.0 = cannot tell real from fake); `g_adv` 0–0.7 (`report/data/texture_train_log_2k.csv`).
* **Plan:** watch over a longer run; if `d` does not fall, tune the discriminator learning rate, number of D steps, or the feature-matching weight.
* **Status:** open (monitor).

### P23 – Faces come from the exemplar (Phase 5)
* **Cause:** training pairs show the same model, so the generator learns to copy the exemplar's face; at try-on the exemplar is a different person.
* **Plan:** Phase 6 pastes the target person's face and hair back (paper Sec. 4.2 identity step).
* **Status:** by design / Phase 6.

### P24 – Offline dry run vs VGG weights (Phase 5)
* **Symptom:** the CPU dry run would have stopped at the 548 MB VGG download.
* **Action:** the smoke config uses an untrained VGG (`texture_vgg_init: random`, values meaningless) and small model widths; real runs download VGG once with explicit permission.
* **Status:** fixed.

### P25 – Unusual image sizes; partial annotation check (Phase 3)
* **Evidence:** 165 images are 750×915, one is 2574×3861 (`report/data/dataset_stats.json`); only 200 annotation files were size-checked.
* **Action so far:** letterboxing handles any size; datasets raise an error on image/label size mismatch.
* **Plan:** run `scripts/prepare_dataset.py --annotation-sample 0` once in Colab.
* **Status:** open (check).

### P26 – Texture validation mixed pair types (Phase 5, session B)
* **Symptom:** session B printed "1271 cross-view pairs" and person-region SSIM rising to 0.677, while Section 11 found
  only 260 real cross-view validation pairs after the pair filter.
* **Cause:** validation iterated over every validation photo; a photo without a (filtered) same-outfit partner was
  scored as a pseudo pair (its own rotated/flipped copy as exemplar), which is a much easier task than a pose change.
  The filter shifted the mix from mostly real to mostly pseudo pairs, so v2 numbers across runs are not comparable.
* **Clean comparison (same 260 real pairs, texture given the ground-truth layout):** baseline texture (2k steps)
  0.566 SSIM / 0.0996 L1 vs `texture_pf` (13k steps, filtered) **0.576 / 0.0926**
  (`report/data/texture_clean_comparison_260pairs.csv`).
* **Action:** texture validation now uses real cross-view pairs only and labels the pair type (metric v3, best.pt
  counters reset); regression test.
* **Status:** fixed.

### P27 – Texture gains little from longer training on real pose changes (Phase 5, session B)
* **Evidence:** +0.010 SSIM / −7% L1 for 6.5× the steps on the 260 real pairs; end-to-end only 0.437 → 0.442.
  The discriminator stayed near "cannot tell" (hinge d ≈ 1.7–1.9 for all 13k steps; P22).
* **Likely causes (to test one at a time):** (a) with the filter, only 2,514 of 10,160 training targets have a real
  partner, so most texture training is on easy pseudo pairs; (b) the weak adversarial signal leaves outputs blurry;
  (c) 128 px resolution.
* **Status:** open. Warp errors still cost more (end-to-end 0.442 vs 0.576 with the ground-truth layout).

---

## Lessons used throughout

* Verify on the real environment early: three bugs (P07, P14, P16) only appeared outside the local CPU tests.
* Put a baseline next to every metric (P15, P20) and compute derived numbers in code (P17).
* Check what a dataset's identifiers really mean before using them for supervision (P21).
* When a data filter changes, re-check what the validation set now contains (P26).
* Keep every fix behind a regression test, so it cannot silently come back.
