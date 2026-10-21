# Report evidence pack

Material for the project report: measured numbers, charts and figures, each linked to the technical
decision it supports. Every value here was actually measured. The source and date of each value are
given, and paper values are kept separate from our own results.

```
report/
├── README.md            this index
├── make_charts.py       rebuilds every chart in figures/ from data/  (python report/make_charts.py)
├── data/                measured numbers (CSV / JSON / YAML), safe to commit
├── figures/             charts + synthetic-only figures, safe to commit
└── private/             real photos (DeepFashion-derived, your own test photos), NOT in git
```

**Privacy / licence:** `private/` is git-ignored. DeepFashion-MultiModal forbids redistribution and
commercial use of its images and derived data; your test photos are personal. You may still decide to
put selected private figures in an academic report; check your institution's rules and cite the dataset.

Problems met and how they were tackled (P01–P25, with evidence links): `docs/problems_log.md`.

New runs: export the training log with
`python scripts/export_run_logs.py --log <drive>/logs/<experiment>/<stage>_train.jsonl --out report/data --name <run>`
and rerun `make_charts.py`.

---

## 1. Evidence by decision

### D-A. Re-implementing SCHP (instead of LIP-JppNet / InPlaceABN)
| Evidence | File | Key numbers |
|---|---|---|
| Original LIP-JppNet / NBF are TensorFlow 1.x and cannot run on Colab | `docs/paper_analysis.md` §4 | — |
| Our network = original architecture | `data/schp_verification.json` | 603/603 identical tensors; max rel. output diff 2.1e-5 |
| Preprocessing identical to the original | same | affine matrices: max diff 0.0 |
| Why plain BatchNorm would be wrong | same | **5.87%** (3,360 / 57,264) of LIP gammas are negative; InPlaceABN uses abs(γ)+eps |
| Real checkpoints load | same | strict 603/603, sha256 prefixes recorded |

### D-B. Body representation: SCHP-Pascal (7 parts) and its silhouette leak
| Evidence | File | Key numbers |
|---|---|---|
| SCHP-Pascal outlines the **clothed** person | `data/body_silhouette_leak.csv`, `figures/body_map_silhouette_leak.png` | IoU with clothed silhouette 0.94–0.97; long dress 0.39 (legs lost) |
| Qualitative parser behaviour on 5 photos | `data/parser_qualitative_user_photos.csv`, `private/user_photos/` | shirt split into outer + top (photo 3); legs missing under dress (photo 4) |
| Design response | `docs/body_representation.md` | body-part jitter ±3 px at 128 px, swap-test metrics |

### D-C. Dataset and splits (DeepFashion-MultiModal, item-disjoint)
| Evidence | File | Key numbers |
|---|---|---|
| Dataset integrity and size | `data/dataset_stats.json` | 44,096 images, 12,701 parsed, 0 corrupt, 0 unparseable |
| Leakage-free split | same | 10,160 / 1,271 / 1,270 images, item-disjoint, seed 0 |
| Cross-view supervision is possible | same | 2,189 train items with ≥2 views; 6,812 / 10,160 train triplets with a partner |
| Synthetic test fixture (software tests only) | `figures/synthetic_fixture_rgb_garment12_body15.png`, `figures/stageA_pipeline_synthetic.png` | — |

### D-D. Warp network: architecture follows paper Fig. 8; training works
| Evidence | File | Key numbers |
|---|---|---|
| Paper shapes reproduced | `tests/test_warp.py::test_paper_feature_shapes_at_256` | body 512×16×16, clothes code 1024×2×2, 4 res blocks |
| Network can learn (sanity) | `data/warp_overfit_one_batch_cpu.csv`, `figures/warp_overfit_one_batch.png` | CE 2.73 → 0.029, pixel acc 3.6% → 100% in 250 steps |
| Training curves | `data/warp_train_log_10k.csv`, `figures/warp_training_losses.png` | CE 2.52 → ~0.18; GAN balanced (d 0.03–0.6, mostly 0.05–0.3) |
| Exact resume after disconnect | `tests/test_warp.py::test_resume_matches_uninterrupted_run` | resumed weights = uninterrupted run (atol 1e-6) |
| Compute cost | `data/compute_and_timing.csv` | 0.24 s/step with GAN; body-map precompute 31.5 min; 2.0 GB peak |

### D-E. Metric design: mIoU over trained classes (bug found and fixed)
| Evidence | File | Key numbers |
|---|---|---|
| Reported mIoU "dropped" because untrained classes entered the mean with IoU 0 | `data/warp_val_log_10k.csv` (`n_classes_in_v1_mean`), `figures/warp_miou_metric_v1_vs_v2.png` | sum of 7 class IoUs ÷ (7–11) reproduces every reported value within 0.001 |
| Corrected curve keeps rising | same | 0.447 → **0.697** (400 val images) |
| Strict swap metric needed | `figures/warp_swap_test_metrics.png` | lenient fidelity ~0.95 accepted skirt→trousers; area agreement 0.74 → 0.84 |

### D-F. Official warp baseline and checkpoint choice
| Evidence | File | Key numbers |
|---|---|---|
| Full validation split (1,271 images) | `data/warp_full_val_eval.csv`, `figures/warp_full_val_per_class_best_vs_last.png` | step 3,000: 0.640; **step 10,000: 0.689**, better on all 7 classes |
| Per-class progress | `figures/warp_per_class_iou_over_steps.png` | weakest: outer 0.56, skirt 0.63 |
| Visual results | `private/colab/warp_eval_last_step10000_fullval_grid.png` and step-500 / 10k grids | see "How to read the grids" below |

### D-G. Texture stage (Phase 5): alignment + SPADE
| Evidence | File | Key numbers |
|---|---|---|
| Design and paper/CoCosNet/own split | `docs/texture_stage.md`, `docs/implementation_deviations.md` D16–D22 | — |
| Correspondence behaves as Eqs. 5–6 describe | `tests/test_texture.py::test_correspondence_picks_matching_positions`, `::test_small_alpha_blurs_and_large_alpha_sharpens` | exact permutation recovery; α controls sharpness |
| Numerically safe correlation | `tests/test_texture.py::test_correlation_runs_in_fp32_under_autocast` | float32 under autocast |
| Generator can learn | `data/texture_overfit_one_batch_cpu.csv`, `figures/texture_overfit_one_batch.png` | SSIM 0.62 → 0.994 in 300 steps |
| Model size / memory | `docs/implementation_status.md` (Phase 5) | 27.4 M (128 px), 97.2 M (256 px); M = 4 MiB / 64 MiB per sample |

### D-H. Texture evaluation needs baselines; same-item pairs are noisy
| Evidence | File | Key numbers |
|---|---|---|
| Whole-image SSIM is inflated by background | `data/texture_eval_step2000_fullval.csv` | whole 0.883 vs person region 0.540 |
| The generator adds value over simpler methods | same | person-region SSIM: copy 0.125, warped I_S 0.384, full texture stage 0.540 |
| Same item ID ≠ same outfit | `data/pair_quality_summary_dfmm.json` | 16,029 pairs: 72.8% same garment classes; median max ΔE 49; 15.8% pass classes + ΔE ≤ 15 |
| Cause: photo sets (`-01_` vs `-02_`) | `data/pair_quality_breakdown.json`, `data/pair_delta_e_hist_by_photo_set.csv`, `figures/pair_delta_e_by_photo_set.png` | same set median ΔE 3.8 vs different sets 57.5 |
| Threshold ΔE 15 from the data | same | 95th / 99th percentile of trusted same-outfit pairs: 11.7 / 17.1 |
| Visual evidence (rows 2–3: different outfits under one item ID) | `private/colab/texture_step2000_val_grid_part1.png` | — |

### D-I. Pair filter effect and first end-to-end try-on (Colab session A)
| Evidence | File | Key numbers |
|---|---|---|
| Filter keeps accuracy, improves try-on faithfulness | `data/warp_pairfilter_vs_baseline_fullval.csv` | mIoU 0.715 vs 0.716; swap area agreement 0.879 vs 0.842 |
| End-to-end quality and where it is lost | `data/e2e_sessionA.csv` | full pipeline 0.437 SSIM; GT-layout upper bound 0.566; copy 0.195 |
| Visual results | `private/colab/e2e_sessionA_val_grid.png`, `e2e_sessionA_swap_grid_part*.png` | — |

### D-J. Texture with the pair filter (Colab session B) and a validation flaw
| Evidence | File | Key numbers |
|---|---|---|
| Clean texture comparison on 260 real pose-change pairs | `data/texture_clean_comparison_260pairs.csv` | 0.566 → 0.576 SSIM; e2e 0.437 → 0.442 |
| Why the training-time curve looked better than reality | `data/texture_pf_val_log_13k_metric_v2.csv`, `docs/problems_log.md` P26 | v2 mixed easy pseudo pairs into validation |
| Curves and grids | `figures/colab_texture_pf_13k_curves.png`, `private/colab/texture_pf_13k_*`, `private/colab/e2e_sessionB_*` | — |

---

## 2. Thresholds and settings used (and why)

Full resolved config: `data/config_baseline_resolved.yaml`. It was generated from the current code, so it also
lists Phase-5 texture fields that the warp runs did not use. [P] = paper, [R] = CoCosNet/SwapNet, [E] = ours.

| Setting | Value | Source | Reason |
|---|---|---|---|
| Image size | 128 × 128 (letterbox, white pad) | [E] | paper uses 256; 128 fits the Colab budget; letterbox avoids squashing 750×1101 photos |
| Split | 80/10/10, item-disjoint, seed 0 | [E] | no garment appears in two splits |
| Clothing label set | garment12 | [E] | the paper's "3 classes" is inconsistent; DFMM / LIP / ATR all map to it |
| Transfer classes | top, outer, dress, skirt, pants, footwear | [E] | the paper's figures transfer full outfits |
| Identity classes from target | hair, face | [P] Sec. 4.2 | |
| Warp body levels / clothes bottleneck | 4 levels (16×16 at 256) / 2×2×1024 | [P] Fig. 8 | |
| Warp residual blocks | 4 | [P] Fig. 8 | |
| Warp skips / norm | body-only / InstanceNorm | [E] | paper unspecified |
| Warp optimiser | Adam 2e-4, β (0.5, 0.999), clip 10 | [E] | paper unspecified (pix2pix convention) |
| Adversarial | LSGAN, λ_adv = 0.1, starts at step 500, real maps smoothed 0.1 | [E] | paper Eq. 3 ambiguous, λ not given |
| Per-garment affine (self triplets) | ±15°, scale 0.85–1.15, shift ±15% | [P] Sec. 5.1 (values [E]) | |
| Cross-view share | 0.5 | [E] | |
| Body-part jitter | up to 3 px at 128 px (scaled with size) | [E] | counters the silhouette leak (D-B) |
| Batch / steps | 8 / 10,000 (≈ 7.9 epochs) | [E] | ~40 min on a T4 |
| Validation during training | first 50 batches (400 images) every 1,000 steps | [E] | full split only in `evaluate_checkpoint` |
| mIoU definition (v2) | mean IoU over skin + 6 garment classes | [E] | see D-E |
| Garment-only background removal | tolerance 30 (RGB max-channel), min garment 2% of image, reject > 95% | [E] | experimental; not validated on real photos |
| SCHP input size | 473 (LIP) / 512 (Pascal) | [R] SCHP | |
| Texture α, correlation size | 100; 32×32 at 128 px (64×64 at 256) | [R] CoCosNet | Phase 5 |
| Texture optimiser | Adam lr_G 1e-4, lr_D 4e-4, β (0, 0.999) | [P] Sec. 5.1 + [R] TTUR | Phase 5 |

## 3. How to read the grids (`private/colab/*grid*.png`)

* **Top rows (validation, with ground truth):** input C_seg (randomly shifted per garment) | target body map
  (SCHP-Pascal: dark red head, green torso, olive upper arms, blue lower arms, purple upper legs, teal lower legs) |
  **prediction C'_seg** | ground truth (grey = not scored: hair, face, bags).
* **Bottom rows (swap test, no ground truth):** garment from one person, body from another. This is the real try-on case.
  The fourth column is white because there is no answer.
* The predicted head is skin-coloured on purpose; face and hair are pasted from the target later (paper Sec. 4.2).

## 4. Published reference values (paper, NOT our results)

| Task | Metric | Dress-up (published) |
|---|---|---|
| Pose transfer | SSIM / LPIPS / FID | 0.95 / 0.228 / 8.60 |
| Label-to-image | mIoU / Acc / FID | 87.6 / 98.9 / 8.60 |

These protocols are undocumented (see `docs/paper_analysis.md` §2.6) and measure different things from our warp mIoU.
Do not put them in the same table column as our numbers.
