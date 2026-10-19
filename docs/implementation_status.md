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
