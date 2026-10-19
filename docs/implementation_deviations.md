# Deviations from the Dress-up paper

Every place where the implementation differs from the paper, with the reason. Status labels:
**faithful**, **adapted**, **pretrained external**, **simplified substitute**, **not implemented**.
Sources: [P] = paper, [R:x] = reference x, [E] = our engineering decision.

| # | Paper [P] | Implementation | Status | Reason |
|---|---|---|---|---|
| D1 | LIP-JppNet for clothing segmentation (Sec. 4.3.1) | SCHP ResNet-101 trained on LIP (same 20 labels), re-implemented in pure PyTorch | pretrained external | JppNet is TensorFlow 1.x and cannot be installed on Colab. SCHP is MIT-licensed and stronger |
| D2 | InPlaceABNSync inside SCHP [R:SCHP] | `ABNCompat`, which scales by abs(γ)+eps exactly like the InPlaceABN kernel | adapted (numerically equivalent) | Avoids a CUDA extension that often fails to compile. Not yet confirmed with the real checkpoint |
| D3 | NBF 12-part body segmentation (Sec. 4.3.1) | DensePose parts merged into `body15` (14 parts, dataset-provided) **or** SCHP Pascal-Person-Part (7 parts) | simplified substitute | NBF is TensorFlow 1.6 and needs SMPL (registration, non-commercial). DensePose is precomputed in DFMM |
| D4 | Fine-tuning the parsers on DeepFashion (Sec. 4.3.1) | Parsers used as-is; DFMM manual parsing used directly as training ground truth | adapted | DFMM provides manual labels, which are better than fine-tuned predictions |
| D5 | "3-channel colour-coded" segmentation (Sec. 4.3.1); CE over c=1..3 (Eq. 2) | Integer label maps (`garment12`, 12 classes), one-hot for networks; `decode_color_map` for colour inputs | adapted | Cross-entropy needs class indices. Three "classes" contradicts the garment variety shown in the paper's figures |
| D6 | Dataset: DeepFashion In-shop, "52,712 training images" (Sec. 5.1) | DeepFashion-MultiModal (subset of In-shop, 12,701 images with manual parsing) | adapted | In-shop access not confirmed (password needed). DFMM provides the needed annotations |
| D7 | Train/test split not described | Item-disjoint split (seed 0, 80/10/10) | adapted | Prevents garment leakage between splits |
| D8 | 256×256 (Sec. 5.1) | Aspect-preserving letterbox to the configured size (128 for the baseline, 256 extended) | adapted | Stretching 750×1101 images to a square distorts bodies. 128 fits Colab compute |
| D9 | Garment-only (product photo) references | Experimental preprocessing only | not in the paper | User requirement; see `clothing_only_mode.md` |
| D10 | Warp, texture, losses, optimiser details | — | not implemented (Phases 4–5) | — |

Planned deviations for Phases 4–5 (already documented in `paper_analysis.md` §2): body-only U-Net
skips by default; a 1×1 fusion conv from 1024 to 512 channels; warping the RGB exemplar as in
CoCosNet; the CoCosNet loss set in addition to the hinge GAN; D lr 4e-4; no SyncBN.
