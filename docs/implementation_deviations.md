# Deviations from the Dress-up paper

Every place where the implementation differs from the paper, with the reason. Status labels:
**faithful**, **adapted**, **pretrained external**, **simplified substitute**, **not implemented**.
Sources: [P] = paper, [R:x] = reference x, [E] = our engineering decision.

| # | Paper [P] | Implementation | Status | Reason |
|---|---|---|---|---|
| D1 | LIP-JppNet for clothing segmentation (Sec. 4.3.1) | SCHP ResNet-101 trained on LIP (same 20 labels), re-implemented in pure PyTorch | pretrained external | JppNet is TensorFlow 1.x and cannot be installed on Colab. SCHP is MIT-licensed and stronger |
| D2 | InPlaceABNSync inside SCHP [R:SCHP] | `ABNCompat`, which scales by abs(γ)+eps exactly like the InPlaceABN kernel | adapted (numerically equivalent) | Avoids a CUDA extension that often fails to compile. Real LIP and Pascal checkpoints load strictly (603/603 tensors); 5.87% of LIP gammas are negative, so the abs(γ) rule matters |
| D3 | NBF 12-part body segmentation (Sec. 4.3.1) | DensePose parts merged into `body15` (14 parts, dataset-provided) **or** SCHP Pascal-Person-Part (7 parts) | simplified substitute | NBF is TensorFlow 1.6 and needs SMPL (registration, non-commercial). DensePose is precomputed in DFMM |
| D4 | Fine-tuning the parsers on DeepFashion (Sec. 4.3.1) | Parsers used as-is; DFMM manual parsing used directly as training ground truth | adapted | DFMM provides manual labels, which are better than fine-tuned predictions |
| D5 | "3-channel colour-coded" segmentation (Sec. 4.3.1); CE over c=1..3 (Eq. 2) | Integer label maps (`garment12`, 12 classes), one-hot for networks; `decode_color_map` for colour inputs | adapted | Cross-entropy needs class indices. Three "classes" contradicts the garment variety shown in the paper's figures |
| D6 | Dataset: DeepFashion In-shop, "52,712 training images" (Sec. 5.1) | DeepFashion-MultiModal (subset of In-shop, 12,701 images with manual parsing) | adapted | In-shop access not confirmed (password needed). DFMM provides the needed annotations |
| D7 | Train/test split not described | Item-disjoint split (seed 0, 80/10/10) | adapted | Prevents garment leakage between splits |
| D8 | 256×256 (Sec. 5.1) | Aspect-preserving letterbox to the configured size (128 for the baseline, 256 extended) | adapted | Stretching 750×1101 images to a square distorts bodies. 128 fits Colab compute |
| D9 | Garment-only (product photo) references | Experimental preprocessing only | not in the paper | User requirement; see `clothing_only_mode.md` |
| D10 | Warp generator (Fig. 8): 512×16×16 body code, 2×2×1024 clothes code, 4 res blocks | same numbers; plus 1×1 fusion conv 1024→512, body-only U-Net skips (configurable), nearest-upsample + 3×3 conv decoder, InstanceNorm, logits output | adapted | paper omits skips, norms and decoder details |
| D11 | L_CE over "c = 1..3" (Eq. 2) | CE over 12 garment12 classes; face/hair/headwear/accessory are ignored (not predictable; face/hair are replaced by the target's own parse, Sec. 4.2) | adapted | see D5 |
| D12 | L_adv (Eq. 3), λ_adv not given | conditional PatchGAN on (garment probs ⊕ B_seg) with spectral norm; LSGAN by default (hinge/vanilla selectable); λ=0.1 after 500 CE-only steps; real maps label-smoothed | adapted | Eq. 3 is ambiguous; λ and D are unspecified |
| D13 | Self-supervised triplets with random affine (Sec. 5.1) | per-class affine (SwapNet-style) **plus** cross-view triplets from same-item photos (`warp_pairing: mixed`) | faithful + addition | DFMM item IDs make real pose-change supervision possible |
| D14 | Body segmentation with NBF (12 parts, body under clothing) | SCHP-Pascal (7 parts) traces the **clothed** silhouette; per-part dilation/erosion augmentation and a swap-test `class_fidelity` metric counter the silhouette shortcut | simplified substitute | see `body_representation.md` |
| D15 | Warp optimiser not given | Adam lr 2e-4, betas (0.5, 0.999), grad-norm clip 10 | engineering | pix2pix convention |
| D16 | Alignment network (Eqs. 5–6): feature pyramids F_x, F_y; correlation; softmax(α M) | domain adaptors to 32×32 (64×64 at 256 px) + shared block; cosine of channel-centred features and softmax computed in **FP32**; α = 100 | faithful (CoCosNet details) | Dress-up gives no adaptor details |
| D17 | Eq. 6 warps T_C (features) | warps the exemplar **RGB** downsampled to the correspondence size | adapted | follows CoCosNet; Dress-up calls the result an image (I_S) |
| D18 | F_y input = exemplar image | exemplar image **+ its garment12 parsing** (switchable) | adapted | semantic labels make the correspondence learnable on a small budget; ablation planned |
| D19 | SPADE translation (Eq. 7) conditioned on I_S, constant code z | learned constant input; SPADE γ/β from I_S **+ layout**; positional normalisation; spectral norm | adapted | layout gives sharp garment boundaries the 32×32 I_S cannot |
| D20 | Texture losses: hinge GAN only (Eqs. 8–9) | hinge GAN (2-scale PatchGAN) + D feature matching + VGG-19 perceptual + warp L1 + cycle regularisation + domain alignment; **no contextual loss** | adapted | adversarial-only does not train at this scale; paired ground truth makes the contextual loss less necessary |
| D21 | Texture optimiser: Adam (0, 0.999), lr_G 1e-4, "imbalanced" lr_D | same, lr_D = 4e-4 | faithful + CoCosNet TTUR | |
| D23 | Identity: replace face/hair *labels* with the target's (Sec. 4.2) | same, **plus** the target's face/hair *pixels* pasted into the result with a feathered (Gaussian σ=1 px) mask | adapted | the texture network is trained on same-person pairs and reproduces the exemplar's face |
| D22 | Texture training data (not described) | ground-truth layouts; exemplar = another photo of the same item (80%) or a distorted copy of the target (20%, CoCosNet pseudo pairs) | adapted | warp outputs are only used at try-on time (Phase 6) |

No SyncBN anywhere (single GPU). Full texture-stage design: `texture_stage.md`.
