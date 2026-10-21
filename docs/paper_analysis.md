# Dress-up — Paper Analysis and Feasibility Audit (Phase 1)

**Paper:** H. Ghodhbani, M. Neji, A. M. Qahtani, O. Almutiry, H. Dhahri, A. M. Alimi,
"Dress-up: deep neural framework for image-based human appearance transfer",
*Multimedia Tools and Applications* 82:23151–23178 (2023). DOI 10.1007/s11042-022-14127-w.
Local copy: `researchpaper.pdf` (28 pages, read in full, including Figs. 1–16, Eqs. 1–9, Tables 1–2).

**Status of this document:** Phase 1 output. **No code has been written, no model downloaded, trained or evaluated.**
Every statement is tagged with its source:

| Tag | Meaning |
|---|---|
| **[P]** | Stated in the Dress-up paper (section / figure / equation given) |
| **[R:x]** | Stated in a reference the paper builds on (x = SwapNet, CoCosNet, SPADE, …), checked against that source |
| **[V]** | Verified during this audit from the project page / repository (Oct 2026) |
| **[E]** | Our engineering decision or interpretation — not from the paper |
| **[?]** | Unknown / ambiguous; must be decided or verified later |

---

## 1. Plain-English explanation of the method

You give the system two photos:

* **I_C** — a person wearing the clothes you want ("reference / clothing image").
* **I_B** — the person who should wear them, in the pose you want ("target / body image").

Directly copying pixels from I_C to I_B fails when the two people stand differently. Dress-up therefore
splits the problem into three steps **[P §3.1, Fig. 1, Fig. 3]**:

1. **Pre-treatment (understand the pictures).** Two pretrained segmentation networks turn each photo into
   a colour-coded label map. From I_C we get a **clothing segmentation C_seg** ("these pixels are the top,
   these are the trousers, …"). From I_B we get a **body-part segmentation B_seg** ("this is the left forearm,
   this is the torso, …"). Label maps contain shape and layout, but no colour or texture.
2. **Warp (move the clothing *shape* onto the new body).** A neural network looks at C_seg and B_seg and draws
   a new clothing label map **C′_seg**: the same garments as in C_seg, but fitted to the pose and body of B_seg.
   This is only a *shape* — still no texture.
3. **Texture (paint the clothes).** A second network takes C′_seg (the layout to draw) and the original photo
   I_C (the colours and patterns to copy). It first works out, for every location in C′_seg, which location of
   I_C it corresponds to (an "alignment"/correspondence step), builds a roughly-aligned "style image" from I_C,
   and then a SPADE generator paints the final photo **I_G**.

Before step 3, the face and hair labels in C′_seg are replaced by the face and hair of the target person so that
the output keeps the target's identity **[P §4.2]**.

The key idea is that predicting a *label map* in the new pose is much easier than predicting a full photo, and
once the layout is known, texture can be copied by semantic correspondence.

---

## 2. Technical explanation of each network

### 2.1 Stage A — Pre-treatment (human parsing)

| Item | Paper statement |
|---|---|
| Clothing segmentation | **LIP-JppNet** [33] (Liang et al., TPAMI 2018): ResNet-101 backbone, atrous conv, ASPP, multi-scale inputs, joint parsing + pose with iterative refinement **[P §4.3.1, Fig. 6, §5.1]**. Trained/fine-tuned on DeepFashion In-shop with random scale/crop/flip **[P §4.3.1, §5.1]**. |
| Body segmentation | **Neural Body Fitting (NBF)** [42] (Omran et al., 3DV 2018): RefineNet (ResNet-101) produces a **12-part** body segmentation; a ResNet-50 regresses SMPL parameters. Dress-up uses **only the 2D part segmentation** and ignores SMPL **[P §4.3.1, Fig. 7, §5.1]**. |
| Why these models | Replace SwapNet's LIP-SSL and Unite-the-People **[P §4.3.1]**. |
| Representation | SwapNet uses 18-channel one-hot clothing maps; Dress-up states it uses a **"3-channel color-coded segmentation"** instead **[P §4.3.1]**. |
| Identity step | Face and hair segments in the clothing representation are replaced by those of the target person before texture synthesis **[P §4.2]**. |

**Role of each segmentation.** C_seg describes *what* garments exist and their categories, independent of the
reference body. B_seg describes *where* the target body parts are, independent of what the target is wearing.
The warp network must combine "what" from C_seg with "where" from B_seg. For this to work, B_seg **must not
leak the target's current clothing**, which is why a body-*part* model (NBF) rather than a clothing parser is
used for I_B **[E — interpretation, consistent with SwapNet]**.

**Observation [P]:** The caption of Fig. 4 says (a) = NBF body segmentation and (b) = LIP clothing segmentation,
but visually panel (a) shows clothing parses (hair/face/garments) and (b) shows body-part maps. The caption is
likely swapped.

### 2.2 Stage B — Garment warp network (dual-path U-Net)

**Paper facts [P §4.3.2, Fig. 8, Eqs. 1–4, §5.1]:**

* `C′_seg = G_warp(B_seg, C_seg)` (Eq. 1).
* Two encoders ("Pose Encoder" for B_seg, "Clothes Encoder" for C_seg) and one decoder.
* Body encoder → feature map **512 × 16 × 16**.
* Clothes encoder → narrow bottleneck **2 × 2 × 1024** ("weakly conditioned"), then **up-sampled** to 512 × 16 × 16.
* The two maps are concatenated and passed through **4 residual blocks** (Fig. 8 labels them "n-Resblocks 16×16×512"),
  then decoded to C′_seg.
* Output is "robustly conditioned on the body representation and weakly conditioned on the clothes representation".
* Loss: `L_warp = L_CE + λ_adv · L_adv` (Eq. 4).
* Training is **self-supervised**: ground-truth triplets are generated with data augmentation and **random affine
  transforms** of the clothing segmentation, so the network must ignore location cues in C_seg and use only
  high-level garment cues **[P §5.1]**.
* Working resolution 256 × 256 **[P §5.1]**.

**Reference implementation detail [R:SwapNet community re-implementation, `andrewjong/SwapNet`, verified V]:**
The (unofficial) SwapNet `WarpModule` uses: body encoder 4 stride-2 blocks (3→64→128→256→512, H/16);
cloth encoder 6 stride-2 blocks (→1024, H/64); two up-sampling blocks *without* skips back to 512 × H/16;
concatenation → 1024 channels → 4 residual blocks; a **dual decoder with U-Net skip connections from both encoders**
at H/8, H/4, H/2; output conv to 19 cloth channels + tanh. There is **no official SwapNet code**.

**Inconsistencies to resolve [?]:**
* At 256 × 256 input, a 512 × 16 × 16 body map implies 4 down-samplings (H/16), and a 2 × 2 cloth bottleneck implies
  7 down-samplings (H/128). The SwapNet re-implementation uses H/64 (4 × 4 at 256). We will follow the **paper's
  numbers** (2 × 2 × 1024) and make depth configurable **[E]**.
* Fig. 8 shows the residual blocks at 512 channels while concatenation yields 1024. We will insert a 1×1 fusion conv
  1024→512 **[E]** (configurable).
* The paper never shows skip connections in Fig. 8 even though it calls the network a "U-Net". We will add skip
  connections **from the body encoder only** by default (matches "strongly conditioned on B_seg"), with a config flag
  for dual skips as in the SwapNet re-implementation **[E]**.
* Normalisation, activation, dropout and the discriminator architecture are not stated **[?]**.

**Eq. 2 — cross-entropy [P]:**

```
L_CE = − Σ_{c=1..3} 1(C_seg(i,j) = c) · log(G_warp(i,j))
```

Per pixel, it is the negative log-probability that the network assigns to the correct clothing class.
Problems: (i) the sum runs over **c = 1..3**, which seems to confuse the 3 RGB channels of the colour-coded image
with semantic classes; CE needs class indices, not colours. (ii) The target is written as C_seg (the input), which
makes sense only in the self-supervised setting where the input is an affine-augmented copy of the target.
**Our decision [E]:** decode colour maps to integer labels; use K-class pixel-wise CE with softmax output
(K = size of our reduced clothing label set, ~8–12), averaged over pixels.

*What CE does to the garment shape:* it forces every pixel to the correct label, so it learns the overall silhouette
and garment boundaries, but on its own it tends to produce blurry/averaged boundaries in ambiguous regions.

**Eq. 3 — adversarial term [P]:**

```
L_adv = E_{x∼p(C_seg)}[D(x)] + E_{z∼p(f1_enc(C_seg,B_seg))}[1 − D(f1_dec(z))]
```

The discriminator D sees real clothing maps x and generated maps f1_dec(z). The equation is written without
logarithms and without specifying whether D sees the map alone or together with B_seg. **Our decision [E]:**
implement a conditional PatchGAN discriminator on (softmax map ⊕ B_seg) with the LSGAN or hinge formulation
(configurable), λ_adv small (start 0.1, [?] paper gives no value). *Its effect:* it penalises shapes that "don't
look like real garments" (ragged edges, fragmented regions), sharpening boundaries that CE alone leaves fuzzy.

### 2.3 Stage C — Texture stage (alignment + SPADE translation)

**Key finding [V]:** Dress-up's texture stage text and Eqs. 5, 6, 8, 9 match, almost verbatim, the
**CoCosNet** paper (Zhang et al., "Cross-domain correspondence learning for exemplar-based image translation",
CVPR 2020, ref. [62]), Eqs. 3, 4, 13; the implementation paragraph (Adam β1=0, β2=0.999, "imbalanced learning
rates", 1e-4 for G, spectral norm on all layers, SyncBN) is also the same as CoCosNet §4. We therefore treat
**CoCosNet as the authoritative reference for details Dress-up omits**, while keeping Dress-up's inputs
(C′_seg and I_C).

#### 2.3.1 Alignment network [P §4.3.3, Fig. 9, Eqs. 5–6]

1. Feature extractors F_x (on C′_seg) and F_y (on I_C) — called "feature pyramid networks" in the paper —
   produce T_B ∈ R^{HW×C} and T_C ∈ R^{HW×C}.
2. Correlation matrix (Eq. 5), cosine similarity of channel-centred features:

   ```
   M(u,v) = T̂_B(u)ᵀ T̂_C(v) / (‖T̂_B(u)‖ ‖T̂_C(v)‖),   T̂(u) = T(u) − mean_c T(u)
   ```

   M(u,v) is high when target location u and reference location v "mean the same thing" (e.g. both are the
   left sleeve cuff).
3. Correspondence style image (Eq. 6):

   ```
   I_S(u) = Σ_v softmax_v(α · M(u,v)) · T_C(v)
   ```

   Each target position takes a weighted average of reference values, weights concentrated on the most
   similar positions. α controls sharpness (large α ≈ pick the single best match).

**Facts from CoCosNet [R:CoCosNet, verified V]:** α = **100**; features live at **64 × 64** (HW = 4096, so
M is 4096 × 4096); what is warped is the **exemplar image downscaled to 64 × 64** (RGB), not the features;
domain adaptors are Conv-InstanceNorm-LeakyReLU blocks followed by a shared adaptive feature block; SyncBN
inside this sub-network.

**Ambiguity [?]:** Dress-up Eq. 6 writes T_C(v) (features) but calls the result a "style image" and Fig. 9 shows an
RGB image. **Our decision [E]:** warp the downsampled RGB reference (as CoCosNet) by default; optionally also warp
features (config flag) for an ablation.

**Tensor shapes (planned, 256 input, 64×64 correspondence) [E]:**

| Tensor | Shape |
|---|---|
| C′_seg one-hot | B × K × 256 × 256 |
| I_C | B × 3 × 256 × 256 |
| T_B, T_C (after reshape) | B × 4096 × C (C = 256) |
| M | B × 4096 × 4096 |
| softmax(αM) | B × 4096 × 4096 |
| I_C downsampled | B × 3 × 64 × 64 → B × 4096 × 3 |
| I_S | B × 4096 × 3 → B × 3 × 64 × 64 |

#### 2.3.2 Translation network (SPADE) [P §4.3.3, Eq. 7]

```
α_i(I_S)[h,w] · (F_i[c,h,w] − μ_i[h,w]) / σ_i[h,w] + β_i(I_S)[h,w]        (Eq. 7)
```

The generator starts from a constant code z and progressively up-samples; before each normalisation layer the
activation is normalised and then re-scaled/shifted with per-pixel parameters α_i, β_i predicted by small convs
from I_S. Because α, β are spatial maps, local style (this region is red-striped) is injected where it belongs,
avoiding the "wash-away" of a single global style vector **[P §4.3.3]**.

Note: per-position mean/std μ_i[h,w], σ_i[h,w] (over channels) is **positional normalisation**, which is what
CoCosNet uses **[R:CoCosNet App. E]**; original SPADE uses batch-norm statistics **[R:SPADE]**.

**Ambiguity [?]:** the paper also describes the translation network as "an adversarially trained encoder-decoder
… inspired by VAE … encoding its style into a latent style vector", which contradicts "converts the constant code z".
**Our decision [E]:** follow CoCosNet (constant z, no VAE).

**What conditions the generator?** Dress-up Eq. 7 conditions only on I_S. In CoCosNet the semantic input is used
through the correspondence. **Our decision [E]:** condition SPADE on concat(I_S upsampled, C′_seg one-hot), so the
layout is also injected directly (configurable; documented deviation).

#### 2.3.3 Texture-stage losses

**Paper [P Eqs. 8–9]:** hinge adversarial loss only:

```
L_D = −E[h(D(T_C))] − E[h(−D(G(T_B, T_C)))],   L_G = −E[D(G(T_B, T_C))],   h(t) = min(0, −1+t)
```

(Dress-up writes D(T_C) where CoCosNet writes D(y_B), i.e. D sees **real images**, not features; we follow that.)

**CoCosNet full objective [R:CoCosNet Eq. 14]:** feature-matching (VGG-19) on pseudo pairs, perceptual (relu4_2),
contextual loss (relu2_2–relu5_2), hinge adversarial, domain-alignment L1, and cycle **correspondence regularisation**
`L_reg = ‖r_{y→x→y} − y‖₁`. CoCosNet's ablation shows that without L_reg the correspondence is not learned correctly.

**Our decision [E]:** an adversarial-only objective will not train a usable model at our scale; we will use the
CoCosNet loss set (weights configurable, contextual loss optional because of memory) and document this as a deviation.

### 2.4 Optimiser and training settings

| Setting | Dress-up [P §5.1] | CoCosNet [R] | Planned for Colab [E] |
|---|---|---|---|
| Texture optimiser | Adam β=(0, 0.999) | same | same |
| LR generator | 1e-4 | 1e-4 | 1e-4 |
| LR discriminator | "imbalanced" (not given) | 4e-4 (TTUR) | 4e-4 |
| Spectral norm | all layers, G and D | same | same |
| Sync BatchNorm | yes, ≥4 × P40 | yes, 8 × V100 | **no** (single GPU; plain BN / InstanceNorm) |
| Resolution | 256 × 256 | 256 × 256 | 128 (Mode B) → 256 (Mode C) |
| Warp optimiser / LR | not given | — | Adam(0.5, 0.999), 2e-4 [E] |
| Batch size, epochs, λ values | not given | not given in paper | configurable |
| Training time | not given | ~4 days / 100 epochs ADE20k on 8×V100 | — |

### 2.5 Dataset [P §5.1]

* DeepFashion **In-shop Clothes Retrieval Benchmark**, described as "52,712 training images".
* **[V]** 52,712 is the size of the *whole* In-shop benchmark (7,982 items). The official partition file splits it into
  train / query / gallery; the train split is commonly reported as 25,882 images **[? — confirm from
  `list_eval_partition.txt` after download]**. The paper does not say which split it trained or tested on.
* Data augmentation is used because pairs of "similar clothes on distinct bodies" are hard to obtain **[P]**.
* Data availability statement: "available from the corresponding author upon reasonable request" **[P]**. No code or
  checkpoints were released **[V — none found]**.

### 2.6 Reported evaluation (published reference values only — never our results)

**Garment transfer [P §5.2.1]:** qualitative only, vs SwapNet and Deep Re-rendering (results taken from those authors).

**Table 1 — Pose transfer, DeepFashion [P]:**

| Method | SSIM ↑ | LPIPS (paper marks ↑) | FID ↓ |
|---|---|---|---|
| ADGAN | 0.77 | 0.226 | 16.69 |
| DiOr | 0.80 | 0.176 | 14.34 |
| Ivan et al. | 0.71 | 0.175 | 12.74 |
| **Dress-up (published)** | **0.95** | **0.228** | **8.60** |

**Table 2 — Label-to-image, DeepFashion [P]:**

| Method | mIoU ↑ | Acc ↑ | FID ↓ |
|---|---|---|---|
| pix2pixHD | 85.2 | 98.8 | 17.76 |
| SMIS | 87.3 | 98.9 | 9.50 |
| SPADE | 87.1 | 98.9 | 10.02 |
| **Dress-up (published)** | **87.6** | **98.9** | **8.60** |

**Protocol concerns — must be resolved before any numerical comparison [E]:**
1. LPIPS is a *distance*; for paired reconstruction lower is better. The paper marks it "↑" and calls it a diversity
   measure. Under the usual convention Dress-up's 0.228 would be the *worst* of the four.
2. The same FID (8.60) is reported for two different tasks.
3. SSIM 0.95 vs ≤ 0.80 for all baselines is an unusually large gap; test pairs, crop, resolution and SSIM settings
   are not given.
4. Table 2's mIoU/Acc protocol (presumably a parser run on generated images, as in SPADE) is not described.
5. Test split, number of test pairs, and FID sample count are not given.

Consequently our numbers will be reported **with our own protocol** next to — never merged with — the published values.

### 2.7 Limitations stated by the authors [P §5.3.2]

* Complex or rarely seen poses are not always transferred correctly (Fig. 16).
* Visible artifacts; garments not always filled in properly.
* Shading, texture warping and detail preservation not fully realistic.
* Output depends on the intermediate segmentation; an unsuitable C′_seg causes failures. Remedy suggested: improve the
  warp stage for large pose differences.

### 2.8 Future work proposed by the authors [P §5.3.3]

1. Improve the warp module's intermediate representation; study how much better human segmentation helps.
2. Support all fashion categories (hats, accessories, bags).
3. More advanced warping and higher-resolution training/generation.
4. Better handling of complex textures.
5. Better scene compositing.

---

## 3. Paper-to-code mapping (planned)

| Paper element | Source | Planned module | Status |
|---|---|---|---|
| Three-stage pipeline | §3.1, Figs. 1 & 3 | `src/inference/pipeline.py::try_on` | Faithfully implemented (structure) |
| Clothing segmentation (LIP-JppNet) | §4.3.1, Fig. 6 | `src/models/segmentation.py` (`SCHPParser`, LIP labels) | **Pretrained external component (substitute)** |
| Body segmentation (NBF 12 parts) | §4.3.1, Fig. 7 | `src/models/segmentation.py` (`BodyPartParser`) | **Simplified experimental substitute** |
| SMPL regression in NBF | §4.3.1 | — | Not implemented (paper also ignores it) |
| Colour-coded segmentation | §4.3.1 | `src/data/preprocessing.py` (palette ↔ labels) | Adapted (one-hot internally) |
| Face/hair replacement | §4.2 | `src/data/preprocessing.py::inject_identity_labels` | Adapted implementation |
| Dual-path U-Net G_warp, Eq. 1 | §4.3.2, Fig. 8 | `src/models/warp_network.py` | Adapted implementation |
| L_CE, Eq. 2 | §4.3.2 | `src/training/losses.py` | Faithfully implemented (K-class form) |
| L_adv (warp), Eq. 3 | §4.3.2 | `losses.py`, `discriminator.py` | Adapted implementation |
| Self-supervised triplets + random affine | §5.1 | `src/data/dataset.py`, `transforms.py` | Adapted implementation |
| Feature extractors F_x, F_y | §4.3.3, Fig. 9 | `src/models/alignment_network.py` | Adapted (CoCosNet-style) |
| Correlation matrix, Eq. 5 | §4.3.3 | `alignment_network.py::correlation` | Faithfully implemented |
| Softmax warping with α, Eq. 6 | §4.3.3 | `alignment_network.py::warp` | Faithfully implemented |
| SPADE generator, Eq. 7 | §4.3.3, Fig. 9 | `src/models/spade_generator.py` | Adapted (reduced width, PN) |
| Hinge adversarial, Eqs. 8–9 | §4.3.3 | `losses.py`, `discriminator.py` | Faithfully implemented |
| Perceptual / FM / contextual / L_reg / domain | CoCosNet (not in Dress-up) | `losses.py` | Adapted (documented addition) |
| Adam β=(0,0.999), lr_G 1e-4 | §5.1 | `train_texture.py`, configs | Faithfully implemented |
| Spectral norm | §5.1 | model modules | Faithfully implemented |
| Synchronized BN, multi-GPU | §5.1 | — | Not implemented (single GPU) |
| 256 × 256 resolution | §5.1 | configs | Adapted (128 for Mode B) |
| DeepFashion In-shop | §5.1 | `src/data/dataset.py` | Adapted (see §6) |
| SSIM / LPIPS / FID | §5.2.2 | `src/evaluation/metrics.py` | Adapted (protocol defined by us) |
| mIoU / pixel acc | §5.2.3 | `metrics.py` | Adapted |
| Pretrained Dress-up weights | — | — | **Not available** |

---

## 4. Components reusable through compatible pretrained models

| Component | Candidate | Source / licence [V] | Task match | Colab compatibility |
|---|---|---|---|---|
| Clothing parser (primary) | **SCHP, LIP checkpoint** (20 LIP labels; ResNet-101; 59.36 mIoU LIP val) | `PeikeLi/Self-Correction-Human-Parsing`, MIT | Same label set and dataset family as LIP-JppNet → closest available substitute | PyTorch. Uses InPlaceABN (JIT-compiled CUDA extension). **Phase 3 update [V]:** re-implemented in pure PyTorch (`src/models/schp_network.py`). InPlaceABN scales by `abs(γ)+eps`, **not** γ (read from the kernel source), so plain `BatchNorm2d` would load the weights but give wrong outputs. `ABNCompat` reproduces the abs(γ)+eps rule. Architecture checked against the original source: all 603 tensor names and shapes match, and outputs agree to 2e-5. The real checkpoint has **not** been loaded yet (see §10). |
| Clothing parser (fallback) | `mattmdjaga/segformer_b2_clothes` (18 ATR labels, mIoU 0.69 reported) | Hugging Face; licence "other" → NVIDIA SegFormer licence (**non-commercial**) | Clothing parser, ATR labels | `transformers`, no compilation |
| Body-part parser | **SCHP, Pascal-Person-Part checkpoint** (7 labels: bg, head, torso, upper arms, lower arms, upper legs, lower legs) | same repo, MIT | Clothing-independent body parts (like NBF), but coarser (7 vs 12) and no left/right | same as above |
| Body-part (optional, Experiment A) | DensePose (24 surface parts → can be merged to 12) | detectron2 (Apache-2.0); DeepFashion-MultiModal ships pre-computed DensePose [V] | Finer than NBF | detectron2 needs building in Colab (several minutes) |
| VGG-19 (perceptual/contextual losses) | torchvision ImageNet weights | BSD | loss network only | preinstalled |
| Inception (FID), AlexNet/VGG (LPIPS) | `torchmetrics`/`clean-fid`, `lpips` | permissive | metrics | pip |
| External reference baseline (optional) | **DiOr** (Dressing in Order) pretrained DeepFashion checkpoints | `cuiaiyu/dressing-in-order`; licence to verify | Person-to-person try-on, but a *different method* | inference reportedly works with current torch; training needs PyTorch 1.0 CUDA ops |

**Rejected / not reusable [V]:**

| Component | Why |
|---|---|
| Official LIP-JppNet | TensorFlow 1.x, MIT, checkpoint on Google Drive. TF 1.x wheels do not exist for Colab's Python ≥ 3.10. Porting is possible but not worth it when SCHP provides the same labels with a stronger model. |
| Official NBF | TensorFlow **1.6**, needs SMPL model files (registration, non-commercial licence). Not installable on current Colab. |
| SPADE official checkpoints | COCO-Stuff / ADE20K / Cityscapes only; CC BY-NC-SA 4.0. Wrong task and label set. |
| CoCosNet DeepFashion checkpoint | MIT, exists, but conditioned on **OpenPose keypoints**, not clothing segmentation → input incompatible with Dress-up's C′_seg. May serve as weight initialisation for the translation network only after a compatibility check **[?]**. |
| Community SwapNet checkpoints | Unofficial, no licence stated, author says "not yet perfect", different label representation (19 cloth channels, different body model). Use as *architecture reference* only. |
| Dress-up weights / code | Not released. |

---

## 5. Components that must be implemented or trained by us

1. Label-mapping layer (LIP 20 → reduced clothing set; PPP 7 body parts) and colour palettes.
2. Dataset indexer, split generator, triplet / pair builders, caching of parser outputs to Drive.
3. **Warp network** (dual-path U-Net) + patch discriminator — **must be trained** (no compatible weights exist).
4. **Alignment network** (domain adaptors, correlation, softmax warp) — **must be trained**.
5. **SPADE translation generator** + multi-scale discriminator — **must be trained**.
6. Loss functions, training loops with AMP, gradient accumulation, checkpoint/resume.
7. Identity-preserving composition (face/hair labels; optional pixel compositing — see §7).
8. Evaluation suite and result tables.

**Consequence:** there is **no way to run the full Dress-up pipeline end-to-end using only pretrained weights.**
"Mode A" can demonstrate Stage A with pretrained parsers, and (optionally, clearly labelled) a different
pretrained method (DiOr) as a reference. A real Dress-up-style end-to-end result requires Mode B training of the warp
and texture networks.

---

## 6. Data: availability and suitability

| Dataset | Content [V] | Access [V] | Suitability |
|---|---|---|---|
| **DeepFashion In-shop** (paper's dataset) | 52,712 images, 7,982 items, ~200k cross-pose pairs; bbox, landmarks, item/identity & pair annotations; project page also lists parsing masks and DensePose | Google Drive. **Non-commercial research only; no redistribution. Some archives are password-protected — you must sign the agreement and email it to the authors to receive the password.** | Faithful to the paper. Item IDs give *same garment, different pose* pairs → real supervision for both warp (cross-view) and texture (exemplar → target). Official split is by item, preventing leakage. Also hosts the standard pose-transfer test pairs used by most papers [? — confirm file]. |
| **DeepFashion-MultiModal** | 44,096 images from DeepFashion (750 × 1101); **manual human-parsing for 12,701 full-body images (24 classes)**, 21 keypoints, DensePose for all | Google Drive (~12 GB total; images 5.4 GB, parsing ~90 MB). Non-commercial research only | Gives **real ground-truth parsing** → valid mIoU for Stage A and true targets for the warp network (no pseudo-labels needed on that subset). The README only says the images are "from DeepFashion". **Phase 3 update [V]:** DiOr's download script renames DFMM images to the In-shop pose-transfer naming and matches them against the In-shop pair lists, and the file names contain In-shop item IDs. So the images are In-shop images, and same-item multi-view pairs can be built from file names. |

**Supervision strategy [E]:**

* **Warp, self-supervised (paper's method):** from one image I: target = C_seg(I); input clothing = per-label random
  affine transform of C_seg(I); body = B_seg(I).
* **Warp, cross-view (our addition, ablation):** for two images of the same item: input clothing = C_seg(view 1);
  body = B_seg(view 2); target = C_seg(view 2). This is genuine pose-change supervision.
* **Texture:** exemplar = image of view 1; layout = C_seg(view 2) with face/hair; ground truth = image of view 2;
  plus CoCosNet pseudo-pairs (exemplar = geometrically distorted view 2).
* **Pseudo-labels:** where manual parsing is not available, the parser output is a *pseudo*-label. That is
  acceptable for training the warp network (it is what Dress-up/SwapNet did) but **not** for evaluating Stage A.
  Segmentation metrics are only reported on manually annotated masks.

**Smoke test:** a tiny fixture of procedurally generated label maps and images (clearly marked synthetic) to test
shapes, losses and checkpointing without any download. It will never be used for reported results.

---

## 7. Feasibility on Google Colab

**GPU assumption:** unknown; design for a 15–16 GB T4 as the common case, degrade gracefully to less.

| Item | Estimate [E] | Mitigation |
|---|---|---|
| Correlation matrix at 64 × 64 | 4096² fp32 = 64 MiB per sample; with softmax & backward ≈ 250 MiB/sample | batch 4–8 fits on T4 |
| Correlation at 128 × 128 | 1 GiB per sample just for M | **avoid**; keep 64 × 64 (or 32 × 32 for 128 px images) |
| Mixed precision in correlation | α·M up to 100 → `exp` overflows fp16 | compute correlation + softmax in **fp32** (autocast disabled locally); rest in fp16/bf16 |
| Warp net @128 px | small; minutes per epoch on a few thousand images | Mode B |
| Texture net @128 px, reduced width | hours for a usable result on ~10k pairs; multiple sessions | checkpoint every N steps, resume after disconnect |
| Texture net @256 px | CoCosNet needed 8 × V100 × days | Mode C only; expect partial training |
| Parser pre-computation | one pass over the subset, cached to Drive | done once per backend |
| Google Drive space | MultiModal images 5.4 GB; In-shop low-res images smaller; free Drive = 15 GB | download only needed parts; store resized 256 px copies; keep checkpoints pruned |
| Session limits | free Colab sessions disconnect | step-based checkpoints, resumable data order |

**Verdict:**
* Stage A with pretrained parsers: **feasible**.
* Warp network training at 128–256 px: **feasible** in Mode B.
* Texture stage: **feasible at reduced scale**; quality will be well below the paper's figures. A faithful-scale
  reproduction is **not feasible** on Colab.
* Reproducing the paper's numbers: **not feasible**, and not verifiable because the protocol is undocumented.

### Proposed modes

* **Mode A — demonstration:** Stage A pretrained parsers + visualisations; optional DiOr reference output (labelled
  "external baseline, not Dress-up"). The pipeline returns a clear error for untrained stages.
* **Mode B — small-scale baseline:** subset of ~5–10k images, 128 px, warp then texture, item-disjoint splits.
* **Mode C — extended:** 256 px, full train split, more steps; resource estimate shown before you start it.

---

## 8. Unresolved implementation details

1. Number and definition of clothing classes (paper: "3-channel colour-coded"; Eq. 2 sums over 3 classes).
2. Exact down-sampling depth giving 2 × 2 × 1024 at 256 px; fusion from 1024 to 512 channels.
3. Whether/where the warp U-Net has skip connections (body only vs both encoders).
4. Warp discriminator architecture, GAN formulation, λ_adv, optimiser, LR, batch size, epochs.
5. Whether B_seg is 12-part NBF exactly and its label order (our substitute has 7 parts).
6. How the face/hair replacement is done (labels only, or also pixels), and **how the target's face/skin texture
   reaches the generator** — the texture stage receives only I_C as exemplar, so the target person's appearance
   is not available to it. Candidate fixes [E]: (a) paste original target pixels for face/hair/background using
   the masks; (b) build the exemplar as a composite of I_C (garments) and I_B (face/hair/skin). Default (a), (b) as
   experiment.
7. Whether Eq. 6 warps features or the RGB exemplar.
8. Translation network: constant z vs VAE encoder; what layout input, if any, the generator receives.
9. Texture losses other than adversarial; their weights; D learning rate.
10. Train/test split, number of test pairs, metric settings (SSIM window, LPIPS backbone, FID sample count).
11. Which garments are transferred (top only vs full outfit; the paper's figures transfer full outfits incl. shoes).
12. Background handling (DeepFashion backgrounds are mostly plain white).

---

## 9. Proposed implementation plan (Phases 2–8)

| Phase | Deliverable | Verification |
|---|---|---|
| 2 | Repo structure, `.gitignore`, config system (YAML + dataclass), notebook skeleton (sections 1–14), `requirements.txt`, README draft | imports, config loading, notebook runs top-to-bottom on CPU in "dry" mode |
| 3 | Dataset indexer, item-disjoint splits, validation/reporting; segmentation adapter interface; SCHP-LIP + SCHP-PPP backends; label maps, palettes, legends | unit tests on synthetic fixture; parser run on ≥2 real images in Colab |
| 4 | Warp net, discriminator, CE + adv losses, triplet builders, training loop with AMP/accumulation/resume | shape tests, finite losses, backward, checkpoint round-trip, overfit-one-batch test |
| 5 | Alignment net (fp32 correlation, α configurable), SPADE generator, multi-scale D, texture losses, training loop | same as Phase 4, plus correspondence visualisation |
| 6 | `try_on()` pipeline, identity composition, Drive saving, actionable errors | end-to-end test **only** once trained weights exist |
| 7 | SSIM/LPIPS/FID/mIoU/timing/memory, results table template (published vs ours kept separate) | metric unit tests on known inputs |
| 8 | `implementation_deviations.md`, `experiments.md` roadmap (Experiments A–H) | review |

**Decisions needing your input:** dataset access (In-shop requires the signed agreement / password; MultiModal is the
practical starting point) — see the question at the end of the Phase 1 report.

---

## 10. Phase 2–3 update (two reference modes, dataset verification, implementation status)

*This section was added after Phase 1. Sections 1–9 stay as the Phase 1 audit, plus three
inline "Phase 3 update" notes.*

### 10.1 Two reference modes

| Mode | Reference image | In the paper? | Status |
|---|---|---|---|
| **A: person-worn** | a person wearing the garment | yes (the paper's only setting) | Stage A implemented; Stages B and C are Phases 4–5 |
| **B: garment-only** | an isolated product photo | **no**, our extension | experimental preprocessing only; needs a different dataset (see `docs/clothing_only_mode.md`) |

Both modes go through one interface, `src/data/reference.py`, which returns a `PreparedReference`
with a `validated` flag. The pipeline refuses to pass `validated=False` references to later stages.

### 10.2 What DeepFashion-MultiModal (DFMM) supports [V]

| Need | DFMM | Consequence |
|---|---|---|
| Person-worn references | 44,096 images (750×1101) | ✅ |
| Clothing ground truth | manual 24-class parsing, but **only for the 12,701 full-body images** | ✅ Ground-truth C_seg for warp targets and for parser evaluation |
| Body representation | DensePose for all images (part index in channel 2, per the Text2Human loader) | ✅ `body15` (14 parts) replaces NBF's 12 parts. DensePose maps the body surface under the clothes, so it does not depend on what the person wears |
| Same garment, several views | In-shop item IDs in file names | ✅ item-disjoint splits; cross-view pairs for the warp and texture stages |
| Product / flat garment images | **none** (the README describes "human images" only) | ❌ garment-only mode cannot be trained or evaluated on DFMM |
| Product ↔ worn pairs | none | ❌ same as above |

**When In-shop itself would be needed:** for the standard pose-transfer test pairs (the paper's Table 1
protocol), and for full-body images that DFMM leaves without parsing. Access requires signing the
DeepFashion agreement and asking the authors for unzip passwords. It is not needed for Phases 4–5.

### 10.3 Body-representation consistency [E]

A warp network trained on one body label set must get the same label set at inference.
`labels.body_label_set` selects it:

* `body15`: training uses DFMM DensePose. **Inference on new photos would need a DensePose model**
  (detectron2), which is not integrated yet.
* `pascal_person_part`: training converts DFMM DensePose to the 7 PPP parts (LUT
  `densepose → pascal_person_part`), and inference uses `schp_pascal`. Coarser, but it runs on any
  photo. The two sources still segment differently (SCHP-PPP was trained on Pascal annotations,
  DensePose on body-surface annotations), so a domain gap remains. This is part of Experiment A.

This choice should be made before Phase 4 training; it is listed in §10.5.

### 10.4 Implementation status after Phase 3

| Component | Status |
|---|---|
| Config system (YAML + dataclasses, `_base_` inheritance, overrides, validation) | done, tested |
| DFMM indexing / validation / statistics / item-disjoint splits | done, tested on the **synthetic fixture only** |
| Label sets and LUTs (LIP, ATR, PPP, DFMM, DensePose → garment12 / body15 / PPP) | done, tested |
| SCHP backends (LIP / ATR / PPP), pure PyTorch | architecture verified against the source; **real checkpoint not loaded yet** |
| SegFormer-clothes backend | written, **never run** (needs a download) |
| Annotation backends (DFMM parsing / DensePose) | done, tested on the fixture |
| Parser output cache | done, tested |
| Garment-only preprocessing (plain-background removal + user category) | experimental, tested on synthetic product images only |
| Stage B warp network (Phase 4) | implemented and unit-tested (88 tests incl. exact resume and overfit-one-batch); **not trained on real data** |
| Stage C texture network (Phase 5) | implemented and unit-tested (correspondence maths, FP32 under autocast, exact resume, overfit SSIM 0.994); **not trained on real data** |

### 10.5 Decisions still open (in addition to §8)

1. `body15` (DensePose) vs `pascal_person_part` (SCHP) as the warp network's body input (§10.3).
2. Whether to approve the downloads: DFMM images + parsing + DensePose (~11 GB), and SCHP-LIP / SCHP-PPP weights (~255 MiB each).
3. For garment-only mode: whether to apply for VITON-HD (CC BY-NC 4.0, tops only) or DressCode (needs an institutional email and a signed form).
