# Stage C: texture stage (alignment network + SPADE translation network)

Source of truth: Dress-up Sec. 4.3.3, Fig. 9, Eqs. 5–9 [P]. Dress-up's text follows CoCosNet (Zhang et al.,
CVPR 2020) closely, and CoCosNet [R] fills the details Dress-up omits (see paper_analysis.md §2.3).
[E] marks our engineering choices.

## 1. Data flow and tensor shapes (128 px baseline, B = batch)

```
layout  L  = garment12 one-hot of the TARGET layout       [B, 12, 128, 128]   (C'_seg + target face/hair)
exemplar Y = reference photo I_C (RGB, [-1,1])             [B,  3, 128, 128]
exemplar parsing P_Y = garment12 one-hot of I_C            [B, 12, 128, 128]   [E] optional, default on

F_x(L)          -> T_B  [B, C, 32, 32]   C = 256             (domain adaptor, 2 stride-2 levels)
F_y(Y ⊕ P_Y)    -> T_C  [B, C, 32, 32]
centre over channels, L2-normalise each position (FP32)
M = T_B^T T_C   -> [B, 1024, 1024]   (HW x HW, Eq. 5)
A = softmax_v(α M), α = 100           (Eq. 6, FP32)
I_S = A · Y↓     -> [B, 3, 32, 32]    (Y↓ = exemplar downsampled to 32 x 32, as in CoCosNet)
generator: const z [B, 16·nf, 4, 4] -> 5 SPADE up-blocks -> RGB [B, 3, 128, 128]   (Eq. 7)
           each block normalised (positional norm) and modulated by γ, β predicted from (I_S ⊕ L)
```

At 256 px the correspondence runs at 64 × 64 (CoCosNet default). M is then 4096² ≈ 16.8 M entries per sample,
about 64 MiB in FP32 plus softmax and backward. At 128 px it is 1024² ≈ 1 M entries, so memory is negligible.

## 2. Faithful / adapted

| Component | Paper / CoCosNet | Here | Status |
|---|---|---|---|
| Correlation, Eq. 5 | cosine similarity of channel-centred features | identical, computed in **FP32** even under AMP | faithful |
| Softmax warping, Eq. 6, α | α "controls sharpness"; CoCosNet α = 100 | α configurable (`model.texture_alpha`, default 100) | faithful |
| What is warped | Dress-up writes T_C(v); CoCosNet warps the exemplar image | exemplar **RGB** at correspondence resolution | adapted (CoCosNet) |
| Exemplar input to F_y | image only | image **+ its garment12 parsing** (`model.texture_exemplar_parsing`) | adapted [E]: semantic labels make the correspondence much easier to learn on a Colab budget; switchable for an ablation |
| Translation network | SPADE, constant code z; Eq. 7 conditioned on I_S | SPADE ResBlocks from a learned constant; γ, β from (I_S ⊕ layout); positional normalisation | adapted: Eq. 7 conditions on I_S only; adding the layout is our choice (paper_analysis §2.3.2) |
| Adversarial, Eqs. 8–9 | hinge loss | hinge, 2-scale PatchGAN with spectral norm, conditioned on the layout | faithful (D architecture [E]) |
| Other losses | not in Dress-up; CoCosNet: VGG feature matching, perceptual, contextual, domain alignment, cycle regularisation | VGG perceptual, D feature matching, domain alignment, cycle regularisation, **warp L1** (warped exemplar vs downsampled ground truth, possible because our pairs are real) | adapted; **contextual loss not implemented** (paired ground truth is available) |
| Optimiser | Adam β = (0, 0.999), lr_G 1e-4, "imbalanced" lr_D | same; lr_D = 4e-4 (CoCosNet TTUR) | faithful + CoCosNet |
| Spectral norm | all layers of G and D | D: all layers. G: SPADE convs | adapted |
| SyncBN, ≥4 P40 | — | single GPU, no SyncBN | adapted |

## 3. Training pairs

DFMM has several photos of the same item (same garment, usually the same model, different pose):

* **Cross-view (paired):** exemplar = photo s, layout = ground-truth garment12 parsing of photo t, target image = photo t.
  The generator must move the texture from pose s to pose t, which is the paper's task.
* **Pseudo pair (CoCosNet):** items with one annotated photo use exemplar = a geometrically distorted copy of the target photo.

Layouts at training time come from DFMM manual parsing (clean). At try-on time they come from the warp
network (C'_seg) with the target's face/hair pasted in, so they are noisier. This train/test gap is known; the
end-to-end test in Phase 6 will measure its effect.

**Identity:** training pairs show the same model, so the exemplar also contains the right face. At try-on
time the exemplar shows a different person, so faces cannot come from the generator. Phase 6 therefore
pastes the target's own face/hair pixels back.

## 4. Validation and model selection

Paired validation pairs (same item, different photo) have ground truth, so per-image L1 and **SSIM** (on
[0,1] images, 11×11 Gaussian window, σ = 1.5, Wang et al. 2004) can be measured on the val split. The checkpoint with the best val SSIM is kept.
FID/LPIPS come in Phase 7 (FID needs thousands of images, see paper_analysis §2.6).

## 5. Pretrained weights needed

VGG-19 ImageNet weights from torchvision (`download.pytorch.org`, BSD licence, ~548 MB) for the perceptual
loss. They are only downloaded when `training.allow_vgg_download: true`, and cached in
`<drive>/pretrained/torch` so Colab downloads them once.
