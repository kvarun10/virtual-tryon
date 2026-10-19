# Garment-only (product photo) reference mode — investigation

**Status:** experimental. This mode is not part of the Dress-up paper. Only a preprocessing
component exists (`GarmentOnlyPreprocessor`). It is unvalidated and the pipeline will not pass its
output to later stages.

## 1. Why the person-worn pipeline does not transfer directly

| Pipeline step | Person-worn reference | Product-photo reference | Can it be reused? |
|---|---|---|---|
| Clothing segmentation | Human parser (SCHP-LIP), trained on people | No person, flat or ghost-mannequin layout, plain background. Human parsers are **not** trained for this and their output is not trustworthy | No. Needs background removal plus a garment category |
| C_seg representation | Garment as worn by a body: 3-D shape, sleeves around arms | Flat layout in product-photo proportions (sleeves spread out, no body occlusion) | **Different distribution**: same label set, different geometry |
| Warp (C_seg + B_seg → C'_seg) | Trained on worn→worn pairs | Needs to learn flat→worn shape changes | Architecture reusable, **weights not**. Needs training on product↔worn pairs |
| Alignment (correspondence) | Worn texture → worn layout: similar geometry | Flat texture → worn layout: large deformation, perspective, folds | Architecture reusable. Must be trained or fine-tuned on product↔worn pairs |
| Texture/SPADE generator | Learns to render garments on people | Same output domain | Probably reusable with fine-tuning (untested) |
| Garment areas not visible | Back of garment, occluded areas | Product shot usually shows the full front; the inside/back is missing | Both need the generator to invent hidden regions |

## 2. What is implemented (Phase 3)

`src/data/reference.py`:

* `remove_plain_background()`: estimates the border colour (median), thresholds colour distance,
  keeps the largest connected component, fills holes (so a background-coloured logo stays part of
  the garment), and applies light morphological smoothing. It refuses images whose border is not
  uniform.
* `GarmentOnlyPreprocessor`: needs a garment category from the user (`top | outer | dress | skirt | pants`)
  and labels the mask with that category in `garment12`. Output: `PreparedReference(validated=False)`.
* `try_on(..., reference_mode="garment_only")` raises `StageNotAvailableError` after Stage A.

Tested only on synthetic drawings of product photos (`tests/test_reference_and_pipeline.py`).
**Not tested on any real product photo.**

Known limitations: no sleeve/part decomposition (a T-shirt is one region, so the warp cannot treat sleeves separately); textured or gradient backgrounds, shadows, and garments the same colour as the background all fail; no orientation normalisation; no handling of ghost-mannequin inner necklines.

## 3. Data needed for real support

DeepFashion-MultiModal has **no** product images and no product↔worn pairs, so this mode cannot be
trained or evaluated on it.

| Dataset | Contents | Access / licence | Fit |
|---|---|---|---|
| **VITON-HD** | 13,679 pairs (11,647 train / 2,032 test) of frontal women and **top** product photos, 1024×768; preprocessed parsing / pose / cloth-mask files (exact folder contents to be checked on download) | Google Drive link; **CC BY-NC 4.0**, research only | Best first choice: freely downloadable. Tops only, frontal poses only |
| **DressCode** | 53,792 garments / 107,584 images: upper, lower, dresses; 1024×768; keypoints, SCHP-ATR label maps, DensePose | Request form, **institutional email required**, signed agreement, not released to companies | More categories. Access depends on approval |
| DeepFashion In-shop | worn images; whether it has "flat" product views is **unverified** | agreement + password | Check after access is granted |

## 4. Plan for an experimental garment-only branch (after the person-worn baseline works)

1. Pick a dataset (VITON-HD is the realistic option).
2. Build product→C_seg using the dataset's own cloth masks, then evaluate `remove_plain_background`
   against them (IoU). That gives the first valid measurement for this mode.
3. Train a separate warp model: input = flat C_seg + target B_seg, target = worn garment parsing from the
   dataset. Same dual-path U-Net code, separate weights and config.
4. Fine-tune the alignment and SPADE stages on product→worn pairs.
5. Evaluate this mode **separately** from person-worn mode (different test sets, so the numbers are not comparable).
