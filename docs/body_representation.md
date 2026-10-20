# Body representation (B_seg) for the warp network

**Baseline decision (2026-10-20, by the user):** SCHP trained on Pascal-Person-Part (7 classes) is the
body representation for Phase 4. It is selected by `segmentation.body_backend: schp_pascal` and
`labels.body_label_set: pascal_person_part`. DensePose (`annotation_densepose` + `body15`) remains
available as a controlled experiment (Experiment A) by changing only these two settings.

## 1. The 7-part label set

| id | label | covers | what is merged or missing compared with NBF's 12 parts (paper) |
|---|---|---|---|
| 0 | background | everything that is not the person | — |
| 1 | head | face + hair + neck region | the hair shape is part of "head" |
| 2 | torso | chest, belly, hips | — |
| 3 | upper_arms | both upper arms | **no left/right** |
| 4 | lower_arms | both forearms **+ hands** | no left/right; hands are not separate |
| 5 | upper_legs | both thighs | no left/right |
| 6 | lower_legs | both shins **+ feet** | no left/right; feet are not separate |

Source of the names: SCHP `simple_extractor.py`. How the other body label sets map onto these parts
(all conversions are tested in `tests/test_labels.py`):

* `densepose → pascal_person_part`: torso 1,2 → torso; hands 3,4 → lower_arms; feet 5,6 → lower_legs;
  upper/lower legs and arms → the matching part; head 23,24 → head.
* `body15 → pascal_person_part`: the same mapping with left and right merged.
* There is **no** conversion from `pascal_person_part` to `body15`, because left/right cannot be recovered.

The warp network receives B_seg as a one-hot tensor with **7 channels** (15 for body15). The channel
count is read from the label set, so switching representations needs no code change.

## 2. Measured property: SCHP-Pascal outlines the clothed person

On the five user photos (`test_photos/`, 2026-10-20, CPU), we compared the silhouette from SCHP-Pascal
with the silhouette from SCHP-LIP. No ground truth was available; these numbers show agreement, not accuracy.

| Photo | IoU(PPP person, LIP person) | Upper-garment pixels labelled torso/arms by PPP |
|---|---|---|
| T-shirt + jeans | 0.963 | 96% |
| T-shirt + jeans, side pose | 0.944 | 95% |
| Patterned shirt | 0.969 | 94% |
| Long maxi dress | **0.393** | **34%** |
| Jacket over shirt | 0.961 | 96% |

Interpretation: SCHP-Pascal assigns body-part labels to the **clothed** silhouette. A loose jacket widens
the "torso". A long dress is mostly labelled *background*, so the legs disappear. Paper Sec. 4.3.1 meant
B_seg to describe the body and C_seg to describe the garment. With this backend, B_seg also partly
describes the target's current garment.

## 3. Effects on garment warping

| Effect | Why | Consequence | Mitigation in Phase 4 |
|---|---|---|---|
| **Silhouette shortcut** | Training targets always show the target person wearing the target garment, and B_seg traces that garment's outline | The network can learn to copy B_seg's outline and ignore C_seg. Transferring a dress onto a person in jeans would then come out jeans-shaped | (a) random per-part dilation/erosion of B_seg during training (`training.warp_body_aug`), so the outline is less reliable; (b) a **swap diagnostic** at validation that pairs C_seg from one item with B_seg from another and measures whether the predicted garment classes follow C_seg (`class_fidelity`) |
| Missing legs under long garments | PPP labels dress-covered legs as background | No leg pose for targets in long dresses or skirts; pants cannot be placed correctly | none in the baseline; DensePose (Experiment A) estimates the body under clothing |
| No left/right | merged parts | Ambiguous when arms cross or in side views; sleeve order is inferred from position | none needed for frontal poses; DensePose has left/right |
| Hands inside lower_arms, feet inside lower_legs | merged parts | Sleeve and trouser ends must be inferred at the lower end of the part; shoe placement too | learnable from data |
| Coarse torso / leg boundary | one hip line between torso and upper_legs | The waistline position of tops and pants depends on where PPP puts it | learnable; shown in qualitative checks |
| Head includes hair | — | Not used for garments; identity comes from the target's LIP parse (paper Sec. 4.2) | — |

## 4. How the DensePose experiment will be run (later)

Change only `segmentation.body_backend: annotation_densepose` and `labels.body_label_set: body15`
(or `pascal_person_part`, to isolate "clothing-independent body" from "more parts"). This requires staging
the DensePose archive (+5.6 GB). Keep the split, resolution, losses and number of steps fixed. Compare
val mIoU, swap `class_fidelity` and the qualitative grids. Caveat: inference on new photos would then need
a DensePose model, which is not integrated yet.
