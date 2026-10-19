# Data and storage guide

## Where files live

| Location | What | Lifetime | Size (images + parsing) |
|---|---|---|---|
| **Google Drive** `MyDrive/VirtualTryOn/` | DFMM **zip archives** (never extracted here), SCHP weights, caches, splits, checkpoints, outputs | permanent | ~5.5 GB of archives + ~0.5 GB weights |
| **Colab local disk** `/content/data/DeepFashion-MultiModal/` | extracted working copy that the code reads | deleted when the runtime ends; re-extracted each session | ~5.5 GB |
| **Colab local disk** `/content/virtual-tryon/` | the code (git clone or zip) | per session | small |
| **Your PC** | code, docs, synthetic tests | permanent | **no dataset** |

Why archives rather than extracted files in Drive: 44k small files on the Drive mount (`/content/drive`)
are slow to list and read, and a training loop over them would bottleneck on I/O. One large zip
reads quickly, and extracting it to the local disk takes a few minutes per session (not yet measured).

```
MyDrive/VirtualTryOn/
├── datasets/DeepFashion-MultiModal/archives/
│   ├── images.zip  (+ images.provenance.json: source id, bytes, sha256, date)
│   ├── parsing.zip (+ provenance)
│   └── keypoints.zip (+ provenance)
├── pretrained/schp/     SCHP weights + provenance
├── checkpoints/<experiment>/
├── cache/{index,splits,reports,segmentation}/
├── outputs/<experiment>/
└── logs/

/content/data/DeepFashion-MultiModal/      (Colab local disk, rebuilt each session)
├── images/      44,096 *.jpg
├── segm/        12,701 *_segm.png
└── keypoints/   keypoints_loc.txt, keypoints_vis.txt
```

## "Images + parsing" preset: exactly what is downloaded

| Archive | DFMM README size | Files | Used for |
|---|---|---|---|
| `image` (id `1U2PljA7NE57jcSSzPs21ZurdIPXdYZtN`) | ~5.4 GB | 44,096 JPG, 750×1101 | all images (only the 12,701 with parsing are used for training) |
| `parsing` (id `1r-5t-VgDaAQidZLVgWtguaG7DvMoyUv9`) | ~90 MB | 12,701 PNG | manual 24-class labels → C_seg ground truth |
| `keypoints` (id `1ZXdOQI-d4zNhqRJdUEWSQvPwAtLdjovo`) | 956 KB | 2 TXT | not used yet; small |
| *not included:* `DensePose` | ~5.6 GB | 44,096 PNG | body15 maps (add later with `--parts densepose`) |

**Not included:** labels (shape/fabric/colour) and text descriptions, which this project does not use.

**Consequence:** without DensePose there are no body15 ground-truth maps. `colab_baseline.yaml` sets
`data.require_densepose: false`, so indexing, splits and SCHP-LIP evaluation against the manual parsing
all work. `PreTreatmentDataset` (which needs body maps) stops with a clear message. For training,
either stage DensePose later or build B_seg from cached SCHP-Pascal output. That is the Phase 4
body-representation decision.

## Transfer plan (one copy per location, no duplicates)

1. **First Colab session** (GPU not needed): Settings `DRY_RUN=False`, `FETCH_DFMM=True`,
   `DFMM_ALLOW_DOWNLOAD=True`, then run Section 6a. Equivalent command:
   `python scripts/fetch_dfmm.py --config configs/colab_baseline.yaml --download`.
   - gdown copies each archive from the authors' Drive **directly into your Drive** `archives/` folder.
     No intermediate copy is kept.
   - It writes a provenance JSON (bytes, SHA-256), then extracts to the Colab local disk.
   - If Google reports a quota error ("too many users"): open the README link in a browser, click
     "Make a copy" (a server-side copy, no download through your PC), and move the copy into `archives/`.
     The script detects files named `images*`, `parsing*`/`segm*`, `keypoints*`.
2. **Every later session:** `FETCH_DFMM=True`, `DFMM_ALLOW_DOWNLOAD=False`. Section 6a only extracts
   (skipped if the local copy is still present in the same runtime).
3. **Your PC:** never downloads the dataset. Tests use the synthetic fixture.

Space needed: Google Drive ~5.5 GB for archives plus ~0.5 GB for SCHP weights. Free Google accounts
have 15 GB shared with Gmail and Photos, so check your free space first. Colab local disk: ~5.5 GB
for the extracted copy (the script checks free space before extracting).

### SCHP weights

The two SCHP files are currently only on your PC, in `outputs/local_drive/pretrained/schp/` (511 MB). Colab
needs them in `MyDrive/VirtualTryOn/pretrained/schp/`. Either upload these two files (and their
`.provenance.json`) to that folder, or run the notebook once with `ALLOW_WEIGHT_DOWNLOAD=True` so Colab
downloads them straight into Drive. After that, the PC copies can be deleted if you don't need local CPU parsing.

## Licence

DeepFashion-MultiModal: non-commercial research only, no redistribution. Keep the Drive folder private
(do not share links to `archives/`).

## Facts and assumptions

| Item | Status |
|---|---|
| File IDs, sizes, counts | DFMM README |
| Archive format is zip | **assumed** (DiOr's script unzips the image archive); non-zip files are rejected with a message |
| Folder names inside archives | **unknown**; `stage_archive` finds the folder with the most matching files and renames it |
| `<stem>_segm.png` naming | Text2Human loader; checked during staging (pattern match + file count vs README) |
| In-shop item IDs in names | DiOr download script; unparseable names are reported |

## Splits and leakage

Images are grouped by **item ID** (one garment). Whole groups are assigned to train/val/test with a
seeded shuffle (`src/data/splits.py`). The split file stores a fingerprint of the image list and
refuses to load against a different image set. `build_same_item_pairs()` creates pairs only within one split.
