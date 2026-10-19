# Dress-up virtual try-on (research implementation)

A student research project that re-implements, as faithfully as the paper and available resources
allow, the three-stage virtual try-on pipeline from

> H. Ghodhbani et al., **"Dress-up: deep neural framework for image-based human appearance transfer"**,
> *Multimedia Tools and Applications* 82 (2023) 23151–23178 (`researchpaper.pdf`).

No official Dress-up code or weights exist. Every component is either re-implemented, replaced by
a documented pretrained substitute, or not yet implemented. See `docs/implementation_deviations.md`.

## Current status: end of Phase 3

| Stage | What it does | Status |
|---|---|---|
| A. Pre-treatment | reference photo → clothing map **C_seg**; target photo → body-part map **B_seg** | ✅ implemented (pretrained SCHP parsers, DFMM annotations) |
| B. Warp | dual-path U-Net: C_seg + B_seg → garment layout on the target body **C'_seg** | ⏳ Phase 4 |
| C. Texture | correspondence alignment + SPADE generator → final image **I_G** | ⏳ Phase 5 |
| Garment-only references (product photos) | not in the paper | 🧪 experimental preprocessing only |

**There is no try-on output yet.** `try_on()` runs Stage A and then stops with an explicit
`StageNotAvailableError`. It never returns a fake result.

## How the paper works (short version)

1. **Pre-treatment:** human parsers turn both photos into label maps. Labels carry layout but no texture.
2. **Warp:** a network redraws the reference's *clothing label map* in the target's pose. Predicting a
   label map is easier than predicting pixels.
3. **Texture:** for every location of the new layout, find the matching location in the reference photo
   (feature correlation + softmax), build a roughly aligned "style image", and let a SPADE generator
   render the final photo from it.

Details, equations and open questions: `docs/paper_analysis.md`.

## Pretrained inference vs training

* **Pretrained components** (used as-is): the SCHP human parsers for Stage A.
* **Must be trained by this project:** the warp network and the texture networks. No compatible
  pretrained weights exist. Results from our training will **not** reproduce the paper's numbers (the paper used ≥4 GPUs and an undocumented protocol).

## Quick start

### Local CPU smoke test (no downloads)

```bash
pip install -r requirements-dev.txt
```
```bash
python -m pytest
```
```bash
python scripts/prepare_dataset.py --config configs/colab_smoke_test.yaml
```

The smoke test uses a **synthetic cartoon fixture** that mimics the dataset layout. It tests the
code paths only; never report metrics from it.

### Google Colab

1. Put the code where Colab can reach it: push this repository to GitHub (private is fine), **or**
   zip the project folder as `virtual-tryon.zip` and upload it to `MyDrive/VirtualTryOn/code/`.
2. Open `notebooks/virtual_try_on_colab.ipynb` in Colab (File → Upload notebook, or open it from GitHub).
3. Runtime → Change runtime type → **GPU** (T4 is enough for Phase 3; CPU works for the dry run).
4. Edit the **Settings** cell: `REPO_URL` (or leave it empty when using the zip), `DRY_RUN`, `ALLOW_WEIGHT_DOWNLOAD`.
5. Run all cells. With `DRY_RUN=True` everything runs on the synthetic fixture.
6. For real data: download the dataset (below), set `DRY_RUN=False`, and run again.

**Connecting Drive:** Section 3 calls `drive.mount()`, which asks for authorisation. Storage layout:
`docs/data.md`. No secrets or tokens are stored anywhere in the project.

### Dataset paths

Default: `<drive_root>/datasets/DeepFashion-MultiModal`. Change it with `data.root` in a config or
an override such as `EXTRA_OVERRIDES = ["data.root=/content/dfmm"]`. If the unzipped folder names
differ, set `data.layout.images_dir` / `parsing_dir` / `densepose_dir`. `scripts/prepare_dataset.py --inspect` shows what is there.

### Pretrained weights (SCHP)

| key | file | labels | size |
|---|---|---|---|
| `schp_lip` | exp-schp-201908261155-lip.pth | 20 LIP | ~255 MiB |
| `schp_atr` | exp-schp-201908301523-atr.pth | 18 ATR | ~255 MiB |
| `schp_pascal` | exp-schp-201908270938-pascal-person-part.pth | 7 body parts | ~255 MiB |

Source: https://github.com/PeikeLi/Self-Correction-Human-Parsing (MIT). Either set
`segmentation.allow_weight_download=true` (uses `gdown`; a `.provenance.json` with SHA-256 is written
next to the file) or copy the files to `<drive_root>/pretrained/schp/` yourself. Then verify:

```bash
python scripts/verify_schp_checkpoint.py --weights schp_lip --checkpoint /content/drive/MyDrive/VirtualTryOn/pretrained/schp/exp-schp-201908261155-lip.pth --images a.jpg b.jpg
```

## Configuration

`src/config.py` defines every option with its default. YAML files in `configs/` override them:

| file | purpose |
|---|---|
| `colab_smoke_test.yaml` | synthetic fixture, CPU, no downloads |
| `colab_baseline.yaml` | Mode B: DFMM, 128 px, single GPU |
| `extended_training.yaml` | Mode C: 256 px (inherits the baseline via `_base_`) |

Overrides: `load_config(path, ["data.image_size=[256,256]", "training.batch_size=2"])`. Unknown keys are errors.

## Training, resuming, inference and evaluation

* **Training each stage / resuming:** Phases 4–5. They will be resumable from step-based checkpoints in `checkpoints/<experiment>/`.
* **Inference:** `try_on(reference, target, cfg, stop_after="pretreatment")` returns C_seg and B_seg.
* **Evaluation now:** `evaluate_clothing_parser()` measures a parser against DFMM's manual parsing (pixel
  accuracy and mIoU in `garment12`, background excluded from mIoU). Image metrics (SSIM, LPIPS, FID) come in Phase 7.

## GPU and storage requirements

* Phase 3: SCHP inference fits easily on a T4 (the model is 66.7M parameters). CPU works but is slow (seconds per image).
* DFMM: about 11 GB for images + DensePose, about 90 MB for parsing. The free Drive tier is 15 GB; see `docs/data.md`.

## Common errors

| Error | Meaning / fix |
|---|---|
| `DatasetUnavailableError` | dataset folder missing; the message lists the download steps |
| `FileNotFoundError ... images_dir` | folder names differ; run `--inspect` and set `data.layout.*` |
| `WeightsUnavailableError` | weights missing; allow the download or copy the file |
| `CheckpointMismatchError` | wrong or corrupted file, or a label-set mismatch (e.g. LIP weights with an ATR backend) |
| `LabelError ... colour-coded` | an RGB visualisation PNG was loaded instead of the raw label PNG |
| `LabelError ... cannot produce body15` | `schp_pascal` gives 7 parts; set `labels.body_label_set=pascal_person_part` |
| `SplitError ... fingerprint` | the image set changed; delete the old split file to regenerate it |
| `StageNotAvailableError` | warp/texture not implemented yet (expected in Phase 3) |
| `GPUUnavailableError` | select a GPU runtime or use `experiment.device=auto` |

## Known limitations

* Parsers and body model are substitutes (SCHP instead of LIP-JppNet, DensePose/SCHP-PPP instead of NBF).
* The real SCHP checkpoints and the real dataset have not been run yet (see `docs/implementation_status.md`).
* DFMM has no product photos, so garment-only mode cannot be trained on it (`docs/clothing_only_mode.md`).
* The paper's evaluation protocol is undocumented; its numbers are kept only as published references.

## Licences

Code in this repository: your choice (not yet set). Third-party pieces: SCHP architecture port (MIT,
© 2020 Peike Li; affine helper MIT © Microsoft). Data and weights are **not** redistributed:
DeepFashion / DeepFashion-MultiModal are non-commercial research only; VITON-HD is CC BY-NC 4.0;
SegFormer weights fall under the NVIDIA SegFormer licence (non-commercial).

## Documentation

`docs/paper_analysis.md` (paper audit + Phase 2–3 update) · `docs/architecture.md` · `docs/data.md` ·
`docs/implementation_deviations.md` · `docs/implementation_status.md` · `docs/clothing_only_mode.md`
