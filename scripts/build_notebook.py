"""Builds notebooks/virtual_try_on_colab.ipynb.

    python scripts/build_notebook.py notebooks/virtual_try_on_colab.ipynb

Edit this file, not the .ipynb, so the notebook stays reproducible from version control.
"""
import sys
import nbformat as nbf

out = sys.argv[1]
cells = []
_guard = {"on": False}  # code cells of Sections 6a-14 are skipped when DEMO_ONLY=True (set in Section 0)


def md(s):
    head = s.lstrip("\n")
    if head.startswith("## 6a."):
        _guard["on"] = True
    if head.startswith("## 16."):
        _guard["on"] = False
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    s = s.strip("\n")
    if _guard["on"]:
        body = "\n".join(("    " + l) if l.strip() else l for l in s.splitlines())
        s = ('if globals().get("DEMO_ONLY"):\n    print("Skipped (DEMO_ONLY=True: only setup and Section 16 run).")\n'
             'else:\n' + body)
    cells.append(nbf.v4.new_code_cell(s))

md(r"""
# Dress-up virtual try-on — Colab notebook

Research implementation of **"Dress-up: deep neural framework for image-based human appearance transfer"**
(Ghodhbani et al., *Multimedia Tools and Applications* 2023).

**Implementation status (Phase 3):**

| Stage | Status |
|---|---|
| A. Pre-treatment (clothing + body segmentation) | implemented: pretrained SCHP substitutes for LIP-JppNet / NBF, DFMM annotations |
| B. Warp network (dual-path U-Net) | implemented and unit-tested (Phase 4); **not yet trained on real data** |
| C. Texture network (alignment + SPADE) | implemented and unit-tested (Phase 5); **not yet trained on real data** |
| End-to-end try-on (Phase 6) | implemented and tested; trained models from Sections 10/10b |
| Garment-only (product photo) references | experimental preprocessing only; no compatible warp model |

Every section can be read on its own. Run the cells from top to bottom. Nothing heavy runs automatically. Downloads only happen if you allow them in the settings cell.
""")

md("## 0. Settings — the only cell you normally edit")
code(r"""
# --- where the code is -------------------------------------------------------------
REPO_URL = ""                       # e.g. "https://github.com/<you>/virtual-tryon.git" (leave "" if you upload a zip)
# Private repo: add a Colab secret named GITHUB_TOKEN (key icon in the left sidebar). Never paste tokens here.
CODE_ZIP_IN_DRIVE = "code/virtual-tryon.zip"   # alternative: zip of the repo stored under DRIVE_PROJECT_ROOT
COLAB_PROJECT_DIR = "/content/virtual-tryon"

# --- storage -------------------------------------------------------------------------
USE_DRIVE = True                    # mount Google Drive for datasets / weights / outputs
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/VirtualTryOn"

# --- what to run ---------------------------------------------------------------------
DRY_RUN = True                      # True: synthetic fixture, CPU is fine, no downloads
REQUIRE_GPU = True                  # False: allow a CPU runtime for GPU-free work (pair analysis, export) to save T4 quota
CONFIG_FILE = "configs/colab_smoke_test.yaml" if DRY_RUN else "configs/colab_baseline.yaml"
ALLOW_WEIGHT_DOWNLOAD = False       # True: fetch SCHP weights (~255 MiB each) from the authors' Google Drive
EXTRA_OVERRIDES = []                # e.g. ["data.image_size=[256,256]", "data.max_samples=200"]

# --- dataset (DeepFashion-MultiModal, real data only) ----------------------------------
FETCH_DFMM = False                  # extract the archives from Drive to Colab's local disk (needed every session)
DFMM_ALLOW_DOWNLOAD = False         # first session only: download missing archives into Drive (~5.5 GB)
DFMM_PARTS = "images_parsing"       # images + parsing + keypoints; "all" adds DensePose (~5.6 GB more)
ANALYZE_PAIRS = False               # measure outfit consistency of same-item photo pairs (Section 6b, ~minutes)

# --- Phase 4: warp network ---------------------------------------------------------------
PRECOMPUTE_BODY_MAPS = False        # once: SCHP-Pascal over all 12,701 annotated images (GPU), archived to Drive
RUN_WARP_TRAINING = False           # start/resume warp training (Section 10)
WARP_MAX_STEPS = 2000               # total optimiser steps; re-running resumes from the last checkpoint
EVAL_WARP_CHECKPOINTS = []          # e.g. ["best", "last"]: score checkpoints on the WHOLE val split (no training)

# --- Phase 5: texture network ---------------------------------------------------------------
ALLOW_VGG_DOWNLOAD = False          # once: VGG-19 ImageNet weights (~548 MB) for the perceptual loss, cached in Drive
RUN_TEXTURE_TRAINING = False        # start/resume texture training (Section 10b)
TEXTURE_MAX_STEPS = 2000            # total optimiser steps; re-running resumes
EVAL_TEXTURE_CHECKPOINTS = []       # e.g. ["best", "last"]

# --- time budgets: training stops cleanly (validate + save) after this many minutes; None = no limit ---
WARP_TIME_LIMIT_MIN = None
TEXTURE_TIME_LIMIT_MIN = None

# --- Phase 6: end-to-end try-on ------------------------------------------------------------------
RUN_TRYON_EVAL = False             # paired end-to-end evaluation + demo grids (Section 11)
TRYON_WARP_CKPT = None             # None = from the config (evaluation.warp_checkpoint, normally "last"); or last | best | path
TRYON_TEXTURE_CKPT = None
TRYON_EVAL_LIMIT = None            # None = all validation pairs
# --- Phase 7: final evaluation (docs/final_evaluation.md) ----------------------------------------------
RUN_FINAL_EVAL = False             # Section 12: parser, warp, paired SSIM/L1/LPIPS, FID/KID, timing
FINAL_SPLIT = "val"                # "val" for checks; "test" ONLY for the one final, logged run
FINAL_PARTS = ["parser", "warp", "paired", "realism", "noise_floor", "timing"]
FINAL_LIMIT = None                 # None = everything (official); a number = quick check, marked NOT official
ALLOW_METRIC_DOWNLOAD = False      # once: LPIPS AlexNet (~233 MB) + FID Inception (~92 MB), cached in <drive>/pretrained/torch
EXPORT_RESULTS = True              # copy all logs/metrics/grids of this session to <drive>/report_exports (Section 14)
SECURE_RESULTS = True              # verify every checkpoint loads, save small generator-only backups (Section 14)
PRUNE_SNAPSHOTS = False            # True: delete redundant step_*.pt snapshots after verification (keeps last.pt + best.pt)
FLUSH_DRIVE_AT_END = True          # push all pending writes to Drive; after this it is safe to delete the runtime

# --- Paper-style benchmark: standard DeepFashion pose-transfer pairs (Section 12b) ----------------------
RUN_PT_LEAKAGE = False             # which standard test pairs we may score (CPU, seconds)
RUN_PT_BENCHMARK = False           # whole-image SSIM / LPIPS / FID on the scorable pairs (GPU, ~20-40 min)
ALLOW_PAIR_DOWNLOAD = False        # once: the two pair lists (0.8 + 9.6 MB, approved 2026-10-26), into <drive>/datasets
PT_LIMIT = None                    # None = all scorable pairs; a number = quick check (not reported)
# --- DeepFashion In-shop inspection (Section 12c; needs the Drive shortcut, text files only) -------------
RUN_INSHOP_INSPECTION = False
INSHOP_DIR = "/content/drive/MyDrive/VirtualTryOn/datasets/In-shop Clothes Retrieval Benchmark"
# --- Section 16: final model demo and results (presentation) ------------------------------------------------
DEMO_ONLY = False                  # True: Run all does only the setup (Sections 0-5) and Section 16; no dataset needed
RUN_FINAL_DEMO = False             # load the final 256 px models from Drive (checkpoints/full_256/warp, v2_256/texture_ft)
RUN_SHOW_RESULTS = False           # print all metric tables, recomputed from report/data/results (no GPU, no data)
RUN_LIVE_BENCHMARK = False         # score the final model live on LIVE_PAIRS standard pairs (needs FETCH_DFMM=True)
LIVE_PAIRS = 100
""")

md(r"""
## 1. Project overview

**Inputs:** `I_C`, the reference image showing the garment, and `I_B`, the target person. **Output:** `I_G`, the target person wearing the garment.

```
 I_C ──► clothing parser ──► C_seg ─┐
                                    ├─► [B] warp: dual-path U-Net ──► C'_seg ─┐
 I_B ──► body-part parser ──► B_seg ┘        (Phase 4)                       ├─► [C] alignment (corr. + softmax)
                                                                             │       + SPADE generator ──► I_G
 I_C ────────────────────────────────────────────────────────────────────────┘       (Phase 5)
        [A] pre-treatment (this phase)
```

* **C_seg**: the garment layout in the `garment12` label set. Only the classes to be transferred are kept.
* **B_seg**: body parts that do not depend on clothing. `body15` comes from DensePose; `pascal_person_part` comes from SCHP.
* Paper details, deviations and open questions: `docs/paper_analysis.md` and `docs/implementation_deviations.md`.
""")

md("## 2. Runtime verification")
code(r"""
import os, sys, platform
import torch
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
print("Python :", sys.version.split()[0], "|", platform.platform())
print("PyTorch:", torch.__version__, "| CUDA build:", torch.version.cuda, "| in Colab:", IN_COLAB)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU    : {p.name} | {total/2**30:.1f} GB total, {free/2**30:.1f} GB free | device cuda:0")
else:
    print("GPU    : none -> device cpu")
    if not DRY_RUN and REQUIRE_GPU:
        raise SystemExit("This session needs a GPU runtime (Runtime > Change runtime type > GPU). For GPU-free work "
                         "(pair analysis, exporting) set REQUIRE_GPU=False; for the CPU checks set DRY_RUN=True.")
    if not REQUIRE_GPU:
        print("CPU session: only run GPU-free steps (keep all training / evaluation flags False).")
""")

md("## 3. Google Drive setup\nMounting asks you to authorise access in a pop-up. Datasets, weights, checkpoints, caches and outputs live in separate folders under `DRIVE_PROJECT_ROOT` (see `src/utils/paths.py`).")
code(r"""
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")          # interactive authorisation
    os.makedirs(DRIVE_PROJECT_ROOT, exist_ok=True)
    print("Drive project root:", DRIVE_PROJECT_ROOT)
else:
    print("Drive not mounted (not in Colab or USE_DRIVE=False); using local folders.")
""")

md("## 4. Code and dependencies\nPyTorch is **not** reinstalled. Colab's build already matches its CUDA driver. Only the missing packages from `requirements.txt` are installed.")
code(r"""
import subprocess, zipfile
from pathlib import Path

def locate_project():
    here = Path.cwd()
    for cand in (here, here.parent):
        if (cand / "src" / "config.py").is_file():
            return cand
    if IN_COLAB:
        target = Path(COLAB_PROJECT_DIR)
        if REPO_URL:
            url = REPO_URL
            try:
                from google.colab import userdata
                token = userdata.get("GITHUB_TOKEN")
            except Exception:  # secret missing or notebook access not granted -> public repo
                token = None
            if token:
                url = REPO_URL.replace("https://", f"https://x-access-token:{token}@", 1)
            if (target / ".git").is_dir():
                cmd = ["git", "-C", str(target), "pull", "--ff-only", url]
            else:
                cmd = ["git", "clone", "--depth", "1", url, str(target)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            msg = (r.stdout + r.stderr).replace(token, "***") if token else r.stdout + r.stderr
            print(msg.strip())
            if r.returncode != 0:
                raise RuntimeError("git failed. Private repo? Add the GITHUB_TOKEN secret and allow notebook access.")
            return target
        if (target / "src").is_dir():
            return target
        zip_path = Path(DRIVE_PROJECT_ROOT) / CODE_ZIP_IN_DRIVE
        if zip_path.is_file():
            zipfile.ZipFile(zip_path).extractall(target.parent)
            if (target / "src").is_dir():
                return target
    raise FileNotFoundError("Project code not found. Set REPO_URL, or upload a zip of the repository to "
                            f"{DRIVE_PROJECT_ROOT}/{CODE_ZIP_IN_DRIVE} (top-level folder 'virtual-tryon').")

PROJECT_DIR = locate_project()
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
# If this runtime imported the project before (e.g. a re-used session), Python would keep using the OLD
# modules even after `git pull`. Drop them so the next import loads the code that is now on disk.
stale = [m for m in sys.modules if m == "src" or m.startswith("src.")]
for m in stale:
    del sys.modules[m]
if stale:
    print(f"reloaded project code ({len(stale)} previously imported modules discarded)")
rev = subprocess.run(["git", "-C", str(PROJECT_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
print("project:", PROJECT_DIR, "| commit:", rev.stdout.strip() or "unknown (not a git checkout)")
""")
code(r"""
# Version / import verification
import importlib
for mod in ["numpy", "PIL", "cv2", "scipy", "matplotlib", "yaml", "tqdm"]:
    m = importlib.import_module(mod)
    print(f"{mod:10s} {getattr(m, '__version__', '?')}")
from src.utils.device import runtime_report, format_runtime_report
print(format_runtime_report(runtime_report()))
""")

md(r"""
## 5. Configuration
All settings live in `src/config.py` (typed dataclasses with defaults) and in the YAML files in `configs/`.

**Limited GPU memory?** Lower these, roughly in order: `data.image_size` (256 → 128), `training.batch_size`. Then raise `training.grad_accum_steps` to keep the effective batch size. Later phases will add `model.texture_corr_size` (64 → 32; correlation memory grows with the 4th power of this size) and the model widths.
""")
code(r"""
from src.config import load_config, save_config
from src.utils.reproducibility import seed_everything
from src.utils.paths import ProjectPaths

overrides = list(EXTRA_OVERRIDES)
if IN_COLAB and USE_DRIVE and not DRY_RUN:
    overrides.append(f"paths.drive_root={DRIVE_PROJECT_ROOT}")
if ALLOW_WEIGHT_DOWNLOAD:
    overrides.append("segmentation.allow_weight_download=true")
if ALLOW_VGG_DOWNLOAD:
    overrides.append("training.allow_vgg_download=true")
if ALLOW_METRIC_DOWNLOAD:
    overrides.append("evaluation.allow_metric_download=true")
cfg = load_config(CONFIG_FILE, overrides)
paths = ProjectPaths.from_config(cfg).ensure()
seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
print("experiment :", cfg.experiment.name, "| mode:", cfg.experiment.run_mode, "| dry run:", cfg.experiment.dry_run)
print("dataset    :", cfg.data.dataset, "at", cfg.data.root)
print("image size :", cfg.data.image_size, "| backends:", cfg.segmentation.clothing_backend, "/", cfg.segmentation.body_backend)
print("storage    :", paths.drive_root.resolve())
save_config(cfg, paths.experiment_dir("outputs", cfg.experiment.name) / "config_resolved.yaml")
# The Phase 7 baseline B0 is frozen: no training run can ever write into these folders (docs/experiments.md).
if not DRY_RUN:
    from src.training.checkpointing import freeze_run, is_frozen
    for _run in ("warp_pf", "texture_pf"):
        _d = paths.checkpoints / "baseline_128" / _run
        if (_d / "last.pt").is_file():
            freeze_run(_d, "B0 baseline: official Phase 7 evaluation 2026-10-22 (tag baseline-v1)")
            print("frozen baseline:", _d, is_frozen(_d))
""")

md(r"""
## 6a. Dataset acquisition (real data only)
* **Drive** (permanent): zip archives in `<drive>/datasets/DeepFashion-MultiModal/archives/`. They are downloaded once and never extracted inside Drive.
* **Colab local disk** (temporary): extracted to `/content/data/DeepFashion-MultiModal`, rebuilt each session. Training reads from here because it is much faster than the Drive mount.
* **Your PC:** nothing is downloaded.

The first time, set `FETCH_DFMM=True` and `DFMM_ALLOW_DOWNLOAD=True`. After that, `FETCH_DFMM=True` alone is enough.
""")
code(r"""
from src.data.fetch import fetch_and_stage, storage_plan
if DRY_RUN or not FETCH_DFMM:
    print("Skipped (DRY_RUN or FETCH_DFMM=False). Plan:")
    print(storage_plan(DFMM_PARTS, cfg.data.archive_dir, cfg.data.root))
else:
    fetch_and_stage(cfg, DFMM_PARTS, allow_download=DFMM_ALLOW_DOWNLOAD)
""")

md(r"""
## 6. Data preparation
This builds the index, checks the images and annotations, makes item-disjoint splits (the split file is reused on later runs) and shows samples.

* **DRY_RUN:** a **synthetic** cartoon fixture is generated. It is not real data and only tests the code path.
* **Real data:** download DeepFashion-MultiModal to `<drive>/datasets/DeepFashion-MultiModal` first (see the README). Then run `!python scripts/prepare_dataset.py --config configs/colab_baseline.yaml --inspect` once to confirm the folder names.
""")
code(r"""
from src.data.dataset import prepare_dataset, PreTreatmentDataset
prepared = prepare_dataset(cfg, progress=False)
if prepared.synthetic:
    print("*** SYNTHETIC FIXTURE - not real data ***")
print(prepared.report.summary())
for split, names in prepared.splits.items():
    s = prepared.stats[split]
    print(f"{split:5s}: {len(names):6d} images | {s['items']:5d} items | {s['items_with_2plus_images']} items with >=2 views")
""")
code(r"""
import matplotlib.pyplot as plt
from src.data.labels import get_label_set
from src.data.preprocessing import tensor_to_image
from src.inference.visualization import overlay, show_panels

G, B = get_label_set(cfg.labels.garment_label_set), get_label_set(cfg.labels.body_label_set)
from src.data.labels import convert_labels, colorize
from src.data.preprocessing import load_image, load_label_map
recs = prepared.records("train")[:2]
if not all(r.densepose for r in recs):   # "images + parsing" download: no body ground truth yet
    for r in recs:
        img = load_image(prepared.index.root / r.image)
        g = convert_labels(load_label_map(prepared.index.root / r.parsing), "dfmm", G.name)
        show_panels([("image", img), ("garment12 (GT parsing)", overlay(img, g, G, 0.7))], legends=[(G, g)], title=r.name)
train_ds = PreTreatmentDataset(prepared, "train", cfg) if all(r.densepose for r in recs) else []
for i in range(min(2, len(train_ds))):
    s = train_ds[i]
    img = tensor_to_image(s["image"]); g = s["garment"].numpy().astype("uint8"); b = s["body"].numpy().astype("uint8")
    assert img.shape[:2] == g.shape == b.shape, "image/label size mismatch"
    show_panels([("image", img), ("garment12 (GT parsing)", overlay(img, g, G, 0.7)), ("body (DensePose)", overlay(img, b, B, 0.7))],
                legends=[(G, g), (B, b)], title=s["name"])
plt.show()
""")

md(r"""
## 6b. Pair consistency (same item ID is not always the same outfit)
In DeepFashion an item ID names one product. The rest of the outfit can change between photos (e.g. the item is a beanie), and such cross-view pairs teach the networks impossible mappings. This cell measures, for every same-item pair, whether the garment classes match and how different the garment colours are (CIE76 ΔE). Results go to the pair table used by `data.pair_filter` (off by default) and to `outputs/<experiment>/pair_quality/`.
""")
code(r"""
if ANALYZE_PAIRS and not DRY_RUN:
    import importlib.util, json
    spec = importlib.util.spec_from_file_location("analyze_pairs", "scripts/analyze_pairs.py")
    ap_mod = importlib.util.module_from_spec(spec); sys.path.insert(0, "scripts"); spec.loader.exec_module(ap_mod)
    table, pq_dir, pq_summary = ap_mod.run(cfg)
    print(json.dumps(pq_summary, indent=1))
    display(Image.open(pq_dir / "pair_delta_e_hist.png")) if "Image" in globals() else None
else:
    print("Skipped (ANALYZE_PAIRS=False or DRY_RUN).")
""")

md(r"""
## 7. Pre-treatment inference (Stage A)
This runs the configured clothing and body backends on two images from the validation split: one is used as the reference and one as the target.

* With the smoke config, the backends read the dataset's own annotations, so no weights are needed.
* With the baseline config, `schp_lip` is used for clothing. It needs the SCHP LIP weights, about 255 MiB. Set `ALLOW_WEIGHT_DOWNLOAD=True`, or put the file in `<drive>/pretrained/schp/` yourself.
""")
code(r"""
from src.inference.pipeline import PreTreatment, try_on
pre = PreTreatment(cfg)
val = prepared.records("val") or prepared.records("train")
ref_rec = val[0]
tgt_rec = next((r for r in val if r.item_id != ref_rec.item_id), val[-1])
from src.data.preprocessing import load_image
ref_img, tgt_img = load_image(prepared.index.root / ref_rec.image), load_image(prepared.index.root / tgt_rec.image)
res = try_on(ref_img, tgt_img, cfg, reference_key=ref_rec.name, target_key=tgt_rec.name, stop_after="pretreatment",
             pretreatment=pre, save_dir=paths.experiment_dir("outputs", cfg.experiment.name) / "pretreatment_example")
show_panels([("I_C reference", ref_img), ("C_seg", G.palette_array()[res.reference.garment_map]),
             ("I_B target", tgt_img), ("B_seg", B.palette_array()[res.target.body_map])],
            legends=[(G, res.reference.garment_map), (B, res.target.body_map)], title="Stage A outputs")
plt.show()
print("notes:", res.notes or "none")
""")
code(r"""
# Parser accuracy against DFMM manual parsing (only meaningful for a network backend on real data)
from src.evaluation.evaluate import evaluate_clothing_parser
if pre.clothing.name.startswith("annotation"):
    print("Clothing backend reads the ground truth itself; evaluation skipped.")
else:
    report = evaluate_clothing_parser(pre.clothing, prepared, cfg, split="val", limit=50)
    print(f"{report['backend']}: pixel acc {report['pixel_accuracy']:.3f} | mIoU (no bg) {report['mean_iou']:.3f} on {report['n_images']} val images")
    print({k: (round(v, 3) if v is not None else None) for k, v in report["per_class_iou"].items()})
""")
code(r"""
# Optional: your own photos (needs network backends, e.g. clothing schp_lip + body schp_pascal with
# labels.body_label_set=pascal_person_part). Annotation backends only work on dataset images.
RUN_UPLOAD = False
if RUN_UPLOAD and IN_COLAB:
    from google.colab import files
    up = files.upload()
    (ref_name, ref_bytes), (tgt_name, tgt_bytes) = list(up.items())[:2]
    import io; from PIL import Image
    ref_u, tgt_u = load_image(Image.open(io.BytesIO(ref_bytes))), load_image(Image.open(io.BytesIO(tgt_bytes)))
    r = try_on(ref_u, tgt_u, cfg, stop_after="pretreatment", pretreatment=pre)
    show_panels([("reference", ref_u), ("C_seg", G.palette_array()[r.reference.garment_map]),
                 ("target", tgt_u), ("B_seg", B.palette_array()[r.target.body_map])],
                legends=[(G, r.reference.garment_map), (B, r.target.body_map)]); plt.show()
""")
code(r"""
# Experimental: garment-only (product photo) reference. Plain backgrounds only; user supplies the category.
# The result is a FLAT layout that the person-worn warp network cannot use (validated=False).
import numpy as np
from PIL import Image, ImageDraw
demo = Image.new("RGB", (120, 140), (245, 245, 245)); d = ImageDraw.Draw(demo)
d.polygon([(30, 20), (90, 20), (115, 50), (95, 60), (95, 125), (25, 125), (25, 60), (5, 50)], fill=(30, 60, 160))
ref_flat = pre.prepare_reference(np.asarray(demo), mode="garment_only", garment_category="top")
show_panels([("synthetic product photo", np.asarray(demo)), ("flat C_seg", G.palette_array()[ref_flat.garment_map])],
            legends=[(G, ref_flat.garment_map)], title=f"validated={ref_flat.validated}")
plt.show(); print(ref_flat.warnings)
""")

md(r"""
## 8. Warp stage, part 1: body maps for training (SCHP-Pascal, once)
Training needs a B_seg map for every annotated image. The network runs here **once** on the GPU (it does not run inside the data loader). The maps are written to the Colab local disk and zipped into Drive (`cache/body_maps_archives/schp_pascal.zip`). Later sessions only unzip them. This takes roughly tens of minutes on a T4 (not yet measured).

Why SCHP-Pascal maps follow the clothing outline, and what the training does about it: `docs/body_representation.md`.
""")
code(r"""
from src.data.body_maps import stage_body_maps, precompute_body_maps, archive_body_maps
print("restore from Drive:", stage_body_maps(cfg))
if PRECOMPUTE_BODY_MAPS and not DRY_RUN:
    recs = [r for s in ("train", "val", "test") for r in prepared.records(s)]
    print(precompute_body_maps(cfg, recs, prepared.index.root))   # resumable
    print("archived to", archive_body_maps(cfg))
else:
    print("Precompute skipped (PRECOMPUTE_BODY_MAPS=False or DRY_RUN).")
""")

md(r"""
## 8b. Warp stage, part 2: model checks (dual-path U-Net)
Dual-path U-Net (paper Fig. 8, Eqs. 1-4). This section checks the model before any training:
shapes, parameter count, finite losses on a real batch, and a short overfit test on one batch, where the loss must drop.
""")
code(r"""
from src.training.train_warp import WarpTrainer, _collate
from src.training.losses import segmentation_ce
trainer = WarpTrainer(cfg, prepared)
g = trainer.gen
print("generator params:", f"{sum(p.numel() for p in g.parameters())/1e6:.1f}M",
      "| discriminator:", f"{sum(p.numel() for p in trainer.disc.parameters())/1e6:.2f}M")
print("feature shapes:", g.feature_shapes())
print(f"train triplets: {len(trainer.train_ds)} ({trainer.train_ds.n_cross_view_available()} with a cross-view partner) | "
      f"val: {len(trainer.val_ds)} | swap pairs: {len(trainer.swap_ds)} | pairing: {cfg.training.warp_pairing}")
batch = _collate([trainer.train_ds[i] for i in range(2)])
print({k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")}, batch["mode"])
""")
code(r"""
# Overfit smoke test on ONE batch (CE only, a throw-away copy of the model). The loss must fall sharply.
import copy, torch
g_tmp = copy.deepcopy(trainer.gen).train()
opt = torch.optim.Adam(g_tmp.parameters(), lr=1e-3)
c, b, t = batch["cloth"].to(trainer.device), batch["body"].to(trainer.device), batch["target"].to(trainer.device)
for s in range(60):
    loss = segmentation_ce(g_tmp(c, b), t)
    assert torch.isfinite(loss), "non-finite loss"
    if s in (0, 19, 59): print(f"step {s+1}: CE {loss.item():.4f}")
    opt.zero_grad(); loss.backward(); opt.step()
del g_tmp, opt
if torch.cuda.is_available(): print(f"peak GPU memory: {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
""")

md(r"""
## 9. Texture network (Stage C): model checks
Alignment network (paper Eqs. 5-6) plus SPADE generator (Eq. 7); see `docs/texture_stage.md`.
Training input is **ground-truth** layouts (target photo's parsing) and an exemplar photo of the same garment. The warp network is not used here; its output feeds this stage at try-on time (Phase 6).
The perceptual loss needs VGG-19 ImageNet weights (~548 MB, BSD). Set `ALLOW_VGG_DOWNLOAD=True` once; they are cached in `<drive>/pretrained/torch`.
""")
code(r"""
from src.training.train_texture import TextureTrainer
from src.models.alignment_network import correlation_bytes
from src.data.texture_dataset import texture_collate
tex = TextureTrainer(cfg, prepared)
print("texture generator:", f"{sum(p.numel() for p in tex.gen.parameters())/1e6:.1f}M params",
      "| discriminator:", f"{sum(p.numel() for p in tex.disc.parameters())/1e6:.1f}M")
cs = cfg.model.texture_corr_size
print(f"correspondence {cs}x{cs}: M is {cs*cs}x{cs*cs}, ~{correlation_bytes(cs, cfg.training.batch_size)/2**20:.0f} MiB per batch (FP32 incl. backward)")
print(f"train targets: {len(tex.train_ds)} ({tex.train_ds.n_cross_view_available()} with a same-item exemplar) | val pairs: {len(tex.val_ds)}")
tb = texture_collate([tex.train_ds[i] for i in range(2)])
print({k: tuple(v.shape) for k, v in tb.items() if hasattr(v, "shape")}, tb["mode"])
""")
code(r"""
# One-batch overfit check (pixel L1 only, throw-away copy). L1 must fall clearly.
import copy, torch, torch.nn.functional as F
g_tmp = copy.deepcopy(tex.gen).train(); opt = torch.optim.Adam(g_tmp.parameters(), lr=1e-3, betas=(0.5, 0.999))
dev = tex.device
L, I, E, P = (tb[k].to(dev) for k in ("layout", "image", "exemplar", "exemplar_parsing"))
for s in range(60):
    loss = F.l1_loss(g_tmp(L, E, P)["fake"], I)
    assert torch.isfinite(loss)
    if s in (0, 19, 59): print(f"step {s+1}: L1 {loss.item():.4f}")
    opt.zero_grad(); loss.backward(); opt.step()
del g_tmp, opt
if torch.cuda.is_available(): print(f"peak GPU memory so far: {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
""")


md(r"""
## 10. Training: warp stage
Set `RUN_WARP_TRAINING=True`. Checkpoints go to `<drive>/checkpoints/<experiment>/warp/`: `last.pt` every `training.checkpoint_every` steps, plus `best.pt` (best val mIoU).
**After a disconnect, run the notebook again with the same settings.** Training resumes at the exact batch where it stopped.
To train longer, raise `WARP_MAX_STEPS` and re-run.

What to watch:
* `ce` should fall.
* Validation `mIoU` (garment classes, background excluded) should rise.
* Swap test (garment from one person, body from another): `class_area_agreement` should rise. It measures whether the predicted garment mix (e.g. top + skirt) matches the input; a skirt that turns into trousers scores low. `class_fidelity` is a lenient version of the same check.
* Per-class IoU: rare classes (skirt, outer, dress) lag behind top/pants. Their trend matters more than the mean.
* mIoU averages the 7 trained classes (skin, top, outer, dress, skirt, pants, footwear). Hair, face, headwear and accessories are never trained (face/hair come from the target person) and are excluded. Runs before 2026-10-21 used an older, buggy definition.
""")
code(r"""
if RUN_WARP_TRAINING:
    result = trainer.fit(max_steps=WARP_MAX_STEPS, time_limit_min=WARP_TIME_LIMIT_MIN)
    print(result)
else:
    print("Skipped (RUN_WARP_TRAINING=False).")
""")
code(r"""
# Score saved checkpoints on all 1,271 validation images (mIoU over the 7 trained classes).
# Use this to compare checkpoints; validation during training only uses the first val_max_batches batches.
for which in EVAL_WARP_CHECKPOINTS:
    m = trainer.evaluate_checkpoint(which)
    print(f"{which:5s} step {m['step']}: mIoU {m['mean_iou']:.4f} | area agreement {m['class_area_agreement']:.3f} | "
          f"acc {m['pixel_accuracy']:.4f} on {m['n_images']} images")
if not EVAL_WARP_CHECKPOINTS:
    print("Skipped (EVAL_WARP_CHECKPOINTS is empty).")
""")
code(r"""
# Training curves and the latest validation grid (columns: input C_seg | B_seg | prediction | ground truth, grey = not supervised;
# lower rows: swap test, no ground truth)
from PIL import Image
from IPython.display import display
log = trainer.log.read()
tr = [r for r in log if r.get("kind") == "train"]; va = [r for r in log if r.get("kind") == "val"]
if tr:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot([r["step"] for r in tr], [r["ce"] for r in tr]); ax[0].set_title("train CE"); ax[0].set_xlabel("step")
    if va:
        ax[1].plot([r["step"] for r in va], [r["mean_iou"] for r in va], marker="o", label="val mIoU")
        ax[1].plot([r["step"] for r in va], [r["class_fidelity"] for r in va], marker="x", label="swap class fidelity (lenient)")
        ax[1].plot([r["step"] for r in va], [r.get("class_area_agreement", float("nan")) for r in va], marker="s",
                   label="swap area agreement (strict)")
        ax[1].legend(); ax[1].set_xlabel("step")
    plt.show()
if va:
    last = va[-1]
    print(f"latest val (step {last['step']}): mIoU {last['mean_iou']:.3f}")
    print("per-class IoU:", {k: round(v, 3) for k, v in last["per_class_iou"].items() if v is not None and k != "background"})
grids = sorted(trainer.out_dir.glob("val_step*.png"))
if grids:
    print(grids[-1].name); display(Image.open(grids[-1]))
else:
    print("No validation grid yet.")
""")

md(r"""
## 10b. Training: texture stage
Set `RUN_TEXTURE_TRAINING=True` (and `ALLOW_VGG_DOWNLOAD=True` the first time). Checkpoints go to `<drive>/checkpoints/<experiment>/texture/`; re-running resumes exactly.
What to watch:
* `vgg` (perceptual), `warp` (warped exemplar vs real photo) and `domain` should fall.
* `d` should stay roughly between 0.5 and 2 (hinge loss); `d` near 0 means the discriminator is winning.
* Validation **person-region SSIM** (generated vs real target photo, white background excluded) should rise and stay above the two baselines printed next to it: copying the exemplar unchanged, and the warped exemplar I_S alone. Whole-image SSIM is inflated by the white background.
* Grid columns: exemplar | target layout | warped exemplar I_S | generated | real photo. The lower rows are a swap with no ground truth.
""")
code(r"""
if RUN_TEXTURE_TRAINING:
    print(tex.fit(max_steps=TEXTURE_MAX_STEPS, time_limit_min=TEXTURE_TIME_LIMIT_MIN))
else:
    print("Skipped (RUN_TEXTURE_TRAINING=False).")
for which in EVAL_TEXTURE_CHECKPOINTS:
    m = tex.evaluate_checkpoint(which)
    print(f"{which:5s} step {m['step']} on {m['n_images']} pairs: SSIM {m['ssim']:.4f} | person-region SSIM {m['ssim_fg']:.4f} "
          f"L1 {m['l1_fg']:.4f} | baselines (person region): copy exemplar SSIM {m['copy_ssim_fg']:.4f}, "
          f"warped I_S only SSIM {m['is_ssim_fg']:.4f}")
""")
code(r"""
from PIL import Image
from IPython.display import display
tlog = tex.log.read()
ttr = [r for r in tlog if r.get("kind") == "train"]; tva = [r for r in tlog if r.get("kind") == "val"]
if ttr:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3))
    st = [r["step"] for r in ttr]
    for k in ("vgg", "warp", "domain", "cycle"):
        ax[0].plot(st, [r.get(k, float("nan")) for r in ttr], label=k)
    ax[0].legend(); ax[0].set_title("reconstruction / alignment losses")
    ax[1].plot(st, [r.get("g_adv", float("nan")) for r in ttr], label="g_adv"); ax[1].plot(st, [r.get("d", float("nan")) for r in ttr], label="d")
    ax[1].legend(); ax[1].set_title("adversarial (hinge)")
    if tva:
        ax[2].plot([r["step"] for r in tva], [r["ssim"] for r in tva], marker="o", label="val SSIM")
        ax[2].plot([r["step"] for r in tva], [r["l1"] for r in tva], marker="s", label="val L1")
        ax[2].legend(); ax[2].set_title("validation (cross-view pairs)")
    plt.show()
tgrids = sorted(tex.out_dir.glob("*_step*.png"))
if tgrids:
    print(tgrids[-1].name); display(Image.open(tgrids[-1]))
""")

md(r"""
## 11. End-to-end try-on (Phase 6)
The full pipeline: Stage A, then warp (C'_seg), then the target's face/hair labels (paper Sec. 4.2), then texture, then the target's face/hair pixels pasted back.
* **Paired evaluation** (validation pairs of the same item, the real photo is the answer): scored on garment + skin pixels only. Face/hair are pasted from the target, so they are excluded. Compared with the texture stage given the ground-truth layout (upper bound) and with copying the exemplar (lower bound).
* **Swap grid:** real try-on between different items (no answer).
* **Your own photos:** set `RUN_UPLOAD_TRYON=True` in the cell below and run just that cell (works without `RUN_TRYON_EVAL`); uses SCHP for Stage A. On your PC: `python scripts/tryon.py --reference <img> --target <img>`.
""")
code(r"""
from src.inference.tryon import load_tryon_models
from src.evaluation.end_to_end import evaluate_end_to_end, tryon_swap_grid
e2e = None
if RUN_TRYON_EVAL:
    models = load_tryon_models(cfg, warp=TRYON_WARP_CKPT or cfg.evaluation.warp_checkpoint, texture=TRYON_TEXTURE_CKPT or cfg.evaluation.texture_checkpoint)
    print("models:", models.info)
    e2e_dir = paths.experiment_dir("outputs", cfg.experiment.name) / "tryon"; e2e_dir.mkdir(exist_ok=True)
    e2e = evaluate_end_to_end(models, prepared, cfg, split="val", limit=TRYON_EVAL_LIMIT, grid_path=e2e_dir / "e2e_val_grid.png")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e2e.items()})
    print("grid columns: reference | real target | C'_seg | layout+identity | warped I_S | generated | result (face pasted) | texture with GT layout")
    display(Image.open(e2e_dir / "e2e_val_grid.png"))
    if tryon_swap_grid(models, prepared, cfg, split="val", n=6, path=e2e_dir / "swap_grid.png") is not None:
        print("swap grid columns: reference | target | C'_seg | warped I_S | generated | result")
        display(Image.open(e2e_dir / "swap_grid.png"))
    else:
        print("No swap grid: the validation split has fewer than two items.")
else:
    print("Skipped (RUN_TRYON_EVAL=False).")
""")
code(r"""
# Your own photos. Set RUN_UPLOAD_TRYON=True and run this cell on its own (after Sections 0-5; no dataset needed).
# Upload 2 images when asked. The file that sorts FIRST by name is the reference (person WEARING the garment), the
# second is the target person, e.g. name them 1_reference.jpg and 2_target.jpg (upload order is not reliable).
# Best: full body, front-facing, one person, plain background. Output is 128 x 128 (model resolution).
RUN_UPLOAD_TRYON = False
if RUN_UPLOAD_TRYON and IN_COLAB:
    from google.colab import files
    import io
    from PIL import Image
    from IPython.display import display
    from src.data.preprocessing import load_image
    from src.inference.pipeline import PreTreatment, try_on
    from src.inference.tryon import load_tryon_models
    if globals().get("models") is None:
        models = load_tryon_models(cfg, warp=TRYON_WARP_CKPT or cfg.evaluation.warp_checkpoint, texture=TRYON_TEXTURE_CKPT or cfg.evaluation.texture_checkpoint)
    if globals().get("pre") is None:
        pre = PreTreatment(cfg)
    up = files.upload()
    if len(up) < 2:
        raise ValueError("Upload two images: reference first, then target.")
    (rn, rb), (tn, tb_) = sorted(up.items())[:2]
    print("reference:", rn, "| target:", tn)
    ref_u, tgt_u = load_image(Image.open(io.BytesIO(rb))), load_image(Image.open(io.BytesIO(tb_)))
    up_dir = paths.experiment_dir("outputs", cfg.experiment.name) / "my_tryon" / f"{Path(rn).stem}__on__{Path(tn).stem}"
    r = try_on(ref_u, tgt_u, cfg, pretreatment=pre, models=models, save_dir=up_dir)
    print("saved to", up_dir)
    display(Image.open(up_dir / "overview.png"))
    print("columns: reference | C_seg | target | B_seg | C'_seg | layout | warped I_S | generated | result")
""")

md(r"""
## 12. Final evaluation (Phase 7)
Protocol, fixed **before** the first test run: `docs/final_evaluation.md` (measurements T1-T10).
* `FINAL_SPLIT="val"`: any number of runs, used for decisions. `FINAL_SPLIT="test"`: reserved for final reporting; every run is appended to `<drive>/cache/test_split_log.jsonl`, unfinished ones too.
* LPIPS and FID need pretrained networks: set `ALLOW_METRIC_DOWNLOAD=True` once (~233 MB AlexNet from download.pytorch.org, ~92 MB Inception from the pytorch-fid GitHub release); they are cached in `<drive>/pretrained/torch`.
* Each part is saved to `results.json` / `results_table.md` as soon as it finishes, in `<drive>/outputs/<experiment>/final_eval/<split>_<time>/`.
* Estimated time on a T4 per split: about 15-40 min (not yet measured; slowest parts: SCHP over all images, the FID try-ons).
""")
code(r"""
from src.evaluation.final_eval import read_test_log, run_final_evaluation
final = None
if RUN_FINAL_EVAL and not DRY_RUN:
    models = load_tryon_models(cfg, warp=TRYON_WARP_CKPT or cfg.evaluation.warp_checkpoint, texture=TRYON_TEXTURE_CKPT or cfg.evaluation.texture_checkpoint)
    print("models:", models.info)
    if FINAL_SPLIT == "test":
        print("earlier test-split runs:", sum(e.get("event") == "start" for e in read_test_log(cfg)))
    final = run_final_evaluation(cfg, prepared, models, split=FINAL_SPLIT, confirm_test=(FINAL_SPLIT == "test"),
                                 parts=FINAL_PARTS, limit=FINAL_LIMIT)
    print(Path(final["out_dir"], "results_table.md").read_text())
    for g in ("paired_grid.png", "realism_grid.png"):
        if Path(final["out_dir"], g).is_file():
            print(g); display(Image.open(Path(final["out_dir"], g)))
else:
    print("Skipped (RUN_FINAL_EVAL=False or DRY_RUN).")
""")

md(r"""
## 13. Experiments (Phase 8)
Each experiment is one config file in `configs/exp_*.yaml` that changes **one** thing, writes into its **own** experiment folder, and is judged on the **validation** split against the frozen baseline B0 (`docs/experiments.md`). B0's checkpoints are marked FROZEN in Section 5, so an experiment cannot overwrite them.

Session recipe (for example Experiment B):
1. Section 0: `CONFIG_FILE = "configs/exp_b_classweights.yaml"`, `FETCH_DFMM = True`, `RUN_WARP_TRAINING = True`, `WARP_MAX_STEPS = 10000` (same as B0), `RUN_FINAL_EVAL = True`, `FINAL_SPLIT = "val"`, `FINAL_PARTS = ["warp", "paired", "realism"]`.
2. Run all. Training resumes if the session drops; the evaluation uses B0's texture network automatically (set in the config).
3. This section compares the experiment with B0 on the same validation pairs and saves `comparison.md` next to the results.
""")
code(r"""
COMPARE_WITH_BASELINE = True
if COMPARE_WITH_BASELINE and not DRY_RUN and cfg.experiment.name != "baseline_128":
    from src.evaluation.compare import compare_results, find_latest_results, format_comparison
    import json as _json
    try:
        b0 = find_latest_results(paths.outputs, "baseline_128", "val")
        ex = find_latest_results(paths.outputs, cfg.experiment.name, "val")
    except FileNotFoundError as e:
        print("Nothing to compare yet:", e)
    else:
        cmp = compare_results(_json.loads(b0.read_text()), _json.loads(ex.read_text()))
        text = format_comparison(cmp, cfg.experiment.name)
        (ex.parent / "comparison.md").write_text(text)
        (ex.parent / "comparison.json").write_text(_json.dumps(cmp, indent=1, default=float))
        print(text)
        print("saved:", ex.parent / "comparison.md")
else:
    print("Skipped (baseline config, DRY_RUN, or COMPARE_WITH_BASELINE=False).")
""")

md(r"""
## 12b. Paper-style benchmark (standard DeepFashion pose-transfer pairs)
The public test pairs used by ADGAN and DiOr (8,570 pairs), scored like the paper's Table 1: whole 176 x 256 images, SSIM / L1 / LPIPS and FID/KID. Pairs whose products we trained on are never scored; our locked test products are skipped (`docs/paper_comparison.md` section 2c). Needs `FETCH_DFMM=True` (photos on the local disk).
""")
code(r"""
pt_out = None
if (RUN_PT_LEAKAGE or RUN_PT_BENCHMARK) and not DRY_RUN:
    from scripts.pose_transfer_benchmark import main as pt_main
    from PIL import Image
    from IPython.display import display
    import datetime as _dt
    pt_out = paths.outputs / cfg.experiment.name / "pose_transfer" / _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pt_args = ["--config", CONFIG_FILE, "--drive-root", str(paths.drive_root),
               "--image-dir", str(Path(cfg.data.root) / cfg.data.layout.images_dir), "--out", str(pt_out),
               *sum((["--override", o] for o in EXTRA_OVERRIDES), [])]
    if ALLOW_PAIR_DOWNLOAD:
        pt_args.append("--allow-pair-download")
    if ALLOW_METRIC_DOWNLOAD:
        pt_args += ["--override", "evaluation.allow_metric_download=true"]
    if not RUN_PT_BENCHMARK:
        pt_args.append("--leakage-only")
    if PT_LIMIT:
        pt_args += ["--limit", str(PT_LIMIT)]
    pt_main(pt_args)
    if (pt_out / "grid.png").is_file():
        print("grid columns: reference (copy) | real target | ours raw | ours + face/hair paste")
        display(Image.open(pt_out / "grid.png"))
else:
    print("Skipped (RUN_PT_LEAKAGE / RUN_PT_BENCHMARK are False, or DRY_RUN).")
""")

md(r"""
## 12c. DeepFashion In-shop inspection (text annotations only)
Needs a Drive shortcut to the shared "In-shop Clothes Retrieval Benchmark" folder in `VirtualTryOn/datasets/` and `FETCH_DFMM=True`.
""")
code(r"""
if RUN_INSHOP_INSPECTION and not DRY_RUN:
    from scripts.inspect_inshop import main as inshop_main
    inshop_main(["--inshop", INSHOP_DIR, "--split-file", cfg.data.split_file,
                 "--dfmm-images", str(Path(cfg.data.root) / cfg.data.layout.images_dir),
                 "--out", str(paths.outputs / "inshop_inspection.json")])
else:
    print("Skipped (RUN_INSHOP_INSPECTION=False or DRY_RUN).")
""")

md("## 14. Saving and exporting\nWith `EXPORT_RESULTS=True` everything this session produced (training logs as CSV, evaluation metrics, grids, pair statistics) is copied to `<drive>/report_exports/<date-time>/`. Download that folder from Drive and put it in the project under `report/private/exports/`; no need to copy numbers by hand.")
code(r"""
if EXPORT_RESULTS and not DRY_RUN:
    import shutil, datetime, json as _json
    from scripts.export_run_logs import flatten, write
    exp_dir = Path(DRIVE_PROJECT_ROOT) / "report_exports" / datetime.datetime.now().strftime("%Y%m%d_%H%M")
    exp_dir.mkdir(parents=True, exist_ok=True)
    for t in [x for x in (globals().get("trainer"), globals().get("tex")) if x is not None]:
        recs = t.log.read()
        write([flatten(r) for r in recs if r.get("kind") == "train"], exp_dir / f"{t.run_name}_train.csv")
        write([flatten(r) for r in recs if r.get("kind", "").startswith(("val", "eval"))], exp_dir / f"{t.run_name}_val.csv")
        for g in sorted(t.out_dir.glob("*.png"))[-3:]:
            shutil.copy(g, exp_dir / f"{t.run_name}_{g.name}")
    if globals().get("e2e"):
        (exp_dir / "end_to_end_metrics.json").write_text(_json.dumps(e2e, indent=1, default=str))
        for g in (Path(e2e["grid"]).parent).glob("*.png"):
            shutil.copy(g, exp_dir / f"tryon_{g.name}")
    if globals().get("pt_out") is not None and Path(pt_out).is_dir():
        shutil.copytree(pt_out, exp_dir / f"pose_transfer_{Path(pt_out).name}", dirs_exist_ok=True)
    if (paths.outputs / "inshop_inspection.json").is_file():
        shutil.copy(paths.outputs / "inshop_inspection.json", exp_dir / "inshop_inspection.json")
    if globals().get("final"):
        shutil.copytree(final["out_dir"], exp_dir / f"{cfg.experiment.name}_final_eval_{final['split']}", dirs_exist_ok=True)
        from src.evaluation.final_eval import test_log_path
        if test_log_path(cfg).is_file():
            shutil.copy(test_log_path(cfg), exp_dir / "test_split_log.jsonl")
    pq = paths.experiment_dir("outputs", cfg.experiment.name) / "pair_quality"
    if pq.is_dir():
        shutil.copytree(pq, exp_dir / "pair_quality", dirs_exist_ok=True)
    (exp_dir / "session_settings.json").write_text(_json.dumps({k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list, bool, type(None)))}, indent=1))
    print("exported to", exp_dir, ":", sorted(p.name for p in exp_dir.iterdir()))
""")
md("### Saved-artifact check")
code(r"""
out_dir = paths.experiment_dir("outputs", cfg.experiment.name)
for p in [out_dir / "config_resolved.yaml", Path(cfg.data.split_file),
          Path(cfg.paths.cache_dir) / "reports" / f"{cfg.data.dataset}_data_report.json", out_dir / "pretreatment_example",
          trainer.ckpt.last, tex.ckpt.last]:
    print(("OK      " if p.exists() else "MISSING ") + str(p))
# Reload check: the saved config must reproduce the same settings in a fresh runtime
assert load_config(out_dir / "config_resolved.yaml").to_dict() == cfg.to_dict()
print("saved config reloads identically")
""")


md(r"""
## 16. Final model: live demo and results (presentation)
**Final model** (docs/modal.md): 256 px, trained on all In-shop photos. For your own photos the demo uses **v2** =
warp `checkpoints/full_256/warp/generator_only.pt` + texture `checkpoints/v2_256/texture_ft/generator_only.pt`
(copy both from `report/private/models/` on the PC into the same folders under the Drive project root).
The reported W1 model additionally needs DensePose maps, which own photos do not have (0.0009 SSIM higher).

Cells: (1) load the final models, (2) try on uploaded photos, (3) all metric tables recomputed from the saved per-pair
scores, (4) optional live benchmark on a few standard pairs.
""")
code(r"""
demo_models = demo_pre = demo_cfg = None
if RUN_FINAL_DEMO or RUN_LIVE_BENCHMARK:
    from src.config import load_config
    from src.inference.pipeline import PreTreatment
    from src.inference.tryon import load_tryon_models
    FINAL_WARP = paths.checkpoints / "full_256" / "warp" / "generator_only.pt"
    FINAL_TEXTURE = paths.checkpoints / "v2_256" / "texture_ft" / "generator_only.pt"
    missing = [str(f) for f in (FINAL_WARP, FINAL_TEXTURE) if not f.is_file()]
    if missing:
        raise FileNotFoundError("Copy the final weights to Drive first: " + ", ".join(missing))
    demo_cfg = load_config("configs/modal_full_256.yaml", [f"paths.drive_root={paths.drive_root}",
                                                           "segmentation.cache_outputs=false"])
    demo_models = load_tryon_models(demo_cfg, warp=FINAL_WARP, texture=FINAL_TEXTURE)
    demo_pre = PreTreatment(demo_cfg)
    print("final models loaded:", demo_models.info, "| device", demo_models.device, "| size", demo_cfg.data.image_size)
else:
    print("Skipped (RUN_FINAL_DEMO=False).")
""")
code(r"""
# LIVE TRY-ON. Run this cell, then upload 2 photos: the file that sorts FIRST by name is the reference (person WEARING
# the garment), the second the target person, e.g. 1_reference.jpg and 2_target.jpg. Best: full body, front-facing,
# one person, plain light background (like DeepFashion). Re-run the cell for each new pair.
if RUN_FINAL_DEMO and IN_COLAB and demo_models is not None:
    import io
    import time as _time
    from google.colab import files
    from PIL import Image
    from IPython.display import display
    from src.data.preprocessing import load_image
    from src.inference.pipeline import try_on
    up = files.upload()
    if len(up) < 2:
        raise ValueError("Upload two images: reference and target.")
    (rn, rb), (tn, tb_) = sorted(up.items())[:2]
    print("reference:", rn, "| target:", tn)
    ref_u, tgt_u = load_image(Image.open(io.BytesIO(rb))), load_image(Image.open(io.BytesIO(tb_)))
    out_dir = paths.outputs / "final_demo" / f"{Path(rn).stem}__on__{Path(tn).stem}"
    t0 = _time.time()
    r = try_on(ref_u, tgt_u, demo_cfg, pretreatment=demo_pre, models=demo_models, save_dir=out_dir)
    print(f"done in {_time.time() - t0:.1f} s; saved to {out_dir}")
    display(Image.open(out_dir / "overview.png"))
    print("columns: reference | C_seg (its garments) | target | B_seg (target body) | C'_seg (warped layout) | layout + "
          "face/hair | warped texture I_S | generated | result (target face/hair pasted)")
else:
    print("Skipped (RUN_FINAL_DEMO=False, not in Colab, or models not loaded).")
""")
code(r"""
# ALL METRICS, recomputed now from the per-pair scores saved when the models were evaluated on Modal
# (report/data/results/*.json: one SSIM / LPIPS / L1 value per test pair). Means and paired 95% CIs are computed here.
if RUN_SHOW_RESULTS:
    from scripts.show_results import main as show_results
    show_results([])
else:
    print("Skipped (RUN_SHOW_RESULTS=False).")
""")
code(r"""
# LIVE BENCHMARK on LIVE_PAIRS standard DeepFashion test pairs (fully automatic, whole 176 x 256 image), using the
# final split (report/data/results/splits_pack256.json) so that no pair of a training product is scored.
# Needs FETCH_DFMM=True (photos), the pair lists (ALLOW_PAIR_DOWNLOAD=True once) and the metric weights on Drive.
# FID on ~100 images is not meaningful (it needs thousands); SSIM / LPIPS are comparable with the full-run numbers.
if RUN_LIVE_BENCHMARK and demo_models is not None:
    from scripts.pose_transfer_benchmark import main as pt_main
    from PIL import Image
    from IPython.display import display
    live_out = paths.outputs / "final_demo" / "live_benchmark"
    args = ["--config", "configs/modal_full_256.yaml", "--drive-root", str(paths.drive_root),
            "--image-dir", str(Path(cfg.data.root) / cfg.data.layout.images_dir), "--out", str(live_out),
            "--limit", str(LIVE_PAIRS), "--override", "data.split_file=report/data/results/splits_pack256.json",
            "--override", f"evaluation.warp_checkpoint={FINAL_WARP}",
            "--override", f"evaluation.texture_checkpoint={FINAL_TEXTURE}",
            "--override", "segmentation.cache_outputs=false"]
    if ALLOW_PAIR_DOWNLOAD:
        args.append("--allow-pair-download")
    pt_main(args)
    display(Image.open(live_out / "grid.png"))
    print("grid columns: reference photo | real target (ground truth) | our output | our output + target face/hair")
else:
    print("Skipped (RUN_LIVE_BENCHMARK=False or models not loaded).")
""")

md(r"""
## 15. Secure the results (run before deleting the runtime)
1. Every `last.pt` / `best.pt` is **fully loaded**, which catches incomplete files (for example when Drive is full).
2. A small `generator_only.pt` backup is written next to it (inference needs only this).
3. Redundant `step_*.pt` snapshots are deleted, but only if `last.pt` loaded correctly. A full warp checkpoint is ~0.8 GB.
4. Drive usage is shown, then all pending writes are flushed to Drive.

When this cell prints **`SAFE TO DELETE RUNTIME`**, everything trained in this session is permanently in your Drive.
""")
code(r"""
if SECURE_RESULTS and not DRY_RUN:
    from src.training.checkpointing import verify_checkpoint, export_generator, prune_snapshots
    ckroot = Path(cfg.paths.checkpoints_dir) / cfg.experiment.name
    problems = []
    for run in sorted(d for d in ckroot.iterdir() if d.is_dir()):
        for name in ("last.pt", "best.pt"):
            f = run / name
            if not f.is_file():
                continue
            try:
                info = verify_checkpoint(f)
                print(f"OK   {run.name}/{name}: step {info['step']}, {info['MiB']} MiB")
                if name == "last.pt":
                    g = export_generator(f)
                    print(f"     backup {run.name}/{g.name}: {g.stat().st_size / 2**20:.0f} MiB")
            except Exception as exc:
                problems.append(f"{run.name}/{name}: {type(exc).__name__}: {exc}")
                print("FAIL", problems[-1])
        if PRUNE_SNAPSHOTS and (run / "last.pt").is_file() and not any(run.name in p for p in problems):
            r = prune_snapshots(run)
            if r["removed"]:
                print(f"     pruned {len(r['removed'])} snapshot(s) in {run.name}, freed {r['freed_MiB']} MiB")
    import shutil as _sh
    tot, used, free = _sh.disk_usage(DRIVE_PROJECT_ROOT)
    print(f"Drive: {used / 2**30:.1f} GB used of {tot / 2**30:.1f} GB ({free / 2**30:.1f} GB free)")
    if problems:
        print("\n!!! NOT SAFE: fix the problems above (free Drive space, then re-run the training cell to re-save).")
    if FLUSH_DRIVE_AT_END and IN_COLAB and not problems:
        from google.colab import drive
        drive.flush_and_unmount()
        print("\nDrive flushed and unmounted. SAFE TO DELETE RUNTIME.")
else:
    print("Skipped (SECURE_RESULTS=False or DRY_RUN).")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": []},
})
nbf.write(nb, out)
print("cells:", len(cells))
