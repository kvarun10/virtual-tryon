"""Builds notebooks/virtual_try_on_colab.ipynb.

    python scripts/build_notebook.py notebooks/virtual_try_on_colab.ipynb

Edit this file, not the .ipynb, so the notebook stays reproducible from version control.
"""
import sys
import nbformat as nbf

out = sys.argv[1]
cells = []
_guard = {"on": False}  # with DEMO_ONLY=True the code cells of Sections 6-17 are skipped


def md(s):
    head = s.lstrip("\n")
    if head.startswith("## 6."):
        _guard["on"] = True
    if head.startswith("## 18."):
        _guard["on"] = False
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    s = s.strip("\n")
    if _guard["on"]:
        body = "\n".join(("    " + l) if l.strip() else l for l in s.splitlines())
        s = ('if DEMO_ONLY:\n    print("Skipped (DEMO_ONLY=True).")\n'
             'else:\n' + body)
    cells.append(nbf.v4.new_code_cell(s))


md(r"""
# Virtual Try-On: segmentation-guided human appearance transfer

Re-implementation of *Dress-up: deep neural framework for image-based human appearance transfer*
(Ghodhbani et al., Multimedia Tools and Applications, 2023): a reference image provides the clothes, a target
image provides the person and pose, and the model generates the target person wearing those clothes.

**Quick demo:** in Section 0 set `DRY_RUN = False`, `DEMO_ONLY = True`, `RUN_FINAL_DEMO = True` and
`RUN_SHOW_RESULTS = True`, then run all cells.
""")

md(r"""
## 0. Settings
All options of the notebook. For a private repository, add a Colab secret named `GITHUB_TOKEN`.
""")
code(r"""
# Code
REPO_URL = "https://github.com/kvarun10/virtual-tryon.git"
CODE_ZIP_IN_DRIVE = "code/virtual-tryon.zip"
COLAB_PROJECT_DIR = "/content/virtual-tryon"

# Storage
USE_DRIVE = True
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/VirtualTryOn"

# Run mode
DRY_RUN = True
REQUIRE_GPU = True
CONFIG_FILE = "configs/colab_smoke_test.yaml" if DRY_RUN else "configs/colab_baseline.yaml"
ALLOW_WEIGHT_DOWNLOAD = False
EXTRA_OVERRIDES = []

# Dataset
FETCH_DFMM = False
DFMM_ALLOW_DOWNLOAD = False
DFMM_PARTS = "images_parsing"
ANALYZE_PAIRS = False

# Training: warp network
PRECOMPUTE_BODY_MAPS = False
RUN_WARP_TRAINING = False
WARP_MAX_STEPS = 2000
EVAL_WARP_CHECKPOINTS = []

# Training: texture network
ALLOW_VGG_DOWNLOAD = False
RUN_TEXTURE_TRAINING = False
TEXTURE_MAX_STEPS = 2000
EVAL_TEXTURE_CHECKPOINTS = []

# Training time limits (minutes, None = no limit)
WARP_TIME_LIMIT_MIN = None
TEXTURE_TIME_LIMIT_MIN = None

# Try-on and evaluation
RUN_TRYON_EVAL = False
TRYON_WARP_CKPT = None
TRYON_TEXTURE_CKPT = None
TRYON_EVAL_LIMIT = None
RUN_FINAL_EVAL = False
FINAL_SPLIT = "val"
FINAL_PARTS = ["parser", "warp", "paired", "realism", "noise_floor", "timing"]
FINAL_LIMIT = None
ALLOW_METRIC_DOWNLOAD = False

# Benchmark (standard DeepFashion pose-transfer pairs)
RUN_PT_LEAKAGE = False
RUN_PT_BENCHMARK = False
ALLOW_PAIR_DOWNLOAD = False
PT_LIMIT = None

# DeepFashion In-shop inspection
RUN_INSHOP_INSPECTION = False
INSHOP_DIR = "/content/drive/MyDrive/VirtualTryOn/datasets/In-shop Clothes Retrieval Benchmark"

# Final model demo and results
DEMO_ONLY = False
RUN_FINAL_DEMO = False
RUN_SHOW_RESULTS = False
RUN_LIVE_BENCHMARK = False
LIVE_PAIRS = 100

# Saving
EXPORT_RESULTS = True
SECURE_RESULTS = True
PRUNE_SNAPSHOTS = False
FLUSH_DRIVE_AT_END = True
""")

md(r"""
## 1. Pipeline overview

```
 reference I_C ──► clothing parser ──► C_seg ─┐
                                              ├─► warp network ──► C'_seg ─┐
 target I_B ────► body parser ──────► B_seg ──┘                            ├─► texture network ──► result I_G
 reference I_C ────────────────────────────────────────────────────────────┘
```

* **Stage A:** parsers turn both images into label maps.
* **Stage B:** the warp network predicts the garment layout on the target body.
* **Stage C:** the texture network paints the reference's clothes onto that layout.
""")

md(r"""
## 2. Runtime check
Prints the Python, PyTorch and GPU versions.
""")
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

md(r"""
## 3. Google Drive
Mounts Google Drive, where datasets, weights and outputs are stored.
""")
code(r"""
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_PROJECT_ROOT, exist_ok=True)
    print("Drive project root:", DRIVE_PROJECT_ROOT)
else:
    print("Drive not mounted (not in Colab or USE_DRIVE=False); using local folders.")
""")

md(r"""
## 4. Code and dependencies
Clones the repository (or extracts the code zip) and installs the requirements.
""")
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
            except Exception:
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
stale = [m for m in sys.modules if m == "src" or m.startswith("src.")]
for m in stale:
    del sys.modules[m]
if stale:
    print(f"reloaded project code ({len(stale)} previously imported modules discarded)")
rev = subprocess.run(["git", "-C", str(PROJECT_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
print("project:", PROJECT_DIR, "| commit:", rev.stdout.strip() or "unknown (not a git checkout)")
""")
md("Checks the installed package versions.")
code(r"""
import importlib
for mod in ["numpy", "PIL", "cv2", "scipy", "matplotlib", "yaml", "tqdm"]:
    m = importlib.import_module(mod)
    print(f"{mod:10s} {getattr(m, '__version__', '?')}")
from src.utils.device import runtime_report, format_runtime_report
print(format_runtime_report(runtime_report()))
""")

md(r"""
## 5. Configuration
Loads the experiment configuration and sets up the storage folders.
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
if not DRY_RUN:
    from src.training.checkpointing import freeze_run, is_frozen
    for _run in ("warp_pf", "texture_pf"):
        _d = paths.checkpoints / "baseline_128" / _run
        if (_d / "last.pt").is_file():
            freeze_run(_d, "B0 baseline: official Phase 7 evaluation 2026-10-22 (tag baseline-v1)")
            print("frozen baseline:", _d, is_frozen(_d))
""")

md(r"""
## 6. Dataset
### 6.1 Download
Downloads DeepFashion-MultiModal into Drive (first session only) and extracts it to the Colab disk.
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
### 6.2 Preparation
Builds the image index, checks the annotations and creates product-disjoint train/validation/test splits.
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
md("Shows sample images with their label maps.")
code(r"""
import matplotlib.pyplot as plt
from src.data.labels import get_label_set
from src.data.preprocessing import tensor_to_image
from src.inference.visualization import overlay, show_panels

G, B = get_label_set(cfg.labels.garment_label_set), get_label_set(cfg.labels.body_label_set)
from src.data.labels import convert_labels, colorize
from src.data.preprocessing import load_image, load_label_map
recs = prepared.records("train")[:2]
if not all(r.densepose for r in recs):
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
### 6.3 Pair consistency
Measures whether two images of the same product also show the same outfit (garment classes and colour difference).
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
## 7. Stage A: pre-treatment
Parses a reference image and a target image into a clothing map and a body map.
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
md("Measures the clothing parser's accuracy against the manual labels.")
code(r"""
from src.evaluation.evaluate import evaluate_clothing_parser
if pre.clothing.name.startswith("annotation"):
    print("Clothing backend reads the ground truth itself; evaluation skipped.")
else:
    report = evaluate_clothing_parser(pre.clothing, prepared, cfg, split="val", limit=50)
    print(f"{report['backend']}: pixel acc {report['pixel_accuracy']:.3f} | mIoU (no bg) {report['mean_iou']:.3f} on {report['n_images']} val images")
    print({k: (round(v, 3) if v is not None else None) for k, v in report["per_class_iou"].items()})
""")
md("Parses your own uploaded images (set `RUN_UPLOAD = True`).")
code(r"""
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
md("Experimental: builds a clothing map from a product image instead of a worn garment.")
code(r"""
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
## 8. Stage B: warp network
### 8.1 Body maps
Computes the body map of every training image once and stores it in Drive.
""")
code(r"""
from src.data.body_maps import stage_body_maps, precompute_body_maps, archive_body_maps
print("restore from Drive:", stage_body_maps(cfg))
if PRECOMPUTE_BODY_MAPS and not DRY_RUN:
    recs = [r for s in ("train", "val", "test") for r in prepared.records(s)]
    print(precompute_body_maps(cfg, recs, prepared.index.root))
    print("archived to", archive_body_maps(cfg))
else:
    print("Precompute skipped (PRECOMPUTE_BODY_MAPS=False or DRY_RUN).")
""")

md(r"""
### 8.2 Model check
Builds the dual-path U-Net and shows its size and the shapes of one training batch.
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
md("Overfits a copy of the model on one batch; the loss must fall.")
code(r"""
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
## 9. Stage C: texture network
Builds the correspondence and SPADE networks and shows their size and the shapes of one training batch.
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
md("Overfits a copy of the model on one batch; the loss must fall.")
code(r"""
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
## 10. Training: warp network
Trains the warp network (`RUN_WARP_TRAINING = True`). Re-running resumes from the last checkpoint.
""")
code(r"""
if RUN_WARP_TRAINING:
    result = trainer.fit(max_steps=WARP_MAX_STEPS, time_limit_min=WARP_TIME_LIMIT_MIN)
    print(result)
else:
    print("Skipped (RUN_WARP_TRAINING=False).")
""")
md("Scores saved checkpoints on the whole validation split.")
code(r"""
for which in EVAL_WARP_CHECKPOINTS:
    m = trainer.evaluate_checkpoint(which)
    print(f"{which:5s} step {m['step']}: mIoU {m['mean_iou']:.4f} | area agreement {m['class_area_agreement']:.3f} | "
          f"acc {m['pixel_accuracy']:.4f} on {m['n_images']} images")
if not EVAL_WARP_CHECKPOINTS:
    print("Skipped (EVAL_WARP_CHECKPOINTS is empty).")
""")
md("Plots the training curves and shows the latest validation grid.")
code(r"""
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
## 11. Training: texture network
Trains the texture network (`RUN_TEXTURE_TRAINING = True`) and scores saved checkpoints.
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
md("Plots the training curves and shows the latest validation grid.")
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
## 12. End-to-end try-on
Runs the full pipeline on validation pairs, scores it and shows example grids (`RUN_TRYON_EVAL = True`).
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
md(r"""
Try-on on your own images (set `RUN_UPLOAD_TRYON = True`). Upload two images: the file whose name sorts first is the
reference, the second is the target (e.g. `1_reference.jpg`, `2_target.jpg`).
""")
code(r"""
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
## 13. Evaluation
Scores the parser, warp and full pipeline with SSIM, L1, LPIPS, FID and KID (`RUN_FINAL_EVAL = True`).
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
## 14. Benchmark
Scores the model on the standard DeepFashion pose-transfer test pairs (`RUN_PT_BENCHMARK = True`).
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
## 15. DeepFashion In-shop inspection
Counts the In-shop images and products available beyond DeepFashion-MultiModal (`RUN_INSHOP_INSPECTION = True`).
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

md(r"""
## 16. Experiments
Compares the current experiment with the baseline on the same validation pairs.
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
## 17. Export results
Copies this session's logs, metrics and grids to `<drive>/report_exports/`.
""")
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
md("Checks that the saved configuration, splits and checkpoints exist.")
code(r"""
out_dir = paths.experiment_dir("outputs", cfg.experiment.name)
for p in [out_dir / "config_resolved.yaml", Path(cfg.data.split_file),
          Path(cfg.paths.cache_dir) / "reports" / f"{cfg.data.dataset}_data_report.json", out_dir / "pretreatment_example",
          trainer.ckpt.last, tex.ckpt.last]:
    print(("OK      " if p.exists() else "MISSING ") + str(p))
assert load_config(out_dir / "config_resolved.yaml").to_dict() == cfg.to_dict()
print("saved config reloads identically")
""")

md(r"""
## 18. Final model: demo and results
Loads the final 256 px models from Drive: `checkpoints/full_256/warp/generator_only.pt` and
`checkpoints/v2_256/texture_ft/generator_only.pt`.
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
md(r"""
**Live try-on.** Upload two images: the file whose name sorts first is the reference (person wearing the clothes),
the second is the target person (e.g. `1_reference.jpg`, `2_target.jpg`). Full-body, front-facing images on a plain
background work best.
""")
code(r"""
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
md("**All metrics**, recomputed from the saved per-pair benchmark scores.")
code(r"""
if RUN_SHOW_RESULTS:
    from scripts.show_results import main as show_results
    show_results([])
else:
    print("Skipped (RUN_SHOW_RESULTS=False).")
""")
md("**Live benchmark** on `LIVE_PAIRS` standard test pairs (needs the dataset on the Colab disk).")
code(r"""
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
## 19. Save results
Verifies every checkpoint, writes small inference-only backups and flushes Drive. Run before closing the runtime.
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
