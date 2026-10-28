"""Modal jobs for the full-data 256 px run (docs/modal.md). Everything lives in the private volume "tryon-vol":

    /vol/raw/          uploads from the PC (scripts/modal_upload.py): In-shop img.zip + partition, Drive zips
    /vol/project/      drive_root: pretrained/, cache/ (old splits), checkpoints/, outputs/, logs/, datasets/
    /vol/pack256/      packed 256 px store (src/data/packed.py), written by `pack`

Commands (run from the repo root; `--detach` keeps a job running if this PC disconnects):

    modal run modal_app.py::setup                         # unpack uploads, fetch pair lists + DFMM parsing (CPU)
    modal run modal_app.py::pack                          # build the store (L4, ~30 min)
    modal run modal_app.py::probe --gpus H100,A100-40GB,L40S
    modal run --detach modal_app.py::train --stage texture --gpu H100 --max-steps 14000
    modal run modal_app.py::benchmark --which b0          # or --which full_256
    modal run modal_app.py::status

Data stays private to the Modal workspace (DeepFashion licence: non-commercial research, no redistribution).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
APP_NAME = "virtual-tryon"
VOL = "/vol"
PROJECT = f"{VOL}/project"
PACK = f"{VOL}/pack256"
LOCAL_PACK = "/tmp/pack256"
CODE = "/root/project"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("tryon-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.6.0", "torchvision==0.21.0", "numpy>=1.24,<2.3", "pillow>=10", "opencv-python-headless>=4.8",
                    "scipy>=1.10", "pyyaml>=6", "tqdm>=4.65", "gdown>=5.1", "lpips>=0.1.4", "pytorch-fid>=0.3.0",
                    "matplotlib>=3.7", "scikit-image>=0.22")
    .env({"PYTHONUNBUFFERED": "1", "TORCH_HOME": f"{PROJECT}/pretrained/torch"})
    .add_local_dir(ROOT / "src", f"{CODE}/src")
    .add_local_dir(ROOT / "scripts", f"{CODE}/scripts")
    .add_local_dir(ROOT / "configs", f"{CODE}/configs")
)

COMMON = dict(image=image, volumes={VOL: vol})
CONFIGS = {"full_256": "modal_full_256.yaml", "v2_256": "modal_v2_256.yaml", "w1_256": "modal_w1_256.yaml"}
GPU_ALIASES = {"A100": "A100-40GB", "A100-40": "A100-40GB", "A100-80": "A100-80GB"}


def _code():
    os.chdir(CODE)
    if CODE not in sys.path:
        sys.path.insert(0, CODE)


def _stage_pack(source: str = PACK) -> str:
    """Copy the packed store to local disk (fast random reads for DataLoader workers)."""
    if not Path(LOCAL_PACK, "meta.json").is_file():
        t = time.time()
        shutil.copytree(source, LOCAL_PACK + ".part", dirs_exist_ok=True)
        os.replace(LOCAL_PACK + ".part", LOCAL_PACK)
        print(f"packed store copied to local disk in {time.time() - t:.0f}s", flush=True)
    return LOCAL_PACK


def _cfg(config: str, extra=()):
    from src.config import load_config

    over = [f"data.packed_dir={LOCAL_PACK}", f"data.split_file={LOCAL_PACK}/splits.json",
            f"data.pair_table={LOCAL_PACK}/pairs.csv", *extra]
    return load_config(f"{CODE}/configs/{config}", over)


# ------------------------------------------------------------------------------------------------ setup (CPU)
@app.function(**COMMON, cpu=4, memory=8192, timeout=3600)
def setup_volume() -> dict:
    """Unpack the uploaded Drive zips into /vol/project; fetch the two pair lists and DFMM parsing (both approved)."""
    import zipfile

    _code()
    raw = Path(VOL, "raw")
    report = {}
    for z in sorted((raw / "drive_zips").glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            top = names[0].split("/")[0]
            dest = Path(PROJECT) if top in ("cache", "pretrained") else Path(PROJECT, "checkpoints", "baseline_128")
            zf.extractall(dest)
            report[z.name] = f"{len(names)} files -> {dest}"
    from src.data.fetch import download_archive
    from src.data.pose_transfer import ensure_pair_files

    report["pose_pairs"] = {k: str(v) for k, v in ensure_pair_files(Path(PROJECT, "datasets", "pose_transfer_pairs"), True).items()}
    report["dfmm_parsing"] = str(download_archive("parsing", Path(PROJECT, "datasets", "DeepFashion-MultiModal", "archives"), True))
    need = ["pretrained/schp/exp-schp-201908261155-lip.pth", "pretrained/schp/exp-schp-201908270938-pascal-person-part.pth",
            "pretrained/torch/vgg19-dcbb9e9d.pth", "pretrained/torch/checkpoints/alexnet-owt-7be5be79.pth",
            "pretrained/torch/checkpoints/pt_inception-2015-12-05-6726825d.pth",
            "cache/splits/deepfashion_multimodal_seed0.json"]
    report["missing"] = [n for n in need if not Path(PROJECT, n).is_file()]
    report["raw_inshop"] = sorted(p.name for p in (raw / "inshop").glob("*"))
    vol.commit()
    return report


# ------------------------------------------------------------------------------------------------ packing (GPU)
@app.function(**COMMON, gpu="L4", cpu=8, memory=32768, timeout=3 * 3600)
def pack_store(limit: int = 0, batch_size: int = 32) -> dict:
    _code()
    import torch

    from src.config import load_config
    from src.data.fetch import find_archive
    from src.data.packing import build_pack
    from src.data.pose_transfer import ensure_pair_files, item_of, read_pairs
    from src.models.segmentation import build_backend

    cfg = load_config(f"{CODE}/configs/modal_full_256.yaml")
    t = time.time()
    shutil.copy(f"{VOL}/raw/inshop/img.zip", "/tmp/img.zip")  # local copy: random member reads
    print(f"img.zip copied in {time.time() - t:.0f}s", flush=True)
    pairs = read_pairs(ensure_pair_files(Path(PROJECT, "datasets", "pose_transfer_pairs"), False)["test"])
    protected = {item_of(s) for p in pairs for s in p}
    parsing = find_archive("parsing", Path(PROJECT, "datasets", "DeepFashion-MultiModal", "archives"))
    clothing = build_backend("schp_lip", cfg, role="clothing", use_cache=False)
    body = build_backend("schp_pascal", cfg, role="body", use_cache=False)
    out = Path("/tmp/pack_build")
    report = build_pack(Path("/tmp/img.zip"), Path(f"{VOL}/raw/inshop/list_eval_partition.txt"), parsing,
                        Path(cfg.paths.cache_dir) / "splits" / "deepfashion_multimodal_seed0.json", protected,
                        clothing, body, out, cfg.labels.body_label_set, batch_size=batch_size, num_workers=7,
                        limit=limit or None)
    report["gpu"] = torch.cuda.get_device_name(0)
    dest = Path(PACK if not limit else PACK + f"_limit{limit}")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(out, dest)
    vol.commit()
    return report


# ------------------------------------------------------------------------------------------------ training (GPU)
def _trainer(stage: str, cfg):
    from src.training.train_texture import TextureTrainer
    from src.training.train_warp import WarpTrainer

    return (WarpTrainer if stage == "warp" else TextureTrainer)(cfg)


@app.function(**COMMON, gpu="H100", cpu=4, memory=24576, timeout=20 * 60)
def probe_gpu(steps: int = 40, batch_size: int = 16, pack: str = "pack256") -> dict:
    """Seconds per step and peak memory of both stages at the real settings (nothing is saved)."""
    _code()
    import torch

    _stage_pack(f"{VOL}/{pack}")
    res = {"gpu": torch.cuda.get_device_name(0), "pack": pack}
    for stage in ("warp", "texture"):
        cfg = _cfg("modal_full_256.yaml", [f"experiment.name=probe", "paths.drive_root=/tmp/probe",
                                           f"paths.pretrained_dir={PROJECT}/pretrained", f"training.batch_size={batch_size}",
                                           "training.val_every=100000", "training.checkpoint_every=100000",
                                           "training.log_every=10", "training.val_max_batches=1"])
        tr = _trainer(stage, cfg)
        tr.save = lambda is_best=False: None  # the probe never writes checkpoints
        torch.cuda.reset_peak_memory_stats()
        times = []
        orig = tr.update_ema

        def timed():
            torch.cuda.synchronize()
            times.append(time.perf_counter())
            orig()

        tr.update_ema = timed
        tr.fit(max_steps=steps, resume=False)
        d = sorted(b - a for a, b in zip(times[10:], times[11:]))  # skip warm-up / cudnn autotune
        res[stage] = {"s_per_step_median": round(d[len(d) // 2], 3), "s_per_step_mean": round(sum(d) / len(d), 3),
                      "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1), "batch": batch_size}
        del tr
        torch.cuda.empty_cache()
    print(json.dumps(res, indent=1))
    return res


@app.function(**COMMON, gpu="H100", cpu=6, memory=32768, timeout=24 * 3600,
              retries=modal.Retries(max_retries=3, initial_delay=10.0))
def train_stage(stage: str, max_steps: int, time_limit_min: float = 0, config: str = "modal_full_256.yaml",
                overrides: str = "") -> dict:
    """Train (or resume) one stage. Every checkpoint is committed to the volume, so a restart (retry, preemption,
    or a later call) continues exactly where it stopped."""
    _code()
    import torch

    _stage_pack()
    vol.reload()
    cfg = _cfg(config, [o for o in overrides.split(";") if o])
    tr = _trainer(stage, cfg)
    tr.on_save = vol.commit
    print(f"GPU {torch.cuda.get_device_name(0)}; {stage}: {len(tr.train_ds)} training samples", flush=True)
    out = tr.fit(max_steps=max_steps, resume=True, time_limit_min=time_limit_min or None)
    from src.training.checkpointing import export_generator

    out["generator_only"] = str(export_generator(tr.ckpt.last))
    vol.commit()
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------------ benchmark (GPU)
@app.function(**COMMON, gpu="L4", cpu=6, memory=24576, timeout=4 * 3600)
def run_benchmark(which: str = "full_256", limit: int = 0, pairs_from: str = "", tag: str = "", overrides: str = "") -> str:
    """Paper-style benchmark on the standard pose-transfer pairs, photos from the packed In-shop store."""
    _code()
    _stage_pack()
    from scripts.pose_transfer_benchmark import main as bench

    if which == "b0":
        ck = f"{PROJECT}/checkpoints/baseline_128"
        args = ["--config", "configs/colab_pairfilter.yaml", "--drive-root", PROJECT,
                "--override", f"evaluation.warp_checkpoint={ck}/warp_pf/generator_only.pt",
                "--override", f"evaluation.texture_checkpoint={ck}/texture_pf/generator_only.pt"]
    else:
        args = ["--config", f"configs/{CONFIGS[which]}", "--drive-root", PROJECT,
                "--override", f"data.split_file={LOCAL_PACK}/splits.json",
                "--override", f"data.packed_dir={LOCAL_PACK}"]
    out = Path(PROJECT, "outputs", "baseline_128" if which == "b0" else which, "pose_transfer",
               time.strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else ""))
    args += ["--packed-dir", LOCAL_PACK, "--out", str(out), "--override", "segmentation.cache_outputs=false"]
    if limit:
        args += ["--limit", str(limit)]
    for o in (o for o in overrides.split(";") if o):
        args += ["--override", o]
    if pairs_from:
        args += ["--pairs-from", pairs_from if pairs_from.startswith("/") else f"{PROJECT}/{pairs_from}"]
    bench(args)
    vol.commit()
    return str(out)


@app.function(**COMMON, cpu=4, memory=16384, timeout=3 * 3600)
def add_densepose_maps(source: str = "dfmm", allow_download: bool = False) -> dict:
    """Add DensePose body maps to /vol/pack256 (blob body_dp). ``dfmm``: DeepFashion-MultiModal densepose archive
    (5.6 GB, official link; downloaded once into datasets/ only with allow_download)."""
    _code()
    from src.data.fetch import download_archive
    from src.data.packing import add_densepose

    if source != "dfmm":
        raise ValueError("only the DFMM archive is wired up here")
    z = download_archive("densepose", Path(PROJECT, "datasets", "DeepFashion-MultiModal", "archives"), allow_download)
    vol.commit()
    local = Path("/tmp/densepose.zip")
    shutil.copy(z, local)
    info = add_densepose(Path(PACK), local, "dfmm", "body15")
    vol.commit()
    print(json.dumps(info, indent=1))
    return info


@app.function(**COMMON, gpu="L4", cpu=6, memory=24576, timeout=3600)
def run_layout_eval(which: str = "full_256", overrides: str = "") -> dict:
    """Warp-stage layout accuracy on all validation photos, every mIoU definition (src/evaluation/layout_eval.py).
    Both models are scored on the same validation pairs (packed store, same pair table)."""
    _code()
    _stage_pack()
    from src.data.packed import load_packed
    from src.evaluation.layout_eval import evaluate_layouts, format_layouts
    from src.inference.tryon import load_tryon_models

    if which == "b0":
        ck = f"{PROJECT}/checkpoints/baseline_128"
        cfg = _cfg("colab_pairfilter.yaml", [f"paths.drive_root={PROJECT}",
                                              f"evaluation.warp_checkpoint={ck}/warp_pf/generator_only.pt",
                                              f"evaluation.texture_checkpoint={ck}/texture_pf/generator_only.pt"])
    else:
        cfg = _cfg(CONFIGS[which], [o for o in overrides.split(";") if o])
    models = load_tryon_models(cfg, warp=cfg.evaluation.warp_checkpoint, texture=cfg.evaluation.texture_checkpoint)
    res = evaluate_layouts(models.warp, load_packed(cfg), cfg, models.device, models.untrained_ids)
    res["checkpoints"] = models.info
    text = format_layouts(res)
    out = Path(PROJECT, "outputs", "baseline_128" if which == "b0" else which, "layout_eval")
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out / f"{stamp}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (out / f"{stamp}.md").write_text(text, encoding="utf-8")
    vol.commit()
    print(text)
    return res


@app.function(**COMMON, gpu="L4", cpu=6, memory=24576, timeout=3 * 3600)
def run_val_tryon(which: str = "full_256", overrides: str = "", tag: str = "", max_per_item: int = 6,
                  oracle: str = "none") -> dict:
    """Model selection: fully automatic try-on on pose-change pairs of VALIDATION products (never benchmark pairs),
    whole-image SSIM / L1 / LPIPS like the benchmark. Compare warps / textures with ``overrides`` (checkpoints)."""
    _code()
    _stage_pack()
    import numpy as np

    from src.data.packed import load_packed
    from src.data.pair_quality import PairFilter
    from src.evaluation.final_eval import metric_weights_dir
    from src.evaluation.image_metrics import LPIPSMetric
    from src.evaluation.pose_transfer_eval import PackedBodyBackend, evaluate_pose_transfer, inshop_loader, validation_pairs
    from src.inference.tryon import load_tryon_models

    cfg = _cfg(CONFIGS[which], ["segmentation.cache_outputs=false", *[o for o in overrides.split(";") if o]])
    prep = load_packed(cfg)
    st = prep.store
    pf = PairFilter(cfg.data.pair_filter, cfg.data.pair_table, cfg.data.pair_max_delta_e, cfg.data.pair_color_metric)
    pairs = validation_pairs([n for n in prep.splits["val"] if st.has_body(n)], pf.ok, max_per_item)
    models = load_tryon_models(cfg, warp=cfg.evaluation.warp_checkpoint, texture=cfg.evaluation.texture_checkpoint)
    body = PackedBodyBackend(st, cfg.labels.body_label_set) if cfg.data.body_source != "schp" else None
    w = metric_weights_dir(cfg)
    res = evaluate_pose_transfer(models, cfg, pairs, {p: "clean_val_items" for p in pairs}, None,
                                 LPIPSMetric(models.device, w, False), None, load=inshop_loader(st), body=body,
                                 grid_path=Path("/tmp/grid.png"), oracle=oracle,
                                 gt_parse=lambda stem: np.ascontiguousarray(st.garment(stem)[:, 41:215]),
                                 extra_body=lambda stem: np.ascontiguousarray(st.body_dp(stem)[:, 41:215]))
    res.update({"which": which, "overrides": overrides, "checkpoints": models.info, "pairs": len(pairs)})
    out = Path(PROJECT, "outputs", "selection", time.strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    shutil.copy("/tmp/grid.png", out / "grid.png")
    vol.commit()
    res["oracle"] = oracle
    summary = {k: res.get(k) for k in ("pairs", "ssim_raw", "ssim_common_raw", "l1_raw", "lpips_raw", "ssim_pasted",
                                       "ssim_copy")}
    summary["oracle"] = oracle
    summary.update({"tag": tag, "dir": str(out), "checkpoints": models.info})
    print(json.dumps(summary, indent=1, default=str))
    return summary


@app.function(**COMMON, cpu=1, memory=2048, timeout=600)
def volume_status() -> dict:
    vol.reload()
    res = {}
    for d in sorted(Path(PROJECT, "checkpoints").glob("*/*")):
        if (d / "last.pt").is_file():
            import torch

            st = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
            res[str(d.relative_to(PROJECT))] = {"step": st.get("step"), "best": st.get("best"), "ema": "gen_ema" in st}
    for log in sorted(Path(PROJECT, "logs").glob("*/*_train.jsonl")):
        lines = log.read_text(encoding="utf-8").splitlines()
        vals = [json.loads(l) for l in lines if '"kind": "val"' in l or '"kind":"val"' in l]
        res[str(log.relative_to(PROJECT))] = {"lines": len(lines), "last_train": json.loads(lines[-1]) if lines else None,
                                             "last_val": vals[-1] if vals else None}
    if Path(PACK, "report.json").is_file():
        res["pack_report"] = json.loads(Path(PACK, "report.json").read_text())
    return res


# ------------------------------------------------------------------------------------------------ local entrypoints
@app.local_entrypoint()
def setup():
    print(json.dumps(setup_volume.remote(), indent=1))


@app.local_entrypoint()
def pack(limit: int = 0, gpu: str = "L4"):
    print(json.dumps(pack_store.with_options(gpu=GPU_ALIASES.get(gpu, gpu)).remote(limit), indent=1, default=str))


@app.local_entrypoint()
def probe(gpus: str = "H100,A100-40GB,L40S", steps: int = 40, batch_size: int = 16, pack: str = "pack256"):
    calls = {g: probe_gpu.with_options(gpu=GPU_ALIASES.get(g, g)).spawn(steps, batch_size, pack) for g in gpus.split(",")}
    print(json.dumps({g: c.get() for g, c in calls.items()}, indent=1))


@app.local_entrypoint()
def train(stage: str, max_steps: int, gpu: str = "H100", time_limit_min: float = 0, overrides: str = "", cpu: int = 0,
          which: str = "full_256", wait: bool = False):
    """``--cpu N`` gives the container N cores and N - 1 DataLoader workers (the warp stage is CPU-bound)."""
    if stage not in ("warp", "texture"):
        raise SystemExit("--stage must be warp or texture")
    opts = {"gpu": GPU_ALIASES.get(gpu, gpu)}
    if cpu:
        opts["cpu"] = float(cpu)
        overrides = ";".join(o for o in (overrides, f"data.num_workers={cpu - 1}") if o)
    fn = train_stage.with_options(**opts)
    if wait:
        print(json.dumps(fn.remote(stage, max_steps, time_limit_min, CONFIGS[which], overrides), indent=1))
        return
    # Default: spawn and return. With `modal run --detach` the job then runs on Modal independently of this PC
    # (a .remote() call is cancelled when the local client disconnects, e.g. laptop sleep; seen 2026-10-28).
    call = fn.spawn(stage, max_steps, time_limit_min, CONFIGS[which], overrides)
    print(f"spawned {stage} training: call {call.object_id}; safe to disconnect. Progress: modal_app.py::status")


@app.local_entrypoint()
def benchmark(which: str = "full_256", limit: int = 0, gpu: str = "L4", pairs_from: str = "", tag: str = "",
              overrides: str = ""):
    print(run_benchmark.with_options(gpu=GPU_ALIASES.get(gpu, gpu)).remote(which, limit, pairs_from, tag, overrides))


@app.local_entrypoint()
def layout_eval(which: str = "full_256", overrides: str = ""):
    run_layout_eval.remote(which, overrides)


@app.function(**COMMON, cpu=2, memory=4096, timeout=600)
def overlay_check(n: int = 6) -> bytes:
    """PNG: photo | SCHP-Pascal body map | DensePose body map, for n training photos (alignment check)."""
    _code()
    import io

    import numpy as np
    from PIL import Image

    from src.data.labels import get_label_set
    from src.data.packed import PackedStore

    st = PackedStore(Path(PACK), body_source="densepose")
    names = [n_ for n_ in st.index if st.has_body(n_)][:: 5000][:n]
    pal_b, pal_p = get_label_set("body15").palette_array(), get_label_set("pascal_person_part").palette_array()
    rows = []
    for name in names:
        img = st.image(name)
        dp = pal_b[st.body(name)]
        st.body_source = "schp"
        sp = pal_p[st.body(name)]
        st.body_source = "densepose"
        blend = (img * 0.5 + dp * 0.5).astype(np.uint8)
        rows.append(np.concatenate([img, sp.astype(np.uint8), dp.astype(np.uint8), blend], 1))
    buf = io.BytesIO()
    Image.fromarray(np.concatenate(rows, 0)).save(buf, "PNG")
    return buf.getvalue()


@app.local_entrypoint()
def check_overlay(out: str = "report/private/modal_runs/densepose_check.png"):
    Path(out).write_bytes(overlay_check.remote())
    print("saved", out)


@app.local_entrypoint()
def densepose(allow_download: bool = False):
    print(json.dumps(add_densepose_maps.remote("dfmm", allow_download), indent=1))


@app.local_entrypoint()
def val_tryon(which: str = "full_256", overrides: str = "", tag: str = "", gpu: str = "L4", oracle: str = "none"):
    run_val_tryon.with_options(gpu=GPU_ALIASES.get(gpu, gpu)).remote(which, overrides, tag, 6, oracle)


@app.local_entrypoint()
def status():
    print(json.dumps(volume_status.remote(), indent=1, default=str))
