"""Phase 7: final evaluation runner (protocol and rules: docs/final_evaluation.md, measurements T1-T10).

The test split is locked: ``split="test"`` raises :class:`TestSplitLockedError` unless ``confirm_test=True``,
and every test run is appended to an audit log (``evaluation.test_log``) *before* any number is computed, so
aborted runs are counted too. Each part's result is written to ``results.json`` as soon as it is done, so a
disconnect loses at most the part that was running.

Parts (``parts=`` selects a subset):

* ``parser``      T1   SCHP-LIP vs manual parsing (all annotated images of the split)
* ``warp``        T2/3 warp mIoU (7 trained classes, deterministic mixed triplets) + swap metrics
* ``paired``      T4-7 real cross-view pairs: gt_layout / e2e (Stage A from annotations) / e2e_auto (Stage A from
                       SCHP networks) / copy; SSIM, L1, LPIPS on the ground-truth garment + skin region
* ``realism``     T8   try-on with a reference from a DIFFERENT item, fully automatic Stage A; FID/KID vs real images
* ``noise_floor`` T9   real images of another split vs real images of this split; FID/KID
* ``timing``      T10  per-stage latency (uncached parsers) and peak GPU memory
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

from src.config import ProjectConfig
from src.data.body_maps import BodyMapSource
from src.data.dataset import PreparedData
from src.data.labels import colorize, get_label_set
from src.data.preprocessing import image_to_tensor, load_image, resize_labels, tensor_to_image
from src.data.texture_dataset import TextureDataset
from src.data.warp_dataset import IGNORE, SwapDiagnosticDataset, WarpTripletDataset
from src.evaluation.end_to_end import _region_metrics
from src.evaluation.evaluate import evaluate_clothing_parser
from src.evaluation.image_metrics import (InceptionFeatures, LPIPSMetric, frechet_distance,
                                          kernel_inception_distance)
from src.evaluation.metrics import ConfusionMatrix
from src.inference.tryon import TryOnModels, _clock, _one_hot, to_model_size, tryon_core
from src.models.segmentation import build_backend, to_project_labels
from src.utils.paths import ProjectPaths

PARTS = ("parser", "warp", "paired", "realism", "noise_floor", "timing")


class TestSplitLockedError(PermissionError):
    __test__ = False  # not a pytest test class


# ---------------------------------------------------------------------------------------------- guard + audit log
def guard_split(split: str, confirm_test: bool) -> None:
    if split == "test" and not confirm_test:
        raise TestSplitLockedError(
            "The test split is reserved for the final, logged evaluation (docs/final_evaluation.md). "
            "Use split='val' for model decisions, or pass confirm_test=True (notebook: FINAL_SPLIT='test')."
        )


def test_log_path(cfg: ProjectConfig) -> Path:
    return Path(cfg.evaluation.test_log) if cfg.evaluation.test_log else Path(cfg.paths.cache_dir) / "test_split_log.jsonl"


def append_test_log(cfg: ProjectConfig, entry: Dict) -> Path:
    path = test_log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return path


def read_test_log(cfg: ProjectConfig) -> List[Dict]:
    path = test_log_path(cfg)
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[2], timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def metric_weights_dir(cfg: ProjectConfig) -> Path:
    return Path(cfg.evaluation.metric_weights_dir) if cfg.evaluation.metric_weights_dir else Path(cfg.paths.pretrained_dir) / "torch"


# ---------------------------------------------------------------------------------------------- helpers
def _progress(it, on: bool, desc: str):
    if not on:
        return it
    from tqdm.auto import tqdm

    return tqdm(it, desc=desc)


class _AutoParser:
    """Stage A clothing parse from the configured network backend, letterboxed to model size (memoised)."""

    def __init__(self, cfg: ProjectConfig, root: Path, backend=None):
        self.cfg, self.root = cfg, Path(root)
        self.backend = backend or build_backend(cfg.segmentation.clothing_backend, cfg, role="clothing")
        self.G = get_label_set(cfg.labels.garment_label_set)
        self.size, self.mode = tuple(cfg.data.image_size), cfg.data.resize_mode
        self._memo: Dict[str, np.ndarray] = {}

    def __call__(self, rec) -> np.ndarray:
        if rec.name not in self._memo:
            full = load_image(self.root / rec.image)
            lab = to_project_labels(self.backend.predict(full, key=rec.name), self.G.name)
            self._memo[rec.name] = resize_labels(lab, self.size, self.mode)[0]
        return self._memo[rec.name]


def _features(extractor: InceptionFeatures, images: List[np.ndarray], batch: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch):
        t = torch.from_numpy(np.stack(images[i:i + batch])).permute(0, 3, 1, 2).float() / 255
        out.append(extractor(t))
    return np.concatenate(out, 0)


def _fid_kid(cfg: ProjectConfig, f1: np.ndarray, f2: np.ndarray) -> Dict:
    kid, kid_std = kernel_inception_distance(f1, f2, cfg.evaluation.kid_subsets, cfg.evaluation.kid_subset_size,
                                             seed=cfg.experiment.seed)
    return {"fid": frechet_distance(f1, f2), "kid": kid, "kid_std": kid_std, "n1": len(f1), "n2": len(f2),
            "kid_subset_size": min(cfg.evaluation.kid_subset_size, len(f1), len(f2))}


# ---------------------------------------------------------------------------------------------- T1
def eval_parser(cfg: ProjectConfig, prepared: PreparedData, split: str, limit: Optional[int] = None,
                backend=None, progress: bool = True) -> Dict:
    backend = backend or build_backend(cfg.segmentation.clothing_backend, cfg, role="clothing")
    return evaluate_clothing_parser(backend, prepared, cfg, split=split, limit=limit, progress=progress)


# ---------------------------------------------------------------------------------------------- T2 / T3
@torch.no_grad()
def eval_warp(models: TryOnModels, prepared: PreparedData, cfg: ProjectConfig, split: str,
              limit: Optional[int] = None, body_source: Optional[BodyMapSource] = None, progress: bool = True) -> Dict:
    from src.training.train_warp import _collate, class_area_agreement

    G = get_label_set(cfg.labels.garment_label_set)
    ds = WarpTripletDataset(prepared, split, cfg, train=False, body_source=body_source)
    swap = SwapDiagnosticDataset(ds, n=cfg.training.warp_swap_samples, seed=cfg.experiment.seed)
    n = min(len(ds), limit) if limit else len(ds)
    sub = torch.utils.data.Subset(ds, range(n))
    bs = cfg.training.batch_size
    cm = ConfusionMatrix(G.num_classes, ignore_index=IGNORE)

    def predict(batch):
        logits = models.warp(batch["cloth"].to(models.device), batch["body"].to(models.device)).float()
        logits[:, models.untrained_ids] = -torch.inf
        return logits.argmax(1).cpu().numpy()

    modes = {"self": 0, "cross_view": 0}
    for batch in _progress(torch.utils.data.DataLoader(sub, batch_size=bs, collate_fn=_collate), progress, "warp eval"):
        cm.update(predict(batch), batch["target"].numpy())
        for m in batch["mode"]:
            modes[m] = modes.get(m, 0) + 1
    transfer = G.indices(cfg.labels.transfer_classes)
    hits = total = 0
    areas = []
    if len(swap):
        for batch in torch.utils.data.DataLoader(swap, batch_size=bs, collate_fn=_collate):
            pred = predict(batch)
            present = batch["cloth"].flatten(2).amax(2).numpy() > 0
            for b in range(pred.shape[0]):
                garment = np.isin(pred[b], transfer)
                total += int(garment.sum())
                hits += int((garment & present[b][pred[b]]).sum())
                areas.append(class_area_agreement(batch["cloth"][b].numpy(), pred[b], transfer))
    areas = [a for a in areas if not np.isnan(a)]
    excl = [0] + list(models.untrained_ids)
    summary = cm.summary(G.names, exclude=excl)
    per_class = {k: v for k, v in summary["per_class_iou"].items() if k not in {G.names[i] for i in excl}}
    return {"n_images": n, "pairing": modes, "metric_version": 2, "mean_iou_7": summary["mean_iou"],
            "pixel_accuracy": summary["pixel_accuracy"], "per_class_iou": per_class,
            "swap_pairs": len(swap), "class_fidelity": hits / total if total else float("nan"),
            "class_area_agreement": float(np.mean(areas)) if areas else float("nan")}


# ---------------------------------------------------------------------------------------------- T4 - T7
@torch.no_grad()
def eval_paired(models: TryOnModels, prepared: PreparedData, cfg: ProjectConfig, split: str,
                lpips_metric: Optional[LPIPSMetric] = None, auto_parser: Optional[_AutoParser] = None,
                limit: Optional[int] = None, grid_path: Optional[Path] = None, grid_rows: int = 8,
                progress: bool = True) -> Dict:
    """Real cross-view pairs (same item, pair filter applies); the target photo is the ground truth.

    Variants: ``gt_layout`` (T4), ``e2e_raw``/``e2e`` (T5, Stage A = manual parsing; raw = before the face/hair paste),
    ``auto_raw``/``auto`` (T6, Stage A = SCHP-LIP for both photos), ``copy`` (T7). The body map is the precomputed
    ``segmentation.body_backend`` map in both e2e variants. Region: garment + skin of the MANUAL target parse.
    """
    G = get_label_set(cfg.labels.garment_label_set)
    ds = TextureDataset(prepared, split, cfg, train=False, cross_view_prob=1.0)
    body_src = BodyMapSource(cfg, prepared.index.root)
    by_name = {r.name: i for i, r in enumerate(ds.records)}
    idx = [i for i in range(len(ds)) if ds.siblings[i]][: limit or None]
    if not idx:
        raise ValueError(f"No cross-view pairs in split '{split}' (check data.pair_filter)")
    region_ids = G.indices(list(cfg.labels.transfer_classes) + ["skin"])
    variants = ["gt_layout", "e2e_raw", "e2e", "copy"] + (["auto_raw", "auto"] if auto_parser else [])
    scores = {v: {"ssim": [], "l1": [], "lpips": []} for v in variants}
    cm_e2e = ConfusionMatrix(G.num_classes, ignore_index=255)
    cm_auto = ConfusionMatrix(G.num_classes, ignore_index=255)
    rows, pair_names = [], []
    t0 = time.time()
    for n, i in enumerate(_progress(idx, progress, "paired eval")):
        s = ds[i]
        pair_names.append([s["src"], s["tgt"]])
        ref_img, real = tensor_to_image(s["exemplar"]), tensor_to_image(s["image"])
        ref_parse = s["exemplar_parsing"].argmax(0).numpy().astype(np.uint8)
        gt_layout = s["layout"].argmax(0).numpy().astype(np.uint8)
        body = resize_labels(body_src.load(ds.records[i]), tuple(cfg.data.image_size), cfg.data.resize_mode)[0]
        core = tryon_core(models, ref_img, ref_parse, real, body, gt_layout)
        out_gt = models.texture(_one_hot(gt_layout, G.num_classes, models.device),
                                image_to_tensor(ref_img)[None].to(models.device),
                                _one_hot(ref_parse, G.num_classes, models.device))
        imgs = {"gt_layout": tensor_to_image(out_gt["fake"][0].float().cpu()), "e2e_raw": core["raw"],
                "e2e": core["result"], "copy": ref_img}
        if auto_parser:
            ref_rec = ds.records[by_name[s["src"]]]
            auto = tryon_core(models, ref_img, auto_parser(ref_rec), real, body, auto_parser(ds.records[i]))
            imgs["auto_raw"], imgs["auto"] = auto["raw"], auto["result"]
        region = np.isin(gt_layout, region_ids)
        for v in variants:
            m = _region_metrics(imgs[v], real, region)
            scores[v]["ssim"].append(m["ssim"]); scores[v]["l1"].append(m["l1"])
            if lpips_metric is not None:
                a = torch.from_numpy(imgs[v]).permute(2, 0, 1)[None].float() / 255
                b = torch.from_numpy(real).permute(2, 0, 1)[None].float() / 255
                scores[v]["lpips"].append(float(lpips_metric(a, b, torch.from_numpy(region)[None])[0]))
        tgt = gt_layout.astype(np.int64).copy()
        tgt[~np.isin(tgt, region_ids + [0])] = 255
        cm_e2e.update(core["c_warp"], tgt)
        if auto_parser:
            cm_auto.update(auto["c_warp"], tgt)
        if grid_path is not None and n < grid_rows:
            tiles = [ref_img, real, colorize(core["c_warp"], G), imgs["gt_layout"], imgs["e2e"]]
            if auto_parser:
                tiles += [colorize(auto["c_warp"], G), imgs["auto"]]
            rows.append(np.concatenate(tiles, 1))
    excl = [0] + list(models.untrained_ids)
    out = {"pairs": len(idx), "pair_filter": cfg.data.pair_filter, "region": "garment + skin of the manual target parse",
           "warp_miou_7_e2e": cm_e2e.mean_iou(exclude=excl), "seconds": round(time.time() - t0, 1),
           "lpips": lpips_metric is not None}
    if auto_parser:
        out["warp_miou_7_auto"] = cm_auto.mean_iou(exclude=excl)
    for v, d in scores.items():
        for k, vals in d.items():
            if vals:
                out[f"{v}_{k}"] = float(np.mean(vals))
                out[f"{v}_{k}_std"] = float(np.std(vals))
    # per-pair scores (same order as ``pairs``) so later experiments can be compared on the same pairs
    out["per_pair"] = {"pairs": pair_names, **{v: {k: vals for k, vals in d.items() if vals} for v, d in scores.items()}}
    if grid_path is not None and rows:
        Image.fromarray(np.concatenate(rows, 0)).save(grid_path)
        out["grid"] = str(grid_path)
        out["grid_columns"] = ("reference | ground truth | C'_seg (e2e) | gt_layout | e2e"
                               + (" | C'_seg (auto) | auto" if auto_parser else ""))
    return out


def paired_bootstrap_ci(a, b, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> Dict[str, float]:
    """Mean of (b - a) over the same pairs with a percentile bootstrap interval (resampling pairs).

    Used to compare an experiment (b) with the baseline (a) on identical pairs (docs/experiments.md)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("paired_bootstrap_ci needs two equal-length 1-D score lists (>= 2 pairs)")
    d = b - a
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(d))}


# ---------------------------------------------------------------------------------------------- T8 / T9
@torch.no_grad()
def eval_realism(models: TryOnModels, prepared: PreparedData, cfg: ProjectConfig, split: str,
                 features: InceptionFeatures, auto_parser: _AutoParser, limit: Optional[int] = None,
                 grid_path: Optional[Path] = None, grid_rows: int = 8, progress: bool = True) -> Dict:
    """Each image of the split is a target once; its reference is a random photo of a DIFFERENT item.
    Fully automatic Stage A (clothing network + precomputed body maps). Generated results vs the real images."""
    ds = TextureDataset(prepared, split, cfg, train=False, cross_view_prob=0.0)
    body_src = BodyMapSource(cfg, prepared.index.root)
    rng = np.random.default_rng(cfg.experiment.seed)
    n = min(len(ds), limit) if limit else len(ds)
    real, fake, rows = [], [], []
    t0 = time.time()
    for t in _progress(range(n), progress, "realism (try-on)"):
        rec_t = ds.records[t]
        cand = [j for j in range(len(ds)) if ds.records[j].item_id != rec_t.item_id]
        if not cand:
            continue
        r = int(cand[rng.integers(0, len(cand))])
        tgt_img, _ = ds._load(rec_t)
        ref_img, _ = ds._load(ds.records[r])
        body = resize_labels(body_src.load(rec_t), tuple(cfg.data.image_size), cfg.data.resize_mode)[0]
        core = tryon_core(models, ref_img, auto_parser(ds.records[r]), tgt_img, body, auto_parser(rec_t))
        real.append(tgt_img); fake.append(core["result"])
        if grid_path is not None and len(rows) < grid_rows:
            rows.append(np.concatenate([ref_img, tgt_img, core["raw"], core["result"]], 1))
    if len(fake) < 2:
        raise ValueError(f"Split '{split}' has too few images of different items for the realism test")
    out = _fid_kid(cfg, _features(features, fake), _features(features, real))
    out.update({"generated": "try-on result (face/hair pasted), reference from a different item",
                "real": f"the {len(real)} real {split} images (model size)", "seconds": round(time.time() - t0, 1)})
    if grid_path is not None and rows:
        Image.fromarray(np.concatenate(rows, 0)).save(grid_path)
        out["grid"] = str(grid_path)
        out["grid_columns"] = "reference (other item) | target | raw | result"
    return out


def eval_noise_floor(prepared: PreparedData, cfg: ProjectConfig, split: str, features: InceptionFeatures,
                     other_split: Optional[str] = None, limit: Optional[int] = None) -> Dict:
    """Real vs real: what FID/KID two sets of REAL photos reach at this sample size (the best achievable)."""
    other = other_split or ("val" if split == "test" else "train")
    a = TextureDataset(prepared, other, cfg, train=False, cross_view_prob=0.0)
    b = TextureDataset(prepared, split, cfg, train=False, cross_view_prob=0.0)
    na, nb = (min(len(a), limit), min(len(b), limit)) if limit else (len(a), len(b))
    fa = _features(features, [a._load(a.records[i])[0] for i in range(na)])
    fb = _features(features, [b._load(b.records[i])[0] for i in range(nb)])
    out = _fid_kid(cfg, fa, fb)
    out["sets"] = f"real {other} vs real {split}"
    return out


# ---------------------------------------------------------------------------------------------- T10
@torch.no_grad()
def measure_timing(models: TryOnModels, prepared: PreparedData, cfg: ProjectConfig, split: str,
                   runs: Optional[int] = None, warmup: int = 3) -> Dict:
    """Single-image latency of every stage. Parsers are built WITHOUT the disk cache, so Stage A is really run
    (the key is only used by annotation backends in tests)."""
    runs = runs or cfg.evaluation.timing_runs
    recs = prepared.records(split)
    if len(recs) < 2:
        raise ValueError("timing needs at least two images")
    cloth = build_backend(cfg.segmentation.clothing_backend, cfg, role="clothing", use_cache=False)
    body = build_backend(cfg.segmentation.body_backend, cfg, role="body", use_cache=False)
    G = get_label_set(cfg.labels.garment_label_set)
    dev = models.device
    cuda = torch.device(dev).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    keys = ("stage_a_clothing_ref", "stage_a_clothing_target", "stage_a_body", "warp", "texture", "total")
    times: Dict[str, List[float]] = {k: [] for k in keys}
    for k in range(warmup + runs):
        ref, tgt = recs[k % len(recs)], recs[(k + 1) % len(recs)]
        ref_full, tgt_full = load_image(prepared.index.root / ref.image), load_image(prepared.index.root / tgt.image)
        t = {}
        start = c = _clock(dev)
        ref_parse = to_project_labels(cloth.predict(ref_full, key=ref.name), G.name)
        t["stage_a_clothing_ref"], c = _clock(dev) - c, _clock(dev)
        tgt_parse = to_project_labels(cloth.predict(tgt_full, key=tgt.name), G.name)
        t["stage_a_clothing_target"], c = _clock(dev) - c, _clock(dev)
        tgt_body = to_project_labels(body.predict(tgt_full, key=tgt.name), cfg.labels.body_label_set)
        t["stage_a_body"] = _clock(dev) - c
        ref_img, ref_p, _ = to_model_size(cfg, ref_full, ref_parse)
        tgt_img, tgt_b, tgt_p, _ = to_model_size(cfg, tgt_full, tgt_body, tgt_parse)
        tryon_core(models, ref_img, ref_p, tgt_img, tgt_b, tgt_p, timings=t)
        t["total"] = _clock(dev) - start
        if k >= warmup:
            for key in keys:
                times[key].append(t[key] * 1000)
    out = {f"{k}_ms": float(np.median(v)) for k, v in times.items()}
    out.update({f"{k}_ms_p90": float(np.percentile(v, 90)) for k, v in times.items()})
    out.update({"runs": runs, "warmup": warmup, "device": str(dev), "image_size": list(cfg.data.image_size),
                "clothing_backend": cfg.segmentation.clothing_backend, "body_backend": cfg.segmentation.body_backend,
                "note": "median over runs; total includes letterboxing; parsers run at their own input size"})
    if cuda:
        out["peak_gpu_mem_mib"] = round(torch.cuda.max_memory_allocated(dev) / 2**20, 1)
        out["gpu"] = torch.cuda.get_device_name(dev)
    return out


# ---------------------------------------------------------------------------------------------- runner
def run_final_evaluation(cfg: ProjectConfig, prepared: PreparedData, models: TryOnModels, split: str = "test",
                         confirm_test: bool = False, parts: Iterable[str] = PARTS, out_dir: Optional[Path] = None,
                         limit: Optional[int] = None, lpips_random_backbone: bool = False,
                         feature_extractor=None, progress: bool = True) -> Dict:
    """Run the selected parts and write ``results.json`` + ``results_table.md`` (+ grids) into ``out_dir``.

    ``limit`` caps images/pairs per part (smoke runs only; the final run uses ``limit=None``).
    ``lpips_random_backbone`` / ``feature_extractor``: offline test hooks, never for reported numbers.
    """
    parts = list(parts)
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise ValueError(f"Unknown parts {sorted(unknown)}; choose from {PARTS}")
    guard_split(split, confirm_test)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir) if out_dir else ProjectPaths.from_config(cfg).outputs / cfg.experiment.name / "final_eval" / f"{split}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    official = limit is None and not lpips_random_backbone and feature_extractor is None
    results: Dict = {"split": split, "started": stamp, "commit": _git_commit(), "official": official,
                     "checkpoints": models.info, "config": {"experiment": cfg.experiment.name,
                     "image_size": list(cfg.data.image_size), "pair_filter": cfg.data.pair_filter,
                     "pair_max_delta_e": cfg.data.pair_max_delta_e, "clothing_backend": cfg.segmentation.clothing_backend,
                     "body_backend": cfg.segmentation.body_backend, "warp_run": cfg.training.warp_run_name,
                     "texture_run": cfg.training.texture_run_name}, "limit": limit, "parts": parts}
    if split == "test":
        previous = [e for e in read_test_log(cfg) if e.get("event") == "start"]
        results["previous_test_runs"] = len(previous)
        append_test_log(cfg, {"event": "start", "time": stamp, "commit": results["commit"], "parts": parts,
                              "limit": limit, "official": official, "checkpoints": models.info, "out_dir": str(out_dir)})

    def save():
        (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
        (out_dir / "results_table.md").write_text(format_results_table(results), encoding="utf-8")

    weights = metric_weights_dir(cfg)
    allow = cfg.evaluation.allow_metric_download
    auto = None
    if {"paired", "realism"} & set(parts):
        auto = _AutoParser(cfg, prepared.index.root)
    features = None
    if {"realism", "noise_floor"} & set(parts):
        features = InceptionFeatures(models.device, weights, allow, extractor=feature_extractor)
    for part in parts:
        print(f"[final eval] {part} ...", flush=True)
        t0 = time.time()
        if part == "parser":
            res = eval_parser(cfg, prepared, split, limit, progress=progress)
        elif part == "warp":
            res = eval_warp(models, prepared, cfg, split, limit, progress=progress)
        elif part == "paired":
            lp = LPIPSMetric(models.device, weights, allow, random_backbone=lpips_random_backbone)
            res = eval_paired(models, prepared, cfg, split, lp, auto, limit, out_dir / "paired_grid.png", progress=progress)
        elif part == "realism":
            res = eval_realism(models, prepared, cfg, split, features, auto, limit, out_dir / "realism_grid.png", progress=progress)
        elif part == "noise_floor":
            res = eval_noise_floor(prepared, cfg, split, features, limit=limit)
        else:
            res = measure_timing(models, prepared, cfg, split, runs=min(cfg.evaluation.timing_runs, limit or 10**9))
        res["part_seconds"] = round(time.time() - t0, 1)
        results[part] = res
        save()
    results["finished"] = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save()
    if split == "test":
        append_test_log(cfg, {"event": "finished", "time": results["finished"], "started": stamp, "out_dir": str(out_dir)})
    results["out_dir"] = str(out_dir)
    return results


def _f(x, nd=3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def format_results_table(r: Dict) -> str:
    """Markdown summary of one run (ours only; the paper's numbers are in report/results_table.md)."""
    L = [f"# Final evaluation: split `{r['split']}` ({'official' if r.get('official') else 'NOT official: limited/test hooks'})",
         "", f"commit {r.get('commit')}, started {r.get('started')}, checkpoints "
         + ", ".join(f"{k} step {v.get('step')}" for k, v in (r.get("checkpoints") or {}).items()), ""]
    if "parser" in r:
        p = r["parser"]
        L += ["## T1 Stage A clothing parser vs manual parsing", "",
              f"{p['n_images']} images: pixel accuracy {_f(p['pixel_accuracy'])}, mIoU {_f(p['mean_iou'])}", ""]
    if "warp" in r:
        w = r["warp"]
        L += ["## T2/T3 Warp", "", f"{w['n_images']} images {w['pairing']}: mIoU (7 classes) {_f(w['mean_iou_7'])}, "
              f"pixel acc {_f(w['pixel_accuracy'])}; swap ({w['swap_pairs']} pairs): class-area agreement "
              f"{_f(w['class_area_agreement'])}, class fidelity {_f(w['class_fidelity'])}", "",
              "| class | IoU |", "|---|---|"] + [f"| {k} | {_f(v)} |" for k, v in w["per_class_iou"].items()] + [""]
    if "paired" in r:
        p = r["paired"]
        L += [f"## T4-T7 Paired try-on ({p['pairs']} real cross-view pairs, region: {p['region']})", "",
              "| variant | SSIM (higher better) | L1 (lower) | LPIPS (lower) |", "|---|---|---|---|"]
        names = {"gt_layout": "T4 texture with true layout", "e2e_raw": "T5 end-to-end, manual Stage A (raw)",
                 "e2e": "T5 end-to-end, manual Stage A (+face/hair paste)", "auto_raw": "T6 fully automatic (raw)",
                 "auto": "T6 fully automatic (+paste)", "copy": "T7 copy reference photo"}
        for v, label in names.items():
            if f"{v}_ssim" in p:
                L.append(f"| {label} | {_f(p[f'{v}_ssim'])} | {_f(p[f'{v}_l1'])} | {_f(p.get(f'{v}_lpips'))} |")
        L += ["", f"warp mIoU on these pairs: manual Stage A {_f(p.get('warp_miou_7_e2e'))}, automatic "
              f"{_f(p.get('warp_miou_7_auto'))}", ""]
    for key, title in (("realism", "T8 Realism of try-on results (different-item references)"),
                       ("noise_floor", "T9 Real-vs-real noise floor")):
        if key in r:
            x = r[key]
            L += [f"## {title}", "", f"FID {_f(x['fid'], 2)}, KID {_f(x['kid'], 4)} ± {_f(x['kid_std'], 4)} "
                  f"({x['n1']} vs {x['n2']} images, KID subsets of {x['kid_subset_size']})", ""]
    if "timing" in r:
        t = r["timing"]
        L += [f"## T10 Cost ({t['runs']} runs, {t.get('gpu', t['device'])}, {t['image_size']} px)", "",
              "| stage | median ms | p90 ms |", "|---|---|---|"]
        for k in ("stage_a_clothing_ref", "stage_a_clothing_target", "stage_a_body", "warp", "texture", "total"):
            L.append(f"| {k} | {_f(t[k + '_ms'], 1)} | {_f(t[k + '_ms_p90'], 1)} |")
        L += ["", f"peak GPU memory: {t.get('peak_gpu_mem_mib', 'n/a')} MiB", ""]
    return "\n".join(L)
