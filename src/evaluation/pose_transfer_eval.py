"""Paper-style benchmark: standard DeepFashion pose-transfer test pairs, whole 176 × 256 images (Dress-up Table 1 setting).

For every scorable pair (reference photo -> target photo of the same item, see :mod:`src.data.pose_transfer`) the
fully automatic pipeline runs exactly as at try-on time: Stage A from the SCHP networks on both photos (clothing parse
of the reference, body parts and face/hair labels of the target), warp, texture. Outputs are mapped into the
176 × 256 protocol frame and compared with the real target photo over the WHOLE image:

* ``raw``: generator output (primary; no pixel of the target photo is used)
* ``pasted``: with the target's face/hair pixels pasted (D23); this copies ground-truth pixels, reported separately
* ``copy``: the reference photo itself (lower bound)

Metrics: SSIM (Gaussian 11, σ 1.5), L1, LPIPS (AlexNet v0.1, distance, lower is better), and FID/KID of the raw outputs
vs the real target photos. Pairs whose item was used to train our networks are never scored; pairs of our locked test
items are skipped (see :func:`categorise_pairs`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.config import ProjectConfig
from src.data.labels import get_label_set
from src.data.pose_transfer import model_output_to_protocol, to_protocol_frame
from src.data.preprocessing import load_image
from src.evaluation.image_metrics import InceptionFeatures, LPIPSMetric
from src.evaluation.metrics import ssim
from src.inference.tryon import TryOnModels, to_model_size, tryon_core
from src.models.segmentation import build_backend, to_project_labels

SCORED = ("clean_unseen", "clean_val_items")


def _t(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float() / 255


@torch.no_grad()
def evaluate_pose_transfer(models: TryOnModels, cfg: ProjectConfig, pairs: Sequence[Tuple[str, str]],
                           categories: Dict[Tuple[str, str], str], image_dir: Path,
                           lpips_metric: Optional[LPIPSMetric] = None, features: Optional[InceptionFeatures] = None,
                           limit: Optional[int] = None, clothing=None, body=None, grid_path: Optional[Path] = None,
                           grid_rows: int = 8, progress: bool = True,
                           load: Optional[Callable[[str], np.ndarray]] = None) -> Dict:
    """``load(stem) -> RGB uint8`` replaces reading ``image_dir/<stem>.jpg`` (e.g. :func:`inshop_loader`)."""
    from src.evaluation.final_eval import _features, _fid_kid

    G = get_label_set(cfg.labels.garment_label_set)
    clothing = clothing or build_backend(cfg.segmentation.clothing_backend, cfg, role="clothing")
    body = body or build_backend(cfg.segmentation.body_backend, cfg, role="body")
    todo = [p for p in pairs if categories.get(p) in SCORED][: limit or None]
    if not todo:
        raise ValueError("No scorable pairs (all leaked, locked or missing)")
    image_dir = Path(image_dir)
    memo: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def stage_a(stem: str):
        if stem not in memo:
            img = load(stem) if load is not None else load_image(image_dir / f"{stem}.jpg")
            memo[stem] = (img, to_project_labels(clothing.predict(img, key=stem), G.name))
            if len(memo) > 64:  # bounded cache
                memo.pop(next(iter(memo)))
        return memo[stem]

    keys = [f"{m}_{v}" for v in ("raw", "pasted", "copy") for m in ("ssim", "l1", "lpips")]
    scores: Dict[str, List[float]] = {k: [] for k in keys}
    per_cat = {c: [] for c in SCORED}
    fake, real, rows = [], [], []
    it = todo
    if progress:
        from tqdm.auto import tqdm

        it = tqdm(todo, desc="pose-transfer benchmark")
    t0 = time.time()
    for n, (ref, tgt) in enumerate(it):
        ref_full, ref_parse = stage_a(ref)
        tgt_full, tgt_parse = stage_a(tgt)
        tgt_body = to_project_labels(body.predict(tgt_full, key=tgt), cfg.labels.body_label_set)
        ref_img, ref_p, _ = to_model_size(cfg, ref_full, ref_parse)
        tgt_img, tgt_b, tgt_p, info = to_model_size(cfg, tgt_full, tgt_body, tgt_parse)
        core = tryon_core(models, ref_img, ref_p, tgt_img, tgt_b, tgt_p)
        gt = to_protocol_frame(tgt_full)
        out = {"raw": model_output_to_protocol(core["raw"], info), "pasted": model_output_to_protocol(core["result"], info),
               "copy": to_protocol_frame(ref_full)}
        b = _t(gt)
        for v, img in out.items():
            a = _t(img)
            scores[f"ssim_{v}"].append(float(ssim(a, b)[0]))
            scores[f"l1_{v}"].append(float((a - b).abs().mean()))
            if lpips_metric is not None:
                scores[f"lpips_{v}"].append(float(lpips_metric(a, b)[0]))
        per_cat[categories[(ref, tgt)]].append(len(scores["ssim_raw"]) - 1)
        fake.append(out["raw"]); real.append(gt)
        if grid_path is not None and n < grid_rows:
            rows.append(np.concatenate([out["copy"], gt, out["raw"], out["pasted"]], 1))
    res = {"pairs_scored": len(todo), "frame": "176 x 256 whole image", "model_size": list(cfg.data.image_size),
           "seconds": round(time.time() - t0, 1), "lpips": lpips_metric is not None,
           "by_category": {c: len(ix) for c, ix in per_cat.items()}}
    for k, v in scores.items():
        if v:
            res[k] = float(np.mean(v)); res[k + "_std"] = float(np.std(v))
    for c, ix in per_cat.items():
        if ix:
            res[f"ssim_raw_{c}"] = float(np.mean([scores["ssim_raw"][i] for i in ix]))
    if features is not None and len(fake) >= 2:
        first: Dict[str, int] = {}
        for i, (_, t) in enumerate(todo):
            first.setdefault(t, i)  # one copy per distinct real target photo
        real_u = [real[i] for i in sorted(first.values())]
        res["fid_kid_raw_vs_real_targets"] = _fid_kid(cfg, _features(features, fake), _features(features, real_u))
    res["per_pair"] = {"pairs": [list(p) for p in todo], **{k: v for k, v in scores.items() if v}}
    if grid_path is not None and rows:
        from PIL import Image

        Image.fromarray(np.concatenate(rows, 0)).save(grid_path)
        res["grid"] = str(grid_path)
        res["grid_columns"] = "reference (copy) | real target | ours raw | ours + face/hair paste"
    return res


INSHOP_CONTENT_COLS = (41, 215)  # In-shop 256 x 256 photos = DFMM photo letterboxed to 174 x 256 at column 41


def inshop_loader(store) -> Callable[[str], np.ndarray]:
    """Photos from the packed In-shop store, cropped to their 174 x 256 content (same aspect as a DFMM photo), so the
    protocol frame and model input are built exactly as for DFMM photos."""
    a, b = INSHOP_CONTENT_COLS
    return lambda stem: np.ascontiguousarray(store.image(stem)[:, a:b])


def format_pose_transfer(res: Dict, leak: Dict) -> str:
    L = ["# Paper-style pose-transfer benchmark (standard DeepFashion test pairs, whole 176 x 256 images)", "",
         f"Standard test pairs: {leak['standard_test_pairs']}; by leakage category: {leak['pairs_by_category']}.",
         f"Scored here: {res['pairs_scored']} pairs ({res['by_category']}). Our networks run at {res['model_size']} and are "
         "upscaled to 176 x 256, which lowers sharpness-sensitive scores.", "",
         "| variant | SSIM (higher better) | L1 (lower) | LPIPS (lower, distance) |", "|---|---|---|---|"]
    for v, label in (("raw", "ours, raw output (primary)"), ("pasted", "ours + target face/hair pixels (uses GT pixels)"),
                     ("copy", "copy the reference photo (lower bound)")):
        L.append(f"| {label} | {res.get('ssim_' + v, float('nan')):.4f} | {res.get('l1_' + v, float('nan')):.4f} | "
                 f"{res.get('lpips_' + v, float('nan')):.4f} |")
    fk = res.get("fid_kid_raw_vs_real_targets")
    if fk:
        L += ["", f"FID {fk['fid']:.2f}, KID {fk['kid']:.4f} ± {fk['kid_std']:.4f} ({fk['n1']} generated vs {fk['n2']} real "
                  "target photos; FID depends on sample size, see docs/paper_comparison.md)"]
    L += ["", "Paper's Table 1 for reference (protocol unpublished; LPIPS there marked 'higher is better'): Dress-up "
              "0.95 / 0.228 / FID 8.60; ADGAN 0.77 / 0.226 / 16.69; DiOr 0.80 / 0.176 / 14.34."]
    return "\n".join(L)
