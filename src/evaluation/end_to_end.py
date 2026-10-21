"""End-to-end evaluation of the full try-on on paired validation data (Phase 6).

Pairs: two photos of the same item (val split, ``data.pair_filter`` applies). The reference (exemplar) is
photo s, the target is photo t, and photo t is also the ground truth, because the person in t already
wears that garment.

Stage A inputs come from the dataset (manual parsing, precomputed body maps), so this measures Stages B + C
and their interaction, not parser errors. Variants scored on the same pairs:

* ``e2e``: warp-predicted layout -> texture   (the real pipeline)
* ``gt_layout``: ground-truth layout -> texture (upper bound for the texture stage; e2e minus this = cost of warp errors)
* ``copy``: the exemplar photo unchanged       (lower bound)

Region: **garment + skin pixels of the ground truth** (face and hair excluded, because the pipeline pastes the
target's own face/hair pixels, which would leak the answer). The warp layout is also scored (mIoU, 7 trained
classes) on the same pairs.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from src.config import ProjectConfig
from src.data.body_maps import BodyMapSource
from src.data.dataset import PreparedData
from src.data.labels import colorize, get_label_set, inject_identity_labels
from src.data.preprocessing import image_to_tensor, resize_labels, tensor_to_image
from src.data.texture_dataset import TextureDataset
from src.evaluation.metrics import ConfusionMatrix, ssim
from src.inference.tryon import TryOnModels, _one_hot, tryon_core


def _region_metrics(img: np.ndarray, real: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    a = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    b = torch.from_numpy(real).permute(2, 0, 1)[None].float() / 255
    m = torch.from_numpy(mask)[None]
    s = float(ssim(a, b, mask=m)[0])
    l1 = float(((a - b).abs().mean(1)[0][m[0]]).mean()) if mask.any() else float("nan")
    return {"ssim": s, "l1": l1}


@torch.no_grad()
def evaluate_end_to_end(models: TryOnModels, prepared: PreparedData, cfg: ProjectConfig, split: str = "val",
                        limit: Optional[int] = None, grid_path=None, grid_rows: int = 6, progress: bool = True) -> Dict:
    G = get_label_set(cfg.labels.garment_label_set)
    ds = TextureDataset(prepared, split, cfg, train=False, cross_view_prob=1.0)
    body_src = BodyMapSource(cfg, prepared.index.root)
    idx = [i for i in range(len(ds)) if ds.siblings[i]][: limit or None]
    if not idx:
        raise ValueError(f"No cross-view pairs in split '{split}' (check data.pair_filter)")
    region_ids = G.indices(list(cfg.labels.transfer_classes) + ["skin"])
    cm = ConfusionMatrix(G.num_classes, ignore_index=255)
    scores = {k: {"ssim": [], "l1": []} for k in ("e2e", "e2e_pasted", "gt_layout", "copy")}
    rows, t0 = [], time.time()
    it = idx
    if progress:
        from tqdm.auto import tqdm

        it = tqdm(idx, desc="end-to-end eval")
    for n, i in enumerate(it):
        s = ds[i]
        assert s["mode"] == "cross_view"
        ref_img = tensor_to_image(s["exemplar"]); real = tensor_to_image(s["image"])
        ref_parse = s["exemplar_parsing"].argmax(0).numpy().astype(np.uint8)
        gt_layout = s["layout"].argmax(0).numpy().astype(np.uint8)
        body = resize_labels(body_src.load(ds.records[i]), tuple(cfg.data.image_size), cfg.data.resize_mode)[0]
        core = tryon_core(models, ref_img, ref_parse, real, body, gt_layout)
        # texture with the ground-truth layout (upper bound)
        out_gt = models.texture(_one_hot(gt_layout, G.num_classes, models.device),
                                image_to_tensor(ref_img)[None].to(models.device),
                                _one_hot(ref_parse, G.num_classes, models.device))
        gt_img = tensor_to_image(out_gt["fake"][0].float().cpu())
        region = np.isin(gt_layout, region_ids)
        for key, img in (("e2e", core["raw"]), ("e2e_pasted", core["result"]), ("gt_layout", gt_img), ("copy", ref_img)):
            m = _region_metrics(img, real, region)
            scores[key]["ssim"].append(m["ssim"]); scores[key]["l1"].append(m["l1"])
        tgt = gt_layout.astype(np.int64).copy()
        tgt[~np.isin(tgt, region_ids + [0])] = 255  # score the warp only on classes it is trained for
        cm.update(core["c_warp"], tgt)
        if grid_path is not None and n < grid_rows:
            rows.append(np.concatenate([ref_img, real, colorize(core["c_warp"], G), colorize(core["layout"], G),
                                        core["warped_exemplar"], core["raw"], core["result"], gt_img], 1))
    untrained = [0] + models.untrained_ids
    out = {"split": split, "pairs": len(idx), "pair_filter": cfg.data.pair_filter, "region": "garments+skin of ground truth",
           "warp_miou_7": cm.mean_iou(exclude=untrained), "seconds": round(time.time() - t0, 1)}
    for k, v in scores.items():
        out[f"{k}_ssim"] = float(np.mean(v["ssim"]))
        out[f"{k}_l1"] = float(np.mean(v["l1"]))
    if grid_path is not None and rows:
        Image.fromarray(np.concatenate(rows, 0)).save(grid_path)
        out["grid"] = str(grid_path)
    return out


def tryon_swap_grid(models: TryOnModels, prepared: PreparedData, cfg: ProjectConfig, split: str = "val", n: int = 6,
                    seed: int = 0, path=None):
    """Real try-on between DIFFERENT items (no ground truth): reference | target | C'_seg | I_S | raw | result.

    Returns None (and writes nothing) if the split has fewer than two items."""
    G = get_label_set(cfg.labels.garment_label_set)
    ds = TextureDataset(prepared, split, cfg, train=False, cross_view_prob=0.0)
    body_src = BodyMapSource(cfg, prepared.index.root)
    rng = np.random.default_rng(seed)
    order = [int(i) for i in rng.permutation(len(ds))]
    pairs, used = [], set()
    for a in order:  # greedily pair photos of different items
        for b in order:
            if len(pairs) >= n:
                break
            if a != b and a not in used and b not in used and ds.records[a].item_id != ds.records[b].item_id:
                pairs.append((a, b)); used |= {a, b}
                break
    rows = []
    for a, b in pairs:
        ra, rb = ds.records[a], ds.records[b]
        sa, sb = ds.pair(a, a, rng), ds.pair(b, b, rng)  # undistorted photos + parses
        ref_img, ref_parse = tensor_to_image(sa["image"]), sa["layout"].argmax(0).numpy().astype(np.uint8)
        tgt_img, tgt_parse = tensor_to_image(sb["image"]), sb["layout"].argmax(0).numpy().astype(np.uint8)
        body = resize_labels(body_src.load(rb), tuple(cfg.data.image_size), cfg.data.resize_mode)[0]
        core = tryon_core(models, ref_img, ref_parse, tgt_img, body, tgt_parse)
        rows.append(np.concatenate([ref_img, tgt_img, colorize(core["c_warp"], G), core["warped_exemplar"],
                                    core["raw"], core["result"]], 1))
    img = np.concatenate(rows, 0) if rows else None
    if path is not None and img is not None:
        Image.fromarray(img).save(path)
    return img
