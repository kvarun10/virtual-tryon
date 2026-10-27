"""Garment-layout (warp stage) accuracy on the whole validation split, under every common mIoU definition.

The paper's Table 2 reports mIoU 87.6 / pixel accuracy 98.9 for the layout but does not say which classes or pixels
count. So we report each definition separately, each clearly labelled (never only the most favourable one):

* ``supervised``: the training metric. Pixels the warp is trained on (transfer garments, skin, background);
  mIoU without background (``miou_excl_bg``) and with it (``miou_incl_bg``).
* ``final_layout_auto``: the layout the texture stage actually receives at try-on time, i.e. the warp output plus
  the target's face/hair labels from the automatic parser (paper Sec. 4.2), against the full manual parse.
  All 12 garment12 classes, every pixel. Fully automatic.
* ``final_layout_gt_identity``: the same, but face/hair taken from the target's manual parse (upper bound for the
  identity labels; uses ground truth, reported separately).

Validation pairs are built as in training validation (fixed RNG per index; mix of cross-view and self pairs).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import ProjectConfig
from src.data.labels import get_label_set, inject_identity_labels
from src.data.warp_dataset import IGNORE, WarpTripletDataset
from src.evaluation.metrics import ConfusionMatrix


def _collate(b):
    return {"cloth": torch.stack([x["cloth"] for x in b]), "body": torch.stack([x["body"] for x in b]),
            "target": torch.stack([x["target"] for x in b]), "tgt": [x["tgt"] for x in b], "mode": [x["mode"] for x in b]}


def _miou(cm: ConfusionMatrix, exclude=()) -> float:
    return float(cm.mean_iou(exclude=exclude))


@torch.no_grad()
def evaluate_layouts(warp: torch.nn.Module, prepared, cfg: ProjectConfig, device, untrained_ids,
                     split: str = "val", batch_size: int = 16, limit: Optional[int] = None) -> Dict:
    G = get_label_set(cfg.labels.garment_label_set)
    K = G.num_classes
    ds = WarpTripletDataset(prepared, split, cfg, train=False)
    store = getattr(prepared, "store", None)
    idx = list(range(len(ds)))[: limit or None]
    loader = DataLoader(torch.utils.data.Subset(ds, idx), batch_size=batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers,
                        collate_fn=_collate)
    by_name = {r.name: r for r in ds.records}
    cms = {k: ConfusionMatrix(K, ignore_index=IGNORE) for k in ("supervised", "final_layout_auto", "final_layout_gt_identity")}
    modes = {"self": 0, "cross_view": 0}
    warp.eval()
    for batch in loader:
        logits = warp(batch["cloth"].to(device), batch["body"].to(device)).float()
        logits[:, untrained_ids] = -torch.inf
        pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
        cms["supervised"].update(pred, batch["target"].numpy())
        for k, name in enumerate(batch["tgt"]):
            modes[batch["mode"][k]] = modes.get(batch["mode"][k], 0) + 1
            gt_full = ds._garment(by_name[name])  # manual parse (all classes)
            auto = ds._fit(store.garment_auto(name)) if store is not None else gt_full
            cms["final_layout_auto"].update(inject_identity_labels(pred[k], auto, G, cfg.labels.preserve_classes), gt_full)
            cms["final_layout_gt_identity"].update(inject_identity_labels(pred[k], gt_full, G, cfg.labels.preserve_classes), gt_full)
    bg = [G.index("background")]
    out = {"images": len(idx), "pairs": modes, "image_size": list(cfg.data.image_size)}
    for k, cm in cms.items():
        iou = cm.iou()
        out[k] = {"pixel_accuracy": cm.pixel_accuracy(), "miou_excl_bg": _miou(cm, bg), "miou_incl_bg": _miou(cm),
                  "per_class_iou": {G.names[i]: (None if np.isnan(v) else round(float(v), 4)) for i, v in enumerate(iou)}}
    if store is None:
        out["final_layout_auto"]["note"] = "no packed store: identity labels taken from the manual parse"
    return out


def format_layouts(res: Dict) -> str:
    rows = [("supervised (training metric: garments + skin + background)", "supervised"),
            ("final layout, face/hair from the automatic parser (fully automatic)", "final_layout_auto"),
            ("final layout, face/hair from the manual parse (uses ground truth)", "final_layout_gt_identity")]
    L = [f"# Garment layout accuracy ({res['images']} validation images, pairs {res['pairs']}, {res['image_size']} px)", "",
         "| definition | mIoU excl. background | mIoU incl. background | pixel accuracy |", "|---|---|---|---|"]
    for label, k in rows:
        r = res[k]
        L.append(f"| {label} | {100 * r['miou_excl_bg']:.1f} | {100 * r['miou_incl_bg']:.1f} | {100 * r['pixel_accuracy']:.1f} |")
    L += ["", "Paper Table 2 (definition not published): mIoU 87.6, pixel accuracy 98.9."]
    return "\n".join(L)
