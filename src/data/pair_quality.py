"""Are two photos of the same DeepFashion item really wearing the same outfit?

In DeepFashion an item ID names one product. Other garments in the photo can change between shots
(observed in the texture run of 2026-10-21: a beanie item shown once with a plaid shirt + jeans and once
with a green sweater + black jeans). Cross-view training pairs with different outfits ask the warp and texture
networks to produce garments they cannot see.

For every image we compute a *garment signature* from the manual parsing: for each transfer class, its share
of the person area and its mean colour in CIELAB. Two photos are compared by:

* ``class_jaccard``: |classes in both| / |classes in either| (1.0 = same garment types);
* ``max_delta_e``: largest colour difference (CIE76 ΔE, ~2 just noticeable, >10 clearly different) over the
  shared classes; mean colours of patterned garments stay similar across views.

:func:`build_pair_table` produces these numbers for all same-item pairs of a split;
:class:`PairFilter` keeps only pairs within thresholds (``data.pair_filter``). It is off by default, so
baseline runs are unchanged. Choose thresholds from the measured distribution
(``scripts/analyze_pairs.py``), not by guessing.
"""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.data.labels import convert_labels, get_label_set
from src.data.preprocessing import load_image, load_label_map


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 [..., 3] -> CIELAB (D65) float [..., 3]."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def garment_signature(image: np.ndarray, garment_map: np.ndarray, class_ids: Iterable[int],
                      min_fraction: float = 0.02) -> Dict[int, Tuple[float, np.ndarray]]:
    """{class id: (share of person pixels, mean Lab colour)} for classes covering >= ``min_fraction`` of the person."""
    person = garment_map > 0
    total = max(int(person.sum()), 1)
    lab = None
    sig = {}
    for k in class_ids:
        mask = garment_map == k
        frac = mask.sum() / total
        if frac < min_fraction:
            continue
        if lab is None:
            lab = rgb_to_lab(image)
        sig[int(k)] = (float(frac), lab[mask].mean(0))
    return sig


def compare_signatures(a: Dict, b: Dict) -> Dict[str, float]:
    ka, kb = set(a), set(b)
    union, shared = ka | kb, ka & kb
    jacc = len(shared) / len(union) if union else 1.0
    de = [float(np.linalg.norm(a[k][1] - b[k][1])) for k in shared]
    return {"class_jaccard": jacc, "same_classes": float(ka == kb), "max_delta_e": max(de) if de else float("nan"),
            "mean_delta_e": float(np.mean(de)) if de else float("nan"), "n_shared": len(shared)}


def _signature_for(rec, root: Path, classes, size: int = 256, min_fraction: float = 0.02):
    img = load_image(root / rec.image)
    lab = convert_labels(load_label_map(root / rec.parsing), "dfmm", "garment12")
    if img.shape[0] > size:  # downscale for speed; nearest for labels
        w = round(img.shape[1] * size / img.shape[0])
        img = np.asarray(Image.fromarray(img).resize((w, size), Image.BOX))
        lab = np.asarray(Image.fromarray(lab).resize((w, size), Image.NEAREST))
    return garment_signature(img, lab, classes, min_fraction)


def build_pair_table(records, root: Path, transfer_classes, progress: bool = True) -> List[Dict]:
    """Metrics for every unordered pair of annotated photos sharing an item ID."""
    G = get_label_set("garment12")
    classes = G.indices(transfer_classes)
    by_item = defaultdict(list)
    for r in records:
        if r.item_id and r.parsing:
            by_item[r.item_id].append(r)
    multi = {k: v for k, v in by_item.items() if len(v) > 1}
    todo = [r for v in multi.values() for r in v]
    it = todo
    if progress:
        from tqdm.auto import tqdm

        it = tqdm(todo, desc="garment signatures")
    sigs = {r.name: _signature_for(r, root, classes) for r in it}
    rows = []
    for item, recs in sorted(multi.items()):
        for a, b in itertools.combinations(sorted(recs, key=lambda r: r.name), 2):
            m = compare_signatures(sigs[a.name], sigs[b.name])
            rows.append({"item_id": item, "a": a.name, "b": b.name, **m,
                         "classes_a": "+".join(G.names[k] for k in sorted(sigs[a.name])),
                         "classes_b": "+".join(G.names[k] for k in sorted(sigs[b.name]))})
    return rows


def write_pair_table(rows: List[Dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def summarise(rows: List[Dict], thresholds=(5, 10, 15, 20, 30)) -> Dict:
    de = np.array([r["max_delta_e"] for r in rows if not np.isnan(r["max_delta_e"])])
    same = np.array([r["same_classes"] for r in rows])
    out = {"pairs": len(rows), "items": len({r["item_id"] for r in rows}), "share_same_classes": float(same.mean()) if len(rows) else None,
           "max_delta_e_quantiles": {q: float(np.quantile(de, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(de) else {},
           "share_kept": {}}
    for t in thresholds:
        keep = [r for r in rows if r["same_classes"] == 1.0 and r["max_delta_e"] <= t]
        out["share_kept"][f"same_classes_and_dE<={t}"] = len(keep) / len(rows) if rows else None
    return out


class PairFilter:
    """Allowed same-item pairs according to a pair table (``data.pair_filter``).

    ``mode``: ``none`` | ``classes`` (identical garment classes) | ``classes_color`` (also max ΔE <= threshold).
    """

    def __init__(self, mode: str = "none", table_path: Optional[Path] = None, max_delta_e: float = 15.0):
        self.mode = mode
        self.allowed = None
        if mode == "none":
            return
        if table_path is None or not Path(table_path).is_file():
            raise FileNotFoundError(
                f"data.pair_filter={mode} needs the pair table at {table_path}. Run scripts/analyze_pairs.py "
                "(or notebook Section 6b) first."
            )
        self.allowed = set()
        with open(table_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                ok = float(r["same_classes"]) == 1.0
                if mode == "classes_color":
                    de = float(r["max_delta_e"])
                    ok = ok and not np.isnan(de) and de <= max_delta_e
                if ok:
                    self.allowed.add(frozenset((r["a"], r["b"])))

    def ok(self, a: str, b: str) -> bool:
        return self.allowed is None or frozenset((a, b)) in self.allowed
