"""Are two photos of the same DeepFashion item really wearing the same outfit?

In DeepFashion an item ID names one product. Other garments in the photo can change between shots
(observed in the texture run of 2026-10-21: a beanie item shown once with a plaid shirt + jeans and once
with a green sweater + black jeans). Cross-view training pairs with different outfits ask the warp and texture
networks to produce garments they cannot see.

For every image we compute a *garment signature* from the manual parsing: for each transfer class, its share
of the person area and its mean colour in CIELAB. Two photos are compared by:

* ``class_jaccard`` / ``same_classes``: do both photos contain the same garment types?
* ``max_delta_e``: largest colour difference (CIE76 ΔE; ~2 just noticeable, >10 clearly different) over the
  shared garments. Sensitive to small regions such as shoes.
* ``weighted_delta_e``: the same differences weighted by each garment's (smaller) area share; more robust.
* ``same_seq``: whether both photos share the photo-set number in the file name (``...-01_1_front`` vs
  ``...-02_1_front``). Different sets may be different shoots or colourways.

:func:`build_pair_table` produces these numbers for all same-item pairs; :func:`breakdown` summarises
them per group; :class:`PairFilter` keeps only allowed pairs (``data.pair_filter``). The filter is off by
default, so baseline runs are unchanged. Thresholds must come from the measured distribution
(``scripts/analyze_pairs.py``), not from guessing.
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

TEXT_COLUMNS = {"item_id", "a", "b", "seq_a", "seq_b", "view_a", "view_b", "classes_a", "classes_b"}


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
    union, shared = ka | kb, sorted(ka & kb)
    jacc = len(shared) / len(union) if union else 1.0
    de = {k: float(np.linalg.norm(a[k][1] - b[k][1])) for k in shared}
    w = {k: min(a[k][0], b[k][0]) for k in shared}
    wsum = sum(w.values())
    return {"class_jaccard": jacc, "same_classes": float(ka == kb),
            "max_delta_e": max(de.values()) if de else float("nan"),
            "mean_delta_e": float(np.mean(list(de.values()))) if de else float("nan"),
            "weighted_delta_e": sum(w[k] * de[k] for k in shared) / wsum if wsum else float("nan"),
            "n_shared": len(shared)}


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
    return pair_rows(multi, sigs)


def pair_rows(groups: Dict[str, list], sigs: Dict) -> List[Dict]:
    """Table rows for every unordered pair within each item group. ``groups``: item -> records with
    ``name`` / ``seq`` / ``view``; ``sigs``: name -> :func:`garment_signature`."""
    G = get_label_set("garment12")
    rows = []
    for item, recs in sorted(groups.items()):
        for a, b in itertools.combinations(sorted(recs, key=lambda r: r.name), 2):
            sa, sb = sigs[a.name], sigs[b.name]
            row = {"item_id": item, "a": a.name, "b": b.name, "seq_a": a.seq or "", "seq_b": b.seq or "",
                   "same_seq": float(a.seq == b.seq), "view_a": a.view or "", "view_b": b.view or "",
                   **compare_signatures(sa, sb),
                   "classes_a": "+".join(G.names[k] for k in sorted(sa)),
                   "classes_b": "+".join(G.names[k] for k in sorted(sb))}
            for k in sorted(set(sa) & set(sb)):
                row[f"de.{G.names[k]}"] = float(np.linalg.norm(sa[k][1] - sb[k][1]))
            rows.append(row)
    return rows


def write_pair_table(rows: List[Dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    return path


def _seq(name: str) -> str:
    from src.data.deepfashion_mm import parse_name

    m = parse_name(name)
    return m["seq"] if m else ""


def read_pair_table(path: Path) -> List[Dict]:
    """Load a pair table (any version). Numbers become floats; ``same_seq`` is derived from names if absent."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k, v in r.items():
                if k in TEXT_COLUMNS:
                    out[k] = v
                else:
                    try:
                        out[k] = float(v) if v not in ("", None) else float("nan")
                    except ValueError:
                        out[k] = v
            if "same_seq" not in out or (isinstance(out["same_seq"], float) and np.isnan(out["same_seq"])):
                out["seq_a"], out["seq_b"] = _seq(out["a"]), _seq(out["b"])
                out["same_seq"] = float(out["seq_a"] == out["seq_b"])
            rows.append(out)
    return rows


def _quantiles(vals, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    vals = np.array([v for v in vals if isinstance(v, float) and not np.isnan(v)])
    return {str(q): round(float(np.quantile(vals, q)), 2) for q in qs} if len(vals) else {}


def summarise(rows: List[Dict], thresholds=(5, 10, 15, 20, 30)) -> Dict:
    same = np.array([r["same_classes"] for r in rows])
    out = {"pairs": len(rows), "items": len({r["item_id"] for r in rows}),
           "share_same_classes": float(same.mean()) if len(rows) else None,
           "max_delta_e_quantiles": _quantiles([r["max_delta_e"] for r in rows]),
           "share_kept": {}}
    if rows and "weighted_delta_e" in rows[0]:
        out["weighted_delta_e_quantiles"] = _quantiles([r["weighted_delta_e"] for r in rows])
    for t in thresholds:
        keep = [r for r in rows if r["same_classes"] == 1.0 and r["max_delta_e"] <= t]
        out["share_kept"][f"same_classes_and_dE<={t}"] = len(keep) / len(rows) if rows else None
    return out


def breakdown(rows: List[Dict], key: str = "same_seq") -> Dict:
    """Consistency statistics per group (default: same photo set "1.0" vs different photo sets "0.0")."""
    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get(key))].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        entry = {"pairs": len(rs), "share_same_classes": round(float(np.mean([r["same_classes"] for r in rs])), 4),
                 "max_delta_e": _quantiles([r["max_delta_e"] for r in rs], (0.25, 0.5, 0.75))}
        if "weighted_delta_e" in rs[0]:
            entry["weighted_delta_e"] = _quantiles([r["weighted_delta_e"] for r in rs], (0.25, 0.5, 0.75))
        out[g] = entry
    return out


class PairFilter:
    """Allowed same-item pairs (``data.pair_filter``).

    ``mode``: ``none`` | ``classes`` (identical garment classes) | ``classes_color`` (also ΔE <= threshold) |
    ``same_seq`` (both photos from the same photo set; needs no table).
    ``color_metric``: ``max`` or ``weighted`` (area-weighted ΔE, see :func:`compare_signatures`).
    """

    def __init__(self, mode: str = "none", table_path: Optional[Path] = None, max_delta_e: float = 15.0,
                 color_metric: str = "max"):
        self.mode = mode
        self.allowed = None
        if mode == "none":
            return
        if mode == "same_seq":
            self.allowed = "same_seq"
            return
        if table_path is None or not Path(table_path).is_file():
            raise FileNotFoundError(
                f"data.pair_filter={mode} needs the pair table at {table_path}. Run scripts/analyze_pairs.py "
                "(or notebook Section 6b) first."
            )
        col = "weighted_delta_e" if color_metric == "weighted" else "max_delta_e"
        self.allowed = set()
        for r in read_pair_table(table_path):
            ok = r["same_classes"] == 1.0
            if mode == "classes_color":
                if col not in r:
                    raise KeyError(f"pair table {table_path} has no '{col}' column; re-run scripts/analyze_pairs.py")
                ok = ok and not np.isnan(r[col]) and r[col] <= max_delta_e
            if ok:
                self.allowed.add(frozenset((r["a"], r["b"])))

    def ok(self, a: str, b: str) -> bool:
        if self.allowed is None:
            return True
        if self.allowed == "same_seq":
            return _seq(a) == _seq(b)
        return frozenset((a, b)) in self.allowed
