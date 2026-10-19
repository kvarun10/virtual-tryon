"""Reproducible, group-disjoint train / val / test splits.

Images are assigned to splits by *group* (the garment item ID for DeepFashion), never
individually. Different photos of the same garment therefore always land in the same split,
which prevents the evaluation from rewarding memorised garments (leakage).

Algorithm: sort group keys, shuffle them with ``numpy.random.default_rng(seed)``, then assign
whole groups greedily so that each split's *image* count approaches its target ratio. The result
depends only on (set of groups, their sizes, ratios, seed), not on file-system order.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

SPLITS = ("train", "val", "test")


class SplitError(ValueError):
    pass


def group_disjoint_split(
    items: Sequence[Tuple[str, str]],
    ratios: Mapping[str, float],
    seed: int = 0,
) -> Dict[str, List[str]]:
    """Split ``(name, group)`` pairs. Returns ``{split: sorted names}``."""
    if set(ratios) != set(SPLITS):
        raise SplitError(f"ratios must have keys {SPLITS}, got {sorted(ratios)}")
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise SplitError(f"ratios must sum to 1, got {total_ratio}")
    groups: Dict[str, List[str]] = defaultdict(list)
    for name, group in items:
        groups[group].append(name)
    if not groups:
        raise SplitError("Nothing to split: the record list is empty")
    keys = sorted(groups)
    order = np.random.default_rng(seed).permutation(len(keys))
    n_total = sum(len(v) for v in groups.values())
    targets = {s: ratios[s] * n_total for s in SPLITS}
    counts = {s: 0 for s in SPLITS}
    out: Dict[str, List[str]] = {s: [] for s in SPLITS}
    for i in order:
        key = keys[i]
        # Assign to the split furthest below its target (relative deficit); ties -> SPLITS order.
        split = max(
            (s for s in SPLITS if ratios[s] > 0),
            key=lambda s: (targets[s] - counts[s]) / max(targets[s], 1e-9),
        )
        out[split].extend(groups[key])
        counts[split] += len(groups[key])
    return {s: sorted(v) for s, v in out.items()}


def check_disjoint(splits: Mapping[str, Iterable[str]], group_of: Mapping[str, str]) -> None:
    """Raise if any image or group appears in more than one split."""
    seen_name: Dict[str, str] = {}
    seen_group: Dict[str, str] = {}
    for split, names in splits.items():
        for n in names:
            if n in seen_name:
                raise SplitError(f"Image '{n}' is in both '{seen_name[n]}' and '{split}'")
            seen_name[n] = split
            g = group_of[n]
            if g in seen_group and seen_group[g] != split:
                raise SplitError(f"Group '{g}' leaks across '{seen_group[g]}' and '{split}'")
            seen_group[g] = split


def fingerprint(names: Iterable[str]) -> str:
    h = hashlib.sha1()
    for n in sorted(names):
        h.update(n.encode("utf-8") + b"\n")
    return h.hexdigest()[:16]


def save_splits(path: Path, splits: Mapping[str, List[str]], meta: Mapping) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_names = [n for v in splits.values() for n in v]
    payload = {
        "meta": {**meta, "fingerprint": fingerprint(all_names), "counts": {k: len(v) for k, v in splits.items()}},
        "splits": {k: list(v) for k, v in splits.items()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return path


def load_splits(path: Path, expected_names: Iterable[str] = None) -> Tuple[Dict[str, List[str]], Dict]:
    """Load splits; if ``expected_names`` is given, verify the split covers the same images."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}. Run scripts/prepare_dataset.py first.")
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    splits, meta = payload["splits"], payload["meta"]
    if expected_names is not None:
        fp = fingerprint(expected_names)
        if fp != meta.get("fingerprint"):
            raise SplitError(
                f"Split file {path} was made for a different image set (fingerprint {meta.get('fingerprint')} "
                f"vs current {fp}). Delete it to regenerate, or point data.split_file elsewhere."
            )
    return splits, meta
