"""Figures: label legends, overlays and comparison grids (matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.data.labels import LabelSet, colorize


def overlay(image: np.ndarray, label_map: np.ndarray, label_set: LabelSet, alpha: float = 0.5) -> np.ndarray:
    """Blend a colour-coded label map over an RGB image (background pixels stay unchanged)."""
    if image.shape[:2] != label_map.shape:
        raise ValueError(f"image {image.shape[:2]} and labels {label_map.shape} differ in size")
    color = colorize(label_map, label_set).astype(np.float32)
    out = image.astype(np.float32)
    fg = label_map > 0
    out[fg] = (1 - alpha) * out[fg] + alpha * color[fg]
    return out.round().astype(np.uint8)


def legend_handles(label_set: LabelSet, present: Optional[Sequence[int]] = None):
    from matplotlib.patches import Patch

    ids = range(label_set.num_classes) if present is None else sorted(set(int(i) for i in present))
    return [Patch(facecolor=np.array(label_set.palette[i]) / 255.0, edgecolor="k", linewidth=0.3,
                  label=f"{i}: {label_set.names[i]}") for i in ids]


def show_panels(
    panels: List[Tuple[str, np.ndarray]],
    legends: Sequence[Tuple[LabelSet, np.ndarray]] = (),
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    panel_size: float = 3.0,
):
    """Plot images side by side; add one legend per (label_set, label_map) listing present classes."""
    import matplotlib.pyplot as plt

    n = len(panels)
    n_leg = len(legends)
    fig, axes = plt.subplots(1, n + n_leg, figsize=(panel_size * (n + 0.9 * n_leg), panel_size * 1.4))
    axes = np.atleast_1d(axes)
    for ax, (name, arr) in zip(axes, panels):
        ax.imshow(arr)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    for ax, (ls, lab) in zip(axes[n:], legends):
        ax.axis("off")
        ax.legend(handles=legend_handles(ls, np.unique(lab)), loc="center", fontsize=7, title=ls.name, title_fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def label_set_legend_figure(label_set: LabelSet, save_path: Optional[Path] = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 0.28 * label_set.num_classes + 0.5))
    ax.axis("off")
    ax.legend(handles=legend_handles(label_set), loc="center", fontsize=8, title=label_set.name, frameon=False)
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
