"""End-to-end try-on interface.

Current status (Phase 3): only Stage A (pre-treatment) is implemented. ``try_on`` runs Stage A
for both images and then raises :class:`StageNotAvailableError` for the warp stage, unless
``stop_after="pretreatment"`` is requested. It never fabricates a "result" image.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.config import ProjectConfig, load_config
from src.data.labels import get_label_set
from src.data.preprocessing import load_image
from src.data.reference import PreparedReference, build_reference_preprocessor
from src.models.segmentation import SegmentationBackend, build_backend, to_project_labels


class StageNotAvailableError(RuntimeError):
    """A pipeline stage has no implementation / trained weights yet."""


@dataclass
class PreparedTarget:
    image: np.ndarray  # H x W x 3 RGB
    body_map: np.ndarray  # H x W, labels.body_label_set (B_seg)
    parsing: Optional[np.ndarray]  # H x W garment12 parsing of the target (for face/hair identity)
    meta: Dict = field(default_factory=dict)


@dataclass
class TryOnResult:
    reference: PreparedReference
    target: PreparedTarget
    warped_garment: Optional[np.ndarray] = None  # C'_seg (Phase 4)
    correspondence: Optional[np.ndarray] = None  # I_S (Phase 5)
    output: Optional[np.ndarray] = None  # I_G (Phase 5)
    completed_stages: tuple = ()
    notes: list = field(default_factory=list)
    extra: Dict = field(default_factory=dict)  # all intermediate maps of a full try-on (model resolution)


class PreTreatment:
    """Stage A for both inputs, using the backends named in the config."""

    def __init__(self, cfg: ProjectConfig, clothing: Optional[SegmentationBackend] = None, body: Optional[SegmentationBackend] = None):
        self.cfg = cfg
        self.clothing = clothing or build_backend(cfg.segmentation.clothing_backend, cfg, role="clothing")
        self.body = body or build_backend(cfg.segmentation.body_backend, cfg, role="body")
        self.reference_pre = build_reference_preprocessor(cfg, self.clothing)

    def prepare_reference(self, image, key: Optional[str] = None, mode: Optional[str] = None,
                          garment_category: Optional[str] = None) -> PreparedReference:
        pre = self.reference_pre if mode in (None, self.reference_pre.mode) else build_reference_preprocessor(self.cfg, self.clothing, mode)
        return pre.prepare(image, key=key, garment_category=garment_category)

    def prepare_target(self, image, key: Optional[str] = None) -> PreparedTarget:
        image = load_image(image)
        body = self.body.predict(image, key)
        body_map = to_project_labels(body, self.cfg.labels.body_label_set)
        parsing = to_project_labels(self.clothing.predict(image, key), self.cfg.labels.garment_label_set)
        return PreparedTarget(image, body_map, parsing, {"body_backend": body.backend, "body_native": body.label_set})


def try_on(reference_image, target_image, config: Optional[ProjectConfig] = None, *, reference_key: Optional[str] = None,
           target_key: Optional[str] = None, reference_mode: Optional[str] = None, garment_category: Optional[str] = None,
           stop_after: str = "texture", pretreatment: Optional[PreTreatment] = None, models=None,
           save_dir: Optional[Path] = None) -> TryOnResult:
    """Run the Dress-up pipeline. ``stop_after``: pretreatment | warp | texture.

    ``models``: a :class:`src.inference.tryon.TryOnModels` (``load_tryon_models(cfg)``) with trained warp and
    texture weights. Without it only Stage A can run. All model outputs are at ``data.image_size``.
    """
    if stop_after not in {"pretreatment", "warp", "texture"}:
        raise ValueError("stop_after must be pretreatment | warp | texture")
    cfg = config or load_config()
    pre = pretreatment or PreTreatment(cfg)
    ref = pre.prepare_reference(reference_image, key=reference_key, mode=reference_mode, garment_category=garment_category)
    tgt = pre.prepare_target(target_image, key=target_key)
    result = TryOnResult(ref, tgt, completed_stages=("pretreatment",), notes=list(ref.warnings))
    if save_dir is not None:
        save_pretreatment(result, Path(save_dir), cfg)
    if stop_after == "pretreatment":
        return result
    if not ref.validated:
        raise StageNotAvailableError(
            f"Reference mode '{ref.mode}' is experimental and has no compatible warp model. "
            "Use stop_after='pretreatment' to inspect its C_seg."
        )
    if models is None:
        raise StageNotAvailableError(
            "No trained models were given. Load them with src.inference.tryon.load_tryon_models(cfg) and pass "
            "models=..., or call try_on(..., stop_after='pretreatment') for C_seg and B_seg only."
        )
    from src.inference.tryon import to_model_size, tryon_core

    ref_img, ref_parse, _ = to_model_size(cfg, ref.image, ref.full_parsing)
    tgt_img, tgt_body, tgt_parse, info = to_model_size(cfg, tgt.image, tgt.body_map, tgt.parsing)
    core = tryon_core(models, ref_img, ref_parse, tgt_img, tgt_body, tgt_parse)
    result.warped_garment = core["c_warp"]
    result.correspondence = core["warped_exemplar"]
    result.output = core["result"]
    result.completed_stages = ("pretreatment", "warp", "texture")
    result.notes.append(f"outputs at model resolution {tuple(cfg.data.image_size)}; target face/hair pixels pasted (D23)")
    result.extra = {**core, "reference_model_size": ref_img, "target_model_size": tgt_img, "target_body": tgt_body,
                    "resize_info": info}
    if save_dir is not None:
        save_tryon(result, Path(save_dir), cfg)
    return result


def save_tryon(result: TryOnResult, out_dir: Path, cfg: ProjectConfig) -> Path:
    """Save every stage of a full try-on as PNGs plus a one-row overview."""
    from PIL import Image

    from src.data.labels import colorize

    out_dir.mkdir(parents=True, exist_ok=True)
    G = get_label_set(cfg.labels.garment_label_set)
    B = get_label_set(cfg.labels.body_label_set)
    e = result.extra
    tiles = [e["reference_model_size"], colorize(e["c_seg"], G), e["target_model_size"], colorize(e["target_body"], B),
             colorize(e["c_warp"], G), colorize(e["layout"], G), e["warped_exemplar"], e["raw"], e["result"]]
    names = ["reference", "C_seg", "target", "B_seg", "C_seg_warped", "layout_with_identity", "warped_exemplar_I_S",
             "generated_raw", "result"]
    for n, t in zip(names, tiles):
        Image.fromarray(t).save(out_dir / f"{n}.png")
    Image.fromarray(np.concatenate(tiles, 1)).save(out_dir / "overview.png")
    return out_dir


def save_pretreatment(result: TryOnResult, out_dir: Path, cfg: ProjectConfig) -> Path:
    from PIL import Image

    from src.data.labels import colorize

    out_dir.mkdir(parents=True, exist_ok=True)
    g = get_label_set(cfg.labels.garment_label_set)
    b = get_label_set(cfg.labels.body_label_set)
    Image.fromarray(result.reference.image).save(out_dir / "reference.png")
    Image.fromarray(result.target.image).save(out_dir / "target.png")
    for name, arr, ls in (("C_seg", result.reference.garment_map, g), ("B_seg", result.target.body_map, b)):
        Image.fromarray(arr).save(out_dir / f"{name}_labels.png")  # raw class ids
        Image.fromarray(colorize(arr, ls)).save(out_dir / f"{name}_color.png")
    meta = {"reference_mode": result.reference.mode, "validated": result.reference.validated,
            "warnings": result.notes, "reference_meta": result.reference.meta, "target_meta": result.target.meta,
            "garment_label_set": g.name, "body_label_set": b.name}
    (out_dir / "pretreatment.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return out_dir
