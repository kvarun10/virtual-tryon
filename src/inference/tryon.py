"""Phase 6: the complete Dress-up try-on with trained warp and texture networks.

Data flow at model resolution (``data.image_size``, letterboxed), paper Fig. 3:

    reference photo I_C --Stage A--> full parse (garment12) --keep transfer classes--> C_seg
    target photo  I_B   --Stage A--> body map B_seg (labels.body_label_set) + parse (for identity)
    Stage B  warp(C_seg, B_seg)                        -> C'_seg   (untrained classes masked)
    Sec. 4.2 C'_seg + target face/hair labels          -> layout
    Stage C  texture(layout, I_C, parse(I_C))          -> I_G (raw)
    identity paste: target face/hair pixels blended in -> result      [E] (see below)

Identity paste [E]: the texture network is trained on pairs that show the same model, so it reproduces
the *exemplar's* face. Paper Sec. 4.2 only swaps the face/hair *labels*; to keep the target's identity in
the pixels we also copy the target's face/hair pixels into the result, with a soft (Gaussian-feathered)
mask edge. Documented as deviation D23.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.config import ProjectConfig
from src.data.labels import get_label_set, inject_identity_labels, keep_classes
from src.data.preprocessing import image_to_tensor, resize_image, resize_labels, tensor_to_image
from src.models.spade_generator import TextureGenerator, build_texture_generator
from src.models.warp_network import DualPathUNet, build_warp_generator
from src.training.checkpointing import CheckpointManager, CheckpointMismatchError
from src.utils.device import select_device
from src.utils.paths import ProjectPaths


class ModelsUnavailableError(FileNotFoundError):
    pass


@dataclass
class TryOnModels:
    cfg: ProjectConfig
    warp: DualPathUNet
    texture: TextureGenerator
    device: torch.device
    untrained_ids: list
    info: Dict = field(default_factory=dict)


def _checkpoint_path(cfg: ProjectConfig, stage: str, which: Union[str, Path]) -> Path:
    if which in ("last", "best"):
        run = getattr(cfg.training, f"{stage}_run_name")
        return ProjectPaths.from_config(cfg).checkpoints / cfg.experiment.name / run / f"{which}.pt"
    return Path(which)


def _load_generator(model: torch.nn.Module, path: Path, stage: str) -> Dict:
    if not path.is_file():
        raise ModelsUnavailableError(
            f"No trained {stage} checkpoint at {path}. Train the {stage} stage first (notebook Section 10/10b) or "
            f"pass the path of an existing checkpoint."
        )
    st = CheckpointManager(path.parent).load(path, map_location="cpu")
    saved = st.get("signature", {}).get("generator")
    if saved is not None and saved != model.cfg:
        diff = {k: (saved.get(k), v) for k, v in model.cfg.items() if saved.get(k) != v}
        raise CheckpointMismatchError(f"{stage} checkpoint {path} was trained with a different architecture "
                                      f"(checkpoint vs config): {diff}")
    model.load_state_dict(st.get("gen_ema", st["gen"]))  # EMA weights when the run kept them
    return {"path": str(path), "step": st.get("step"), "weights": "ema" if "gen_ema" in st else "live"}


def load_tryon_models(cfg: ProjectConfig, warp: Union[str, Path] = "last", texture: Union[str, Path] = "last",
                      device: Optional[torch.device] = None) -> TryOnModels:
    """Build both generators from the config and load their trained weights (generator only, eval mode)."""
    device = device or select_device(cfg.experiment.device)
    G = get_label_set(cfg.labels.garment_label_set)
    Kb = get_label_set(cfg.labels.body_label_set).num_classes
    wg = build_warp_generator(cfg, G.num_classes, Kb, G.num_classes)
    tg = build_texture_generator(cfg, G.num_classes)
    info = {"warp": _load_generator(wg, _checkpoint_path(cfg, "warp", warp), "warp"),
            "texture": _load_generator(tg, _checkpoint_path(cfg, "texture", texture), "texture")}
    supervised = set(cfg.labels.transfer_classes) | {"background", "skin"}
    untrained = [i for i, n in enumerate(G.names) if n not in supervised]
    return TryOnModels(cfg, wg.to(device).eval(), tg.to(device).eval(), device, untrained, info)


def _one_hot(lab: np.ndarray, k: int, device) -> torch.Tensor:
    t = torch.from_numpy(lab.astype(np.int64)).to(device)
    return F.one_hot(t, k).permute(2, 0, 1).float()[None]


def _clock(device) -> float:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def feather(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    from scipy import ndimage

    return np.clip(ndimage.gaussian_filter(mask.astype(np.float32), sigma), 0, 1) if sigma > 0 else mask.astype(np.float32)


@torch.no_grad()
def tryon_core(models: TryOnModels, ref_image: np.ndarray, ref_parse: np.ndarray, target_image: np.ndarray,
               target_body: np.ndarray, target_parse: np.ndarray, feather_sigma: float = 1.0,
               timings: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    """All inputs already at model resolution: RGB uint8 [H,W,3]; parses garment12 [H,W]; body in the body label set.

    ``timings``: if a dict is given, seconds for ``warp`` and ``texture`` (incl. identity paste) are written into it
    (CUDA synchronised)."""
    cfg, dev = models.cfg, models.device
    G = get_label_set(cfg.labels.garment_label_set)
    Kb = get_label_set(cfg.labels.body_label_set).num_classes
    size = tuple(cfg.data.image_size)
    for name, arr in (("ref_image", ref_image), ("ref_parse", ref_parse), ("target_image", target_image),
                      ("target_body", target_body), ("target_parse", target_parse)):
        if arr.shape[:2] != size:
            raise ValueError(f"{name} has size {arr.shape[:2]}, expected model size {size}")
    t0 = _clock(dev)
    c_seg = keep_classes(ref_parse, G, cfg.labels.transfer_classes)
    logits = models.warp(_one_hot(c_seg, G.num_classes, dev), _one_hot(target_body, Kb, dev)).float()
    logits[:, models.untrained_ids] = -torch.inf
    c_warp = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    layout = inject_identity_labels(c_warp, target_parse, G, cfg.labels.preserve_classes)  # paper Sec. 4.2
    t1 = _clock(dev)
    out = models.texture(_one_hot(layout, G.num_classes, dev), image_to_tensor(ref_image)[None].to(dev),
                         _one_hot(ref_parse, G.num_classes, dev))
    raw = tensor_to_image(out["fake"][0].float().cpu())
    warped = F.interpolate(out["warped"].float(), size=size, mode="nearest")[0].cpu()
    ident = np.isin(layout, G.indices(cfg.labels.preserve_classes))
    m = feather(ident, feather_sigma)[..., None]
    result = (m * target_image.astype(np.float32) + (1 - m) * raw.astype(np.float32)).round().astype(np.uint8)
    if timings is not None:
        timings["warp"], timings["texture"] = t1 - t0, _clock(dev) - t1
    return {"c_seg": c_seg, "c_warp": c_warp, "layout": layout, "warped_exemplar": tensor_to_image(warped),
            "raw": raw, "result": result, "identity_mask": ident}


def to_model_size(cfg: ProjectConfig, image: np.ndarray, *label_maps: np.ndarray):
    """Letterbox an image (white pad) and label maps (background pad) to ``data.image_size``."""
    size, mode = tuple(cfg.data.image_size), cfg.data.resize_mode
    img, info = resize_image(image, size, mode, cfg.data.pad_value_image)
    labs = [resize_labels(l, size, mode)[0] for l in label_maps]
    return (img, *labs, info)
