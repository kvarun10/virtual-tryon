"""Central configuration: typed dataclasses loaded from YAML.

Design
------
* The dataclass defaults below are the single source of default values.
* A YAML file only needs to list the values it changes. A file may inherit from another with
  ``_base_: other.yaml`` (path relative to the file).
* Command-line / notebook overrides use dotted keys: ``["data.image_size=[128,128]"]``.
* Unknown keys raise an error, so a typo such as ``bach_size`` never passes silently.
* Path strings may contain ``{placeholders}`` that refer to ``paths.*`` fields and to
  ``data.dataset`` / ``data.split_seed``; :func:`load_config` resolves them.

Fields under ``training`` and ``model`` are reserved for Phases 4-5; they are defined now so
that experiment files have a stable schema, but no code reads them yet.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

PathLike = Union[str, Path]


class ConfigError(ValueError):
    """Raised for invalid or unknown configuration values."""


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 42
    run_mode: str = "smoke"  # smoke | baseline | extended
    device: str = "auto"  # auto | cpu | cuda
    deterministic: bool = False  # torch deterministic algorithms (slower)
    dry_run: bool = False  # CPU-only checks, no downloads, tiny data


@dataclass
class PathsConfig:
    # On Colab set drive_root to e.g. /content/drive/MyDrive/VirtualTryOn
    drive_root: str = "./outputs/local_drive"
    datasets_dir: str = "{drive_root}/datasets"
    pretrained_dir: str = "{drive_root}/pretrained"  # third-party weights (SCHP, ...)
    checkpoints_dir: str = "{drive_root}/checkpoints"  # weights trained by this project
    cache_dir: str = "{drive_root}/cache"  # parser outputs, indexes, splits
    outputs_dir: str = "{drive_root}/outputs"  # figures, generated images, reports
    logs_dir: str = "{drive_root}/logs"


@dataclass
class DataLayoutConfig:
    """Relative file layout inside ``data.root``.

    Defaults follow the DeepFashion-MultiModal README and the Text2Human / DiOr loaders. The
    unzipped folder names were NOT verified against a real download; run
    ``scripts/prepare_dataset.py --inspect`` once and adjust these values if needed.
    """

    images_dir: str = "images"  # DiOr's download script unzips the image archive to "images"
    parsing_dir: str = "segm"  # unverified folder name
    densepose_dir: str = "densepose"
    keypoints_loc: str = "keypoints/keypoints_loc.txt"
    keypoints_vis: str = "keypoints/keypoints_vis.txt"
    image_ext: str = ".jpg"
    parsing_suffix: str = "_segm.png"
    densepose_suffix: str = "_densepose.png"
    densepose_part_channel: int = 2  # Text2Human reads np.array(img)[:, :, 2]


@dataclass
class DataConfig:
    dataset: str = "deepfashion_multimodal"  # deepfashion_multimodal | synthetic
    # Extracted, working copy that the code reads. On Colab: local disk (fast, temporary).
    root: str = "{datasets_dir}/DeepFashion-MultiModal"
    # Persistent zip archives (on Colab: Google Drive). See src/data/fetch.py.
    archive_dir: str = "{datasets_dir}/DeepFashion-MultiModal/archives"
    layout: DataLayoutConfig = field(default_factory=DataLayoutConfig)
    image_size: List[int] = field(default_factory=lambda: [256, 256])  # [H, W]
    resize_mode: str = "letterbox"  # letterbox (keep aspect, pad) | stretch
    pad_value_image: int = 255  # DeepFashion backgrounds are white
    require_parsing: bool = True  # keep only images with manual parsing
    require_densepose: bool = True  # false: allowed without the DensePose archive (no body15 ground truth)
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    split_seed: int = 0
    split_file: str = "{cache_dir}/splits/{dataset}_seed{split_seed}.json"
    index_file: str = "{cache_dir}/index/{dataset}_index.json"
    max_samples: Optional[int] = None  # cap per split (smoke tests)
    image_check: str = "header"  # none | header | full  (full decodes every image)
    num_workers: int = 2


@dataclass
class SegmentationConfig:
    clothing_backend: str = "schp_lip"  # see src/models/segmentation.py::BACKENDS
    body_backend: str = "annotation_densepose"
    device: str = "auto"
    cache_outputs: bool = True
    allow_weight_download: bool = False  # must be switched on explicitly


@dataclass
class LabelConfig:
    garment_label_set: str = "garment12"
    body_label_set: str = "body15"
    # Classes copied from the reference into C_seg (paper figures transfer the full outfit).
    transfer_classes: List[str] = field(
        default_factory=lambda: ["top", "outer", "dress", "skirt", "pants", "footwear"]
    )
    # Classes taken from the target person to keep identity (paper Sec. 4.2: face and hair).
    preserve_classes: List[str] = field(default_factory=lambda: ["hair", "face"])


@dataclass
class ReferenceConfig:
    mode: str = "person_worn"  # person_worn | garment_only
    garment_category: Optional[str] = None  # required for garment_only, e.g. "top"
    background_tolerance: float = 30.0  # RGB distance to the border colour (garment_only)
    min_garment_fraction: float = 0.02  # reject masks covering less of the image


@dataclass
class TrainingConfig:  # reserved for Phases 4-5
    batch_size: int = 4
    epochs: int = 1
    max_steps: Optional[int] = None
    grad_accum_steps: int = 1
    amp: bool = True
    lr_generator: float = 1e-4  # paper Sec. 5.1
    lr_discriminator: float = 4e-4  # CoCosNet TTUR value; Dress-up only says "imbalanced"
    betas: List[float] = field(default_factory=lambda: [0.0, 0.999])  # paper Sec. 5.1
    warp_lr: float = 2e-4  # not given in the paper (engineering choice)
    warp_betas: List[float] = field(default_factory=lambda: [0.5, 0.999])
    lambda_adv_warp: float = 0.1  # not given in the paper
    checkpoint_every: int = 500
    log_every: int = 50
    val_every: int = 1000
    early_stop_patience: int = 0  # 0 disables


@dataclass
class ModelConfig:  # reserved for Phases 4-5
    warp_base_channels: int = 64
    warp_cloth_bottleneck: int = 2  # paper: 2 x 2 x 1024 at 256 px
    warp_res_blocks: int = 4  # paper Fig. 8
    warp_skip: str = "body"  # body | dual | none
    texture_corr_size: int = 64  # CoCosNet: 64 x 64 correspondence
    texture_alpha: float = 100.0  # CoCosNet default softmax sharpness
    texture_feat_channels: int = 256
    generator_base_channels: int = 64
    discriminator_base_channels: int = 64
    spectral_norm: bool = True


@dataclass
class ProjectConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------
# Dict <-> dataclass
# --------------------------------------------------------------------------------------


def _from_dict(cls, data: Dict[str, Any], prefix: str = ""):
    if not isinstance(data, dict):
        raise ConfigError(f"Section '{prefix or 'root'}' must be a mapping, got {type(data).__name__}")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(
            f"Unknown config key(s) in '{prefix or 'root'}': {unknown}. "
            f"Valid keys: {sorted(known)}"
        )
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        default_obj = f.default_factory() if f.default_factory is not dataclasses.MISSING else None
        if dataclasses.is_dataclass(default_obj):
            kwargs[name] = _from_dict(type(default_obj), value, f"{prefix}{name}.")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "split_ratios":
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml_with_bases(path: Path, _seen: Optional[set] = None) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    _seen = _seen or set()
    if path.resolve() in _seen:
        raise ConfigError(f"Circular _base_ include at {path}")
    _seen.add(path.resolve())
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")
    base_ref = data.pop("_base_", None)
    if base_ref:
        base = _read_yaml_with_bases(path.parent / base_ref, _seen)
        data = _deep_merge(base, data)
    return data


def _apply_override(data: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"Override must look like 'section.key=value', got '{override}'")
    key, raw = override.split("=", 1)
    parts = key.strip().split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"Override '{override}': '{part}' is not a section")
    raw = raw.strip()
    # "{cache_dir}/x" would otherwise be parsed as a YAML flow mapping.
    if re.search(r"\{[A-Za-z_]+\}", raw):
        node[parts[-1]] = raw
        return
    try:
        node[parts[-1]] = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Cannot parse value in override '{override}': {exc}") from None


# --------------------------------------------------------------------------------------
# Placeholder resolution and validation
# --------------------------------------------------------------------------------------


def _resolve_placeholders(cfg: ProjectConfig) -> None:
    """Expand ``{name}`` placeholders in path-like strings, iterating until stable."""
    names: Dict[str, Any] = {}

    def refresh_names() -> None:
        names.update(dataclasses.asdict(cfg.paths))
        names["dataset"] = cfg.data.dataset
        names["split_seed"] = cfg.data.split_seed
        names["experiment"] = cfg.experiment.name

    def fmt(value: str) -> str:
        for _ in range(10):
            try:
                new = value.format(**names)
            except KeyError as exc:
                raise ConfigError(f"Unknown placeholder {exc} in '{value}'") from None
            if new == value:
                return new
            value = new
        raise ConfigError(f"Placeholder expansion did not converge for '{value}'")

    refresh_names()
    for f in dataclasses.fields(cfg.paths):
        setattr(cfg.paths, f.name, fmt(getattr(cfg.paths, f.name)))
    refresh_names()
    for attr in ("root", "archive_dir", "split_file", "index_file"):
        setattr(cfg.data, attr, fmt(getattr(cfg.data, attr)))


def validate_config(cfg: ProjectConfig) -> None:
    exp, data = cfg.experiment, cfg.data
    if exp.run_mode not in {"smoke", "baseline", "extended"}:
        raise ConfigError(f"experiment.run_mode must be smoke|baseline|extended, got {exp.run_mode}")
    if exp.device not in {"auto", "cpu", "cuda"}:
        raise ConfigError(f"experiment.device must be auto|cpu|cuda, got {exp.device}")
    if data.dataset not in {"deepfashion_multimodal", "synthetic"}:
        raise ConfigError(f"data.dataset '{data.dataset}' is not supported")
    if (
        not isinstance(data.image_size, (list, tuple))
        or len(data.image_size) != 2
        or any((not isinstance(v, int)) or v < 32 or v % 16 for v in data.image_size)
    ):
        raise ConfigError(
            f"data.image_size must be [H, W] with integers >= 32 divisible by 16, got {data.image_size}"
        )
    if data.resize_mode not in {"letterbox", "stretch"}:
        raise ConfigError("data.resize_mode must be letterbox|stretch")
    if data.image_check not in {"none", "header", "full"}:
        raise ConfigError("data.image_check must be none|header|full")
    ratios = data.split_ratios
    if set(ratios) != {"train", "val", "test"}:
        raise ConfigError(f"data.split_ratios needs exactly train/val/test, got {sorted(ratios)}")
    if any(v < 0 for v in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ConfigError(f"data.split_ratios must be non-negative and sum to 1, got {ratios}")
    if cfg.reference.mode not in {"person_worn", "garment_only"}:
        raise ConfigError("reference.mode must be person_worn|garment_only")
    if cfg.labels.garment_label_set != "garment12":
        raise ConfigError("labels.garment_label_set must be garment12 (the only project garment set)")
    if cfg.labels.body_label_set not in {"body15", "pascal_person_part"}:
        raise ConfigError("labels.body_label_set must be body15 or pascal_person_part")
    overlap = set(cfg.labels.transfer_classes) & set(cfg.labels.preserve_classes)
    if overlap:
        raise ConfigError(f"labels.transfer_classes and preserve_classes overlap: {sorted(overlap)}")
    if cfg.training.grad_accum_steps < 1 or cfg.training.batch_size < 1:
        raise ConfigError("training.batch_size and grad_accum_steps must be >= 1")


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def load_config(
    path: Optional[PathLike] = None,
    overrides: Optional[Sequence[str]] = None,
    resolve: bool = True,
) -> ProjectConfig:
    """Load a :class:`ProjectConfig` from YAML (optional) plus dotted overrides."""
    data: Dict[str, Any] = _read_yaml_with_bases(Path(path)) if path else {}
    for override in overrides or []:
        _apply_override(data, override)
    cfg = _from_dict(ProjectConfig, data)
    if resolve:
        _resolve_placeholders(cfg)
    validate_config(cfg)
    return cfg


def save_config(cfg: ProjectConfig, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
    return path


def config_from_dict(data: Dict[str, Any], resolve: bool = True) -> ProjectConfig:
    cfg = _from_dict(ProjectConfig, copy.deepcopy(data))
    if resolve:
        _resolve_placeholders(cfg)
    validate_config(cfg)
    return cfg
