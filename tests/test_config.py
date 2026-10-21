from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, ProjectConfig, load_config, save_config

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_valid():
    cfg = load_config()
    assert isinstance(cfg, ProjectConfig)
    assert cfg.training.betas == [0.0, 0.999] and cfg.training.lr_generator == 1e-4  # paper Sec. 5.1
    assert cfg.data.root.endswith("datasets/DeepFashion-MultiModal")  # placeholders resolved


@pytest.mark.parametrize("name", ["colab_smoke_test.yaml", "colab_baseline.yaml", "extended_training.yaml", "colab_pairfilter.yaml"])
def test_shipped_configs_load(name):
    cfg = load_config(ROOT / "configs" / name)
    assert "{" not in cfg.data.split_file and "{" not in cfg.paths.cache_dir


def test_base_inheritance():
    cfg = load_config(ROOT / "configs" / "extended_training.yaml")
    assert cfg.data.image_size == [256, 256]  # own value
    assert cfg.segmentation.clothing_backend == "schp_lip"  # inherited from colab_baseline
    assert cfg.paths.drive_root == "/content/drive/MyDrive/VirtualTryOn"


def test_overrides_and_placeholders():
    cfg = load_config(overrides=["paths.drive_root=/x", "data.image_size=[128, 64]", "data.split_seed=7"])
    assert cfg.data.image_size == [128, 64]
    assert cfg.paths.cache_dir == "/x/cache"
    assert cfg.data.split_file == "/x/cache/splits/deepfashion_multimodal_seed7.json"


@pytest.mark.parametrize(
    "override",
    [
        "data.bach_size=4",  # typo -> unknown key
        "data.image_size=[100, 100]",  # not divisible by 16
        "data.split_ratios={train: 0.5, val: 0.1, test: 0.1}",
        "experiment.device=tpu",
        "labels.preserve_classes=[top]",  # overlaps transfer classes
        "labels.body_label_set=lip",
        "reference.mode=flat",
    ],
)
def test_invalid_values_rejected(override):
    with pytest.raises(ConfigError):
        load_config(overrides=[override])


def test_save_roundtrip(tmp_path):
    cfg = load_config(overrides=["experiment.name=roundtrip"])
    path = save_config(cfg, tmp_path / "cfg.yaml")
    cfg2 = load_config(path)
    assert cfg2.to_dict() == cfg.to_dict()


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")


def test_circular_base(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"_base_": "b.yaml"}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"_base_": "a.yaml"}))
    with pytest.raises(ConfigError, match="Circular"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize("name", ["colab_smoke_test.yaml", "colab_baseline.yaml", "extended_training.yaml", "colab_pairfilter.yaml"])
def test_shipped_configs_name_valid_backends(name):
    from src.models.segmentation import BACKENDS

    cfg = load_config(ROOT / "configs" / name)
    assert BACKENDS.get(cfg.segmentation.clothing_backend) == "clothing"
    assert BACKENDS.get(cfg.segmentation.body_backend) == "body"
    if cfg.segmentation.body_backend == "schp_pascal":
        assert cfg.labels.body_label_set == "pascal_person_part"  # 7 parts cannot become body15
