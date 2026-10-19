"""Archive staging, tested with zip files built from the synthetic fixture (no network)."""

import json
import shutil
import zipfile

import pytest

from src.config import load_config
from src.data.dataset import prepare_dataset
from src.data.fetch import FetchError, download_archive, fetch_and_stage, find_archive, stage_archive, storage_plan
from src.data.synthetic import FixtureSpec, make_fixture


def _zip_dir(src_dir, zip_path, inner_prefix):
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sorted(src_dir.iterdir()):
            zf.write(f, f"{inner_prefix}/{f.name}")


@pytest.fixture
def fake_archives(tmp_path):
    """Zips with layouts that differ from the canonical one (unknown real layout)."""
    fx = tmp_path / "fx"
    make_fixture(fx, FixtureSpec(n_items=3, views_per_item=3, parsing_fraction=1.0))
    arch = tmp_path / "drive" / "archives"
    arch.mkdir(parents=True)
    _zip_dir(fx / "images", arch / "images.zip", "DeepFashion-MultiModal/image")  # nested + other name
    _zip_dir(fx / "segm", arch / "segm.zip", "parsing")
    _zip_dir(fx / "keypoints", arch / "keypoints.zip", "keypoints")
    _zip_dir(fx / "densepose", arch / "densepose.zip", "dp")
    return arch


def test_plan_touches_nothing(tmp_path):
    plan = storage_plan("images_parsing", tmp_path / "a", tmp_path / "b")
    assert set(plan["parts"]) == {"images", "parsing", "keypoints"} and plan["local_pc_bytes"] == 0
    assert 5.4e9 < plan["drive_bytes_approx"] < 5.6e9
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_missing_archive_without_permission(tmp_path):
    with pytest.raises(FetchError, match="allow_download"):
        download_archive("images", tmp_path)


def test_stage_normalises_layout_and_is_idempotent(fake_archives, tmp_path):
    root = tmp_path / "local" / "DFMM"
    info = stage_archive("images", find_archive("images", fake_archives), root)
    assert info["files"] == 9 and (root / "images").is_dir()
    assert "warning" in info  # 9 != 44,096: count mismatch is reported, not hidden
    again = stage_archive("images", find_archive("images", fake_archives), root)
    assert again == info  # second call skips extraction


def test_fetch_then_prepare(fake_archives, tmp_path):
    cfg = load_config(overrides=[f"paths.drive_root={(tmp_path / 'drive').as_posix()}",
                                 f"data.archive_dir={fake_archives.as_posix()}",
                                 f"data.root={(tmp_path / 'local' / 'DFMM').as_posix()}"])
    fetch_and_stage(cfg, "all", allow_download=False, progress=lambda *_: None)
    prepared = prepare_dataset(cfg)
    assert prepared.report.n_images == 9 and prepared.report.n_with_parsing == 9
    assert not list(fake_archives.glob("*.part"))  # Drive folder only holds the archives


def test_non_zip_is_rejected(tmp_path):
    bad = tmp_path / "images.bin"
    bad.write_bytes(b"not a zip")
    with pytest.raises(FetchError, match="not a zip"):
        stage_archive("images", bad, tmp_path / "root")


def test_images_parsing_only_without_densepose(fake_archives, tmp_path):
    """The 'images + parsing' download: indexing/splits work; body maps fail with a clear message."""
    from src.data.dataset import DatasetUnavailableError, PreTreatmentDataset

    base = [f"paths.drive_root={(tmp_path / 'drive').as_posix()}", f"data.archive_dir={fake_archives.as_posix()}",
            f"data.root={(tmp_path / 'local' / 'DFMM').as_posix()}"]
    fetch_and_stage(load_config(overrides=base), "images_parsing", progress=lambda *_: None)
    with pytest.raises(DatasetUnavailableError, match="require_densepose=false"):
        prepare_dataset(load_config(overrides=base))
    cfg = load_config(overrides=base + ["data.require_densepose=false"])
    prepared = prepare_dataset(cfg)
    assert prepared.report.n_with_parsing == 9 and prepared.report.n_with_densepose == 0
    with pytest.raises(FileNotFoundError, match="densepose"):
        PreTreatmentDataset(prepared, "train", cfg)[0]
