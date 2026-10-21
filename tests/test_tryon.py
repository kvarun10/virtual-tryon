"""Phase 6: full try-on pipeline (Stage A -> warp -> identity labels -> texture -> face/hair paste).

Tiny models trained for 2 steps on the SYNTHETIC fixture: these tests check wiring, shapes, checkpoint
loading, identity handling and evaluation code, not image quality.
"""

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.data.dataset import prepare_dataset
from src.data.labels import get_label_set
from src.data.preprocessing import load_image
from src.evaluation.end_to_end import evaluate_end_to_end, tryon_swap_grid
from src.inference.pipeline import PreTreatment, StageNotAvailableError, try_on
from src.inference.tryon import ModelsUnavailableError, load_tryon_models, tryon_core
from src.training.checkpointing import CheckpointMismatchError
from src.training.train_texture import TextureTrainer
from src.training.train_warp import WarpTrainer

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["data.image_size=[64,64]", "data.max_samples=null", "model.texture_corr_size=16",
         "model.texture_feat_channels=32", "model.generator_base_channels=8", "model.discriminator_base_channels=8",
         "training.batch_size=2", "training.val_max_batches=1", "training.log_every=1000", "training.warp_swap_samples=2"]


def cfg_for(tmp_path, *extra):
    return load_config(ROOT / "configs" / "colab_smoke_test.yaml", [f"paths.drive_root={tmp_path.as_posix()}/drive", *SMALL, *extra])


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tryon")
    cfg = cfg_for(tmp)
    prepared = prepare_dataset(cfg)
    WarpTrainer(cfg, prepared, verbose=False).fit(max_steps=2, resume=False)
    TextureTrainer(cfg, prepared, verbose=False).fit(max_steps=2, resume=False)
    return cfg, prepared, load_tryon_models(cfg)


def test_models_load_in_eval_mode(trained):
    cfg, _, models = trained
    assert not models.warp.training and not models.texture.training
    assert models.info["warp"]["step"] == 2 and models.info["texture"]["step"] == 2
    G = get_label_set("garment12")
    assert set(models.untrained_ids) == set(G.indices(["hair", "face", "headwear", "accessory"]))


def test_full_try_on_and_saved_outputs(trained, tmp_path):
    cfg, prepared, models = trained
    a, b = prepared.records("train")[0], prepared.records("train")[3]
    pre = PreTreatment(cfg)
    res = try_on(load_image(prepared.index.root / a.image), load_image(prepared.index.root / b.image), cfg,
                 reference_key=a.name, target_key=b.name, pretreatment=pre, models=models, save_dir=tmp_path / "out")
    assert res.completed_stages == ("pretreatment", "warp", "texture")
    assert res.output.shape == (64, 64, 3) and res.output.dtype == np.uint8
    G = get_label_set("garment12")
    assert not np.isin(res.warped_garment, G.indices(["hair", "face", "headwear", "accessory"])).any()
    for name in ("overview", "reference", "C_seg", "B_seg", "C_seg_warped", "layout_with_identity", "warped_exemplar_I_S",
                 "generated_raw", "result"):
        assert (tmp_path / "out" / f"{name}.png").is_file()
    with pytest.raises(StageNotAvailableError, match="load_tryon_models"):
        try_on(load_image(prepared.index.root / a.image), load_image(prepared.index.root / b.image), cfg,
               reference_key=a.name, target_key=b.name, pretreatment=pre)


def test_identity_paste_uses_target_pixels(trained):
    cfg, prepared, models = trained
    G = get_label_set("garment12")
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    target = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    parse = np.zeros((64, 64), np.uint8)
    parse[10:40, 20:44] = G.index("top")
    tparse = parse.copy()
    tparse[2:12, 26:38] = G.index("face")  # target face block
    body = np.zeros((64, 64), np.uint8)
    body[10:60, 20:44] = 2
    core = tryon_core(models, ref, parse, target, body, tparse, feather_sigma=0.0)
    face = tparse == G.index("face")
    assert (core["layout"][face] == G.index("face")).all()  # paper Sec. 4.2: target face label in the layout
    assert np.array_equal(core["result"][face], target[face])  # pixels pasted from the target
    with pytest.raises(ValueError, match="model size"):
        tryon_core(models, ref[:32], parse, target, body, tparse)


def test_missing_and_mismatched_checkpoints(trained, tmp_path):
    cfg, _, _ = trained
    with pytest.raises(ModelsUnavailableError, match="Train the warp"):
        load_tryon_models(cfg_for(tmp_path))  # empty drive
    wider = load_config(ROOT / "configs" / "colab_smoke_test.yaml",
                        [f"paths.drive_root={Path(cfg.paths.drive_root).as_posix()}", *SMALL, "model.warp_base_channels=32"])
    with pytest.raises(CheckpointMismatchError, match="warp checkpoint"):
        load_tryon_models(wider)


def test_end_to_end_evaluation_and_swap_grid(trained, tmp_path):
    cfg, prepared, models = trained
    out = evaluate_end_to_end(models, prepared, cfg, split="train", limit=3, grid_path=tmp_path / "e2e.png", progress=False)
    for k in ("e2e_ssim", "e2e_pasted_ssim", "gt_layout_ssim", "copy_ssim", "e2e_l1", "warp_miou_7"):
        assert k in out and np.isfinite(out[k]), k
    assert out["pairs"] == 3 and (tmp_path / "e2e.png").is_file()
    grid = tryon_swap_grid(models, prepared, cfg, split="train", n=3, path=tmp_path / "swap.png")
    assert grid is not None and grid.shape[1] == 6 * 64
