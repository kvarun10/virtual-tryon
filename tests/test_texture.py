"""Phase 5 tests: correspondence maths, texture model shapes, losses, data, resume, overfit.

CPU only, synthetic fixture, tiny widths, an untrained VGG (no downloads). Passing means the stage is
wired correctly and can fit data; it says nothing about image quality on real data.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.config import load_config
from src.data.dataset import prepare_dataset
from src.data.texture_dataset import TextureDataset, TextureSwapDataset, texture_collate
from src.evaluation.metrics import ssim
from src.models.alignment_network import AlignmentNetwork, correlation_bytes
from src.models.discriminator import MultiScaleDiscriminator
from src.models.spade_generator import SPADE, PositionalNorm, TextureGenerator
from src.training.checkpointing import CheckpointMismatchError
from src.training.losses import (VGGPerceptualLoss, VGGWeightsUnavailableError, feature_matching, multiscale_hinge_d,
                                 multiscale_hinge_g)
from src.training.train_texture import TextureTrainer

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["data.image_size=[64,64]", "model.texture_corr_size=16", "model.generator_base_channels=8",
         "model.texture_feat_channels=32", "model.discriminator_base_channels=8", "training.texture_vgg_init=random",
         "training.batch_size=2", "training.val_max_batches=2", "training.log_every=1000", "data.max_samples=null"]


def cfg_for(tmp_path, *extra):
    return load_config(ROOT / "configs" / "colab_smoke_test.yaml", [f"paths.drive_root={tmp_path.as_posix()}/drive", *SMALL, *extra])


# ------------------------------------------------------------------ correspondence maths (Eqs. 5-6)


def _align_with_features(fx, fy, alpha):
    net = AlignmentNetwork(12, 3, (64, 64), corr_size=4, feat_channels=16, alpha=alpha)
    net.encode_x = lambda layout: fx
    net.encode_y = lambda ex: fy
    return net


def test_correspondence_picks_matching_positions():
    """Features that identify positions: target u matches exemplar v = perm[u] -> I_S(u) = exemplar(perm[u])."""
    hw, c = 16, 16
    perm = torch.randperm(hw, generator=torch.Generator().manual_seed(0))
    ident = torch.eye(c)
    fy = ident.T.reshape(1, c, 4, 4)  # exemplar position v has one-hot feature e_v
    fx = ident[:, perm].reshape(1, c, 4, 4)  # target position u has feature e_perm[u]
    exemplar = torch.rand(1, 3, 64, 64) * 2 - 1
    net = _align_with_features(fx, fy, alpha=100.0)
    out = net(torch.zeros(1, 12, 64, 64), exemplar, exemplar, return_attention=True)
    small = out["exemplar_small"].flatten(2)[0]  # [3, HW]
    torch.testing.assert_close(out["warped"].flatten(2)[0], small[:, perm], atol=1e-3, rtol=0)
    torch.testing.assert_close(out["attention"].sum(-1), torch.ones(1, hw))  # rows sum to 1 (Eq. 6)
    torch.testing.assert_close(out["cycle"], out["exemplar_small"], atol=1e-3, rtol=0)  # perfect cycle for a bijection


def test_small_alpha_blurs_and_large_alpha_sharpens():
    fx = torch.randn(1, 16, 4, 4)
    exemplar = torch.rand(1, 3, 64, 64)
    soft = _align_with_features(fx, fx.clone(), alpha=0.001)(torch.zeros(1, 12, 64, 64), exemplar, exemplar)
    mean = soft["exemplar_small"].flatten(2).mean(-1, keepdim=True)
    torch.testing.assert_close(soft["warped"].flatten(2), mean.expand(-1, -1, 16), atol=1e-3, rtol=0)  # ~uniform average
    sharp = _align_with_features(fx, fx.clone(), alpha=1000.0)(torch.zeros(1, 12, 64, 64), exemplar, exemplar)
    torch.testing.assert_close(sharp["warped"], sharp["exemplar_small"], atol=1e-3, rtol=0)  # self-match = identity


def test_correlation_runs_in_fp32_under_autocast():
    g = TextureGenerator(12, (64, 64), corr_size=16, feat_channels=32, nf=8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = g(torch.rand(1, 12, 64, 64), torch.rand(1, 3, 64, 64), torch.rand(1, 12, 64, 64), return_attention=True)
    assert out["attention"].dtype == torch.float32 and out["warped"].dtype == torch.float32
    assert correlation_bytes(32, 1, factor=1) == 1024 * 1024 * 4 and correlation_bytes(64, 1, factor=1) == 4096 * 4096 * 4


# ------------------------------------------------------------------ model shapes / errors


@pytest.mark.parametrize("size,corr", [(64, 16), (128, 32)])
@pytest.mark.parametrize("ex_parsing", [True, False])
def test_texture_generator_shapes(size, corr, ex_parsing):
    g = TextureGenerator(12, (size, size), corr_size=corr, feat_channels=32, nf=8, exemplar_parsing=ex_parsing)
    out = g(torch.rand(2, 12, size, size), torch.rand(2, 3, size, size) * 2 - 1,
            torch.rand(2, 12, size, size) if ex_parsing else None)
    assert out["fake"].shape == (2, 3, size, size) and out["fake"].abs().max() <= 1
    assert out["warped"].shape == (2, 3, corr, corr) and out["feat_x"].shape == (2, 32, corr, corr)


def test_texture_generator_errors():
    with pytest.raises(ValueError, match="power-of-two"):
        TextureGenerator(12, (96, 96), corr_size=32)
    g = TextureGenerator(12, (64, 64), corr_size=16, feat_channels=32, nf=8)
    with pytest.raises(ValueError, match="parsing"):
        g(torch.rand(1, 12, 64, 64), torch.rand(1, 3, 64, 64), None)
    with pytest.raises(ValueError, match="differ"):
        g(torch.rand(1, 12, 64, 64), torch.rand(1, 3, 32, 32), torch.rand(1, 12, 32, 32))


def test_spade_and_positional_norm():
    x = torch.randn(2, 8, 4, 4) * 5 + 3
    y = PositionalNorm()(x)
    torch.testing.assert_close(y.mean(1), torch.zeros(2, 4, 4), atol=1e-5, rtol=0)
    s = SPADE(8, 15)
    assert s(x, torch.rand(2, 15, 64, 64)).shape == x.shape  # guidance resized to the activation


def test_multiscale_discriminator_and_losses():
    d = MultiScaleDiscriminator(15, base=8, scales=2)
    real, fake = d(torch.rand(2, 15, 64, 64)), d(torch.rand(2, 15, 64, 64))
    assert len(real) == 2 and len(real[0][1]) == 4
    for v in (multiscale_hinge_d(real, fake), multiscale_hinge_g(fake), feature_matching(real, fake)):
        assert torch.isfinite(v)
    assert feature_matching(real, real) == 0


def test_vgg_gating_and_random_init(tmp_path):
    with pytest.raises(VGGWeightsUnavailableError, match="allow_vgg_download"):
        VGGPerceptualLoss("imagenet", cache_dir=tmp_path, allow_download=False)
    v = VGGPerceptualLoss("random")
    a = torch.rand(1, 3, 32, 32) * 2 - 1
    assert v(a, a).item() == 0 and v(a, -a).item() > 0
    assert not any(p.requires_grad for p in v.parameters())


def test_ssim_properties():
    a = torch.rand(2, 3, 32, 32)
    torch.testing.assert_close(ssim(a, a), torch.ones(2))
    torch.testing.assert_close(ssim(a, a.flip(0).flip(0) * 0.5), ssim(a * 0.5, a))  # symmetric
    assert (ssim(a, torch.rand_like(a)) < 0.2).all()
    with pytest.raises(ValueError):
        ssim(a, a[:, :, :16])


# ------------------------------------------------------------------ data


def test_texture_pairs(tmp_path):
    cfg = cfg_for(tmp_path)
    prepared = prepare_dataset(cfg)
    ds = TextureDataset(prepared, "train", cfg, train=True, cross_view_prob=1.0)
    s = ds[0]
    assert s["layout"].shape == (12, 64, 64) and s["image"].shape == (3, 64, 64)
    assert s["exemplar_parsing"].shape == (12, 64, 64) and -1 <= s["image"].min() and s["image"].max() <= 1
    for i in range(len(ds)):
        it = ds[i]
        assert it["mode"] == "cross_view" and it["src"] != it["tgt"]
        assert prepared.index[it["src"]].item_id == prepared.index[it["tgt"]].item_id
    pseudo = TextureDataset(prepared, "train", cfg, train=True, cross_view_prob=0.0)
    p = pseudo[0]
    assert p["mode"] == "pseudo" and p["src"] == p["tgt"] and not torch.equal(p["exemplar"], p["image"])
    val = TextureDataset(prepared, "val", cfg, train=False, cross_view_prob=1.0)
    assert torch.equal(val[0]["exemplar"], val[0]["exemplar"])
    swap = TextureSwapDataset(ds, n=4)
    for k in range(len(swap)):
        it = swap[k]
        assert prepared.index[it["src"]].item_id != prepared.index[it["tgt"]].item_id
    batch = texture_collate([ds[0], ds[1]])
    assert batch["layout"].shape == (2, 12, 64, 64) and len(batch["mode"]) == 2


# ------------------------------------------------------------------ trainer


def _params(t):
    return torch.cat([p.detach().flatten() for p in t.gen.parameters()])


def test_texture_losses_finite_and_all_params_get_gradients(tmp_path):
    t = TextureTrainer(cfg_for(tmp_path), verbose=False)
    batch = texture_collate([t.train_ds[0], t.train_ds[1]])
    rec = t.micro_step(batch, accum=1)
    assert all(np.isfinite(v) for v in rec.values())
    assert rec["g_adv"] != 0 and rec["vgg"] > 0 and rec["warp"] > 0 and rec["domain"] > 0 and rec["d"] > 0
    no_grad = [n for n, p in t.gen.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not no_grad, f"parameters without finite gradients: {no_grad[:5]}"


def test_texture_resume_matches_uninterrupted_run(tmp_path):
    over = ["training.val_every=1000", "training.checkpoint_every=1000", "training.epochs=10"]
    a = TextureTrainer(cfg_for(tmp_path / "a", *over), verbose=False)
    per_epoch = len(a.train_ds) // 2
    total, cut = per_epoch + 1, per_epoch // 2 + 1
    a.fit(max_steps=total, resume=False)
    TextureTrainer(cfg_for(tmp_path / "b", *over), verbose=False).fit(max_steps=cut, resume=False)
    c = TextureTrainer(cfg_for(tmp_path / "b", *over), verbose=False)
    assert c.fit(max_steps=total)["step"] == total and c.epoch == 1
    assert torch.allclose(_params(a), _params(c), atol=1e-5), "resumed texture run diverged"


def test_texture_checkpoint_mismatch_and_eval(tmp_path):
    TextureTrainer(cfg_for(tmp_path), verbose=False).fit(max_steps=2, resume=False)
    t = TextureTrainer(cfg_for(tmp_path), verbose=False)
    m = t.evaluate_checkpoint("last")
    assert m["n_images"] == len(t.val_ds) and 0 <= m["ssim"] <= 1 and m["kind"] == "eval_last"
    assert list(t.out_dir.glob("eval_last_step*.png")) and list(t.out_dir.glob("val_step*.png"))
    other = TextureTrainer(cfg_for(tmp_path, "model.generator_base_channels=16"), verbose=False)
    with pytest.raises(CheckpointMismatchError):
        other.resume()


def test_texture_overfit_one_batch(tmp_path):
    """The generator must be able to reproduce one fixed batch (pixel L1 only)."""
    cfg = cfg_for(tmp_path)
    prepared = prepare_dataset(cfg)
    ds = TextureDataset(prepared, "val", cfg, train=False, cross_view_prob=1.0)
    b = texture_collate([ds[i] for i in range(min(2, len(ds)))])
    torch.manual_seed(0)
    g = TextureGenerator(12, (64, 64), corr_size=16, feat_channels=32, nf=16)
    opt = torch.optim.Adam(g.parameters(), lr=2e-3, betas=(0.5, 0.999))
    first = None
    for _ in range(300):
        loss = F.l1_loss(g(b["layout"], b["exemplar"], b["exemplar_parsing"])["fake"], b["image"])
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        fake = g(b["layout"], b["exemplar"], b["exemplar_parsing"])["fake"]
    s = ssim((fake + 1) / 2, (b["image"] + 1) / 2).mean().item()
    assert loss.item() < 0.25 * first, f"L1 {first:.3f} -> {loss.item():.3f}"
    assert s > 0.8, f"SSIM after overfitting only {s:.3f}"
