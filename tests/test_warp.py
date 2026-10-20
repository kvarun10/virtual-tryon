"""Phase 4 tests: warp data, model shapes, losses, checkpoint/resume, overfit-one-batch.

Everything runs on CPU with the SYNTHETIC fixture and small model widths. Passing these tests means
the code is wired correctly and can fit data; it says nothing about quality on real data.
"""

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data.body_maps import (BodyMapsUnavailableError, BodyMapSource, archive_body_maps, precompute_body_maps,
                                stage_body_maps)
from src.data.dataset import prepare_dataset
from src.data.labels import convert_labels, get_label_set
from src.data.preprocessing import load_label_map
from src.data.warp_dataset import IGNORE, SwapDiagnosticDataset, WarpTripletDataset, jitter_body_parts, per_class_affine
from src.models.discriminator import PatchDiscriminator
from src.models.segmentation import SegmentationBackend, SegmentationResult
from src.models.warp_network import DualPathUNet
from src.training.checkpointing import CheckpointMismatchError
from src.training.losses import GANLoss, masked_probs, real_maps, segmentation_ce
from src.training.train_warp import WarpTrainer

SMALL = ["data.image_size=[64,64]", "model.warp_base_channels=8", "model.warp_max_channels=32",
         "model.warp_res_blocks=1", "model.discriminator_base_channels=8", "training.batch_size=2",
         "training.val_max_batches=2", "training.warp_swap_samples=4", "training.log_every=1000"]


def cfg_for(tmp_path, *extra):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs" / "colab_smoke_test.yaml",
                       [f"paths.drive_root={tmp_path.as_posix()}/drive", "data.max_samples=null", *SMALL, *extra])


# ------------------------------------------------------------------ model shapes


def test_paper_feature_shapes_at_256():
    g = DualPathUNet(12, 7, 12, (256, 256))  # default widths = paper numbers
    s = g.feature_shapes()
    assert s["body_top"] == (1, 512, 16, 16)  # [P] 512 x 16 x 16
    assert s["cloth_code"] == (1, 1024, 2, 2)  # [P] 2 x 2 x 1024
    assert s["cloth_up"] == (1, 512, 16, 16)  # [P] up-sampled to 512 x 16 x 16
    assert sum(isinstance(m, torch.nn.Module) for m in g.res) == 4  # [P] 4 residual blocks


@pytest.mark.parametrize("skip", ["body", "dual", "none"])
@pytest.mark.parametrize("size,kb", [((64, 64), 7), ((128, 128), 15), ((128, 64), 7)])
def test_generator_output_shapes(skip, size, kb):
    g = DualPathUNet(12, kb, 12, size, base=8, max_channels=32, res_blocks=1, skip=skip)
    out = g(torch.randn(2, 12, *size), torch.randn(2, kb, *size))
    assert out.shape == (2, 12, *size)
    d = PatchDiscriminator(12 + kb, base=8)
    assert d(torch.cat([out, torch.randn(2, kb, *size)], 1)).ndim == 4


def test_generator_rejects_bad_inputs():
    with pytest.raises(ValueError, match="power of two"):
        DualPathUNet(12, 7, 12, (96, 96), cloth_bottleneck=2)
    with pytest.raises(ValueError, match="too small"):
        DualPathUNet(12, 7, 12, (32, 32), cloth_bottleneck=4)
    g = DualPathUNet(12, 7, 12, (64, 64), base=8, max_channels=32, res_blocks=1)
    with pytest.raises(ValueError, match="body_label_set"):
        g(torch.zeros(1, 12, 64, 64), torch.zeros(1, 15, 64, 64))  # body15 maps into a 7-part model
    with pytest.raises(ValueError, match="image_size"):
        g(torch.zeros(1, 12, 128, 128), torch.zeros(1, 7, 128, 128))


# ------------------------------------------------------------------ data


def test_per_class_affine_and_body_jitter():
    lab = np.zeros((64, 64), np.uint8)
    lab[10:30, 20:40] = 4
    lab[35:60, 22:38] = 8
    rng = np.random.default_rng(0)
    out = per_class_affine(lab, 12, rng, 15, (0.85, 1.15), 0.15)
    assert out.shape == (12, 64, 64) and out[4].sum() > 0 and out[8].sum() > 0
    assert not np.array_equal(out[4] > 0, lab == 4)  # moved
    assert out[[0, 4, 8]].max(0).min() == 1 and out[1:].sum() == out[[4, 8]].sum()  # only input classes
    body = np.zeros((40, 40), np.uint8)
    body[10:30, 10:30] = 2
    body[30:38, 12:28] = 5
    outs = [jitter_body_parts(body, np.random.default_rng(s), 3) for s in range(8)]
    assert all(set(np.unique(j)) <= {0, 2, 5} for j in outs)  # never invents parts
    assert sum(not np.array_equal(j, body) for j in outs) >= 4  # each part keeps its shape with p=1/3
    assert np.array_equal(jitter_body_parts(body, np.random.default_rng(0), 0), body)  # disabled


def test_triplets(tmp_path):
    cfg = cfg_for(tmp_path)
    prepared = prepare_dataset(cfg)
    ds = WarpTripletDataset(prepared, "train", cfg, train=True)
    G = get_label_set("garment12")
    s = ds[0]
    assert s["cloth"].shape == (12, 64, 64) and s["body"].shape == (15, 64, 64) and s["target"].shape == (64, 64)
    assert torch.allclose(s["body"].sum(0), torch.ones(64, 64))  # one-hot
    present = set(s["target"].unique().tolist())
    assert G.index("face") not in present and G.index("hair") not in present and IGNORE in present
    # input C_seg contains only transfer classes
    cloth_classes = {k for k in range(12) if s["cloth"][k].sum() > 0} - {0}
    assert cloth_classes <= set(G.indices(cfg.labels.transfer_classes))
    # cross-view: source is another photo of the same item
    cv = WarpTripletDataset(prepared, "train", cfg, train=True, pairing="cross_view")
    for i in range(len(cv)):
        item = cv[i]
        assert item["mode"] == "cross_view" and item["src"] != item["tgt"]
        assert prepared.index[item["src"]].item_id == prepared.index[item["tgt"]].item_id
    # reproducibility: same epoch -> same sample; new epoch -> new augmentation; validation fixed
    assert torch.equal(ds[1]["cloth"], ds[1]["cloth"])
    a = ds[1]["cloth"]; ds.set_epoch(1)
    assert not torch.equal(a, ds[1]["cloth"])
    val = WarpTripletDataset(prepared, "val", cfg, train=False)
    assert torch.equal(val[0]["cloth"], val[0]["cloth"])
    swap = SwapDiagnosticDataset(ds, n=4)
    for k in range(len(swap)):
        it = swap[k]
        assert prepared.index[it["src"]].item_id != prepared.index[it["tgt"]].item_id


class _FakePascal(SegmentationBackend):
    """Stands in for SCHP-Pascal: converts the fixture's DensePose to 7 parts (no network)."""

    name, role, label_set = "schp_pascal", "body", "pascal_person_part"

    def __init__(self, root, layout):
        self.root, self.layout = root, layout

    def predict(self, image, key=None):
        raise NotImplementedError

    def describe(self):
        return {"name": "schp_pascal", "fake": True}


def test_precomputed_body_maps_roundtrip(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path, "segmentation.body_backend=schp_pascal", "labels.body_label_set=pascal_person_part")
    prepared = prepare_dataset(cfg)
    with pytest.raises(BodyMapsUnavailableError, match="Section 8a"):
        BodyMapSource(cfg, prepared.index.root)
    fake = _FakePascal(prepared.index.root, cfg.data.layout)
    names = {}

    def predict(image, key=None):  # look the record up by pixel content order
        rec = names["current"]
        dp = load_label_map(prepared.index.root / rec.densepose, channel=2)
        return SegmentationResult(convert_labels(dp, "densepose", "pascal_person_part"), "pascal_person_part", "schp_pascal", "body")

    fake.predict = predict
    recs = prepared.records("train")
    import src.data.body_maps as bm

    orig = bm.load_image
    monkeypatch.setattr(bm, "load_image", lambda p: (names.__setitem__("current", next(r for r in recs if r.image in str(p).replace("\\", "/"))), orig(p))[1])
    info = precompute_body_maps(cfg, recs, prepared.index.root, backend=fake, progress=False)
    assert info["computed"] == len(recs)
    assert precompute_body_maps(cfg, recs, prepared.index.root, backend=fake, progress=False)["computed"] == 0  # resumable
    archive = archive_body_maps(cfg)
    import shutil

    shutil.rmtree(tmp_path / "drive" / "cache" / "body_maps")
    assert stage_body_maps(cfg)["status"] == "restored" and archive.is_file()
    src = BodyMapSource(cfg, prepared.index.root)
    assert src.load(recs[0]).max() < 7 and src.num_classes == 7
    ds = WarpTripletDataset(prepared, "train", cfg, train=True, body_source=src)
    assert ds[0]["body"].shape[0] == 7


# ------------------------------------------------------------------ losses


def test_losses_finite_and_backward():
    torch.manual_seed(0)
    g = DualPathUNet(12, 7, 12, (64, 64), base=8, max_channels=32, res_blocks=1)
    d = PatchDiscriminator(19, base=8)
    cloth, body = torch.rand(2, 12, 64, 64), torch.rand(2, 7, 64, 64)
    target = torch.randint(0, 12, (2, 64, 64))
    target[:, :10] = 255
    logits = g(cloth, body)
    ce = segmentation_ce(logits, target)
    for mode in ("lsgan", "hinge", "vanilla"):
        gan = GANLoss(mode)
        fake = masked_probs(logits, target)
        assert torch.isfinite(gan.g_loss(d(torch.cat([fake, body], 1))))
        assert torch.isfinite(gan.d_loss(d(torch.cat([real_maps(target, 12, 0.1), body], 1)), d(torch.cat([fake.detach(), body], 1))))
    (ce + 0.1 * GANLoss("lsgan").g_loss(d(torch.cat([masked_probs(logits, target), body], 1)))).backward()
    assert torch.isfinite(ce) and ce.item() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters())
    # masked pixels carry no D signal; all-ignore batches give a zero, differentiable loss
    assert masked_probs(logits, target)[:, :, :10].abs().sum() == 0
    zero = segmentation_ce(logits, torch.full_like(target, 255))
    assert zero.item() == 0.0 and zero.requires_grad
    with pytest.raises(ValueError):
        segmentation_ce(logits, torch.full((2, 64, 64), 20))


# ------------------------------------------------------------------ trainer: resume / mismatch / overfit


def _params(trainer):
    return torch.cat([p.detach().flatten() for p in trainer.gen.parameters()])


def test_resume_matches_uninterrupted_run(tmp_path):
    over = ["training.warp_adv_start_step=1", "training.val_every=1000", "training.checkpoint_every=1000",
            "training.epochs=10"]
    a = WarpTrainer(cfg_for(tmp_path / "a", *over), verbose=False)
    per_epoch = len(a.train_ds) // 2
    total, cut = per_epoch + 2, per_epoch // 2 + 1  # interrupted mid-epoch 0, finishes inside epoch 1
    a.fit(max_steps=total, resume=False)
    b = WarpTrainer(cfg_for(tmp_path / "b", *over), verbose=False)
    b.fit(max_steps=cut, resume=False)
    assert b.epoch == 0 and 0 < b.batch_in_epoch < per_epoch
    c = WarpTrainer(cfg_for(tmp_path / "b", *over), verbose=False)  # fresh process, same checkpoint folder
    out = c.fit(max_steps=total)
    assert out["step"] == total and c.epoch == 1
    assert torch.allclose(_params(a), _params(c), atol=1e-6), "resumed run diverged from the uninterrupted run"
    assert (tmp_path / "b" / "drive" / "checkpoints" / "smoke_test" / "warp" / "last.pt").is_file()


def test_checkpoint_mismatch_detected(tmp_path):
    WarpTrainer(cfg_for(tmp_path), verbose=False).fit(max_steps=1, resume=False)
    other = WarpTrainer(cfg_for(tmp_path, "labels.body_label_set=pascal_person_part"), verbose=False)
    with pytest.raises(CheckpointMismatchError, match="body_label_set"):
        other.resume()


def test_validation_outputs(tmp_path):
    t = WarpTrainer(cfg_for(tmp_path), verbose=False)
    m = t.validate()
    assert 0 <= m["pixel_accuracy"] <= 1 and np.isfinite(m["ce"])
    assert np.isnan(m["class_fidelity"]) or 0 <= m["class_fidelity"] <= 1
    assert list(t.out_dir.glob("val_step*.png"))
    assert t.use_amp is False  # CPU


def test_overfit_one_batch(tmp_path):
    """The generator must be able to memorise one fixed batch (CE only)."""
    cfg = cfg_for(tmp_path, "model.warp_base_channels=16", "model.warp_max_channels=64")
    prepared = prepare_dataset(cfg)
    ds = WarpTripletDataset(prepared, "val", cfg, train=False)  # deterministic samples
    batch = [ds[i] for i in range(min(2, len(ds)))]
    cloth = torch.stack([b["cloth"] for b in batch]); body = torch.stack([b["body"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    torch.manual_seed(0)
    g = DualPathUNet(12, body.shape[1], 12, (64, 64), base=16, max_channels=64, res_blocks=1)
    opt = torch.optim.Adam(g.parameters(), lr=2e-3)
    first = None
    for _ in range(250):
        loss = segmentation_ce(g(cloth, body), target)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = g(cloth, body).argmax(1)
    valid = target != IGNORE
    acc = (pred[valid] == target[valid]).float().mean().item()
    assert loss.item() < 0.1 * first, f"loss {first:.3f} -> {loss.item():.3f}"
    assert acc > 0.97, f"pixel accuracy after overfitting only {acc:.3f}"
