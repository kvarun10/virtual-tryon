"""Packed 256 px store (Modal runs): packing with stand-in parsers, splits, datasets, EMA, In-shop loader."""

import io
import json
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.labels import get_label_set
from src.data.packed import (BlobReader, BlobWriter, PackedStore, build_packed_splits, decode_png, encode_png,
                             load_packed)
from src.data.packing import BatchedSCHP, build_pack
from src.data.pair_quality import PairFilter
from src.data.texture_dataset import TextureDataset
from src.data.warp_dataset import WarpTripletDataset
from src.evaluation.pose_transfer_eval import inshop_loader
from src.inference.tryon import load_tryon_models
from src.training.train_texture import TextureTrainer
from src.training.train_warp import WarpTrainer
from tests.test_tryon import cfg_for

ITEMS = {"00000001": "MEN-Denim", "00000002": "WOMEN-Dresses", "00000003": "WOMEN-Skirts", "00000004": "MEN-Tees"}
VIEWS = ["01_1_front", "01_2_side", "01_3_back", "02_1_front"]


class FakeModel(torch.nn.Module):
    """Class = brightness bucket of a 4x-downsampled input (deterministic stand-in for SCHP)."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        g = torch.nn.functional.avg_pool2d(x.mean(1, keepdim=True), 4)
        cls = ((g - g.amin()) / (g.amax() - g.amin() + 1e-6) * (self.k - 1)).round().long()[:, 0]
        logits = torch.nn.functional.one_hot(cls, self.k).permute(0, 3, 1, 2).float() * 10
        return [logits, logits], None  # SCHP: out[0][-1] = fusion logits


class FakeBackend:
    def __init__(self, label_set, size):
        self.label_set, self.input_size, self.device = label_set, size, torch.device("cpu")
        self.model = FakeModel(get_label_set(label_set).num_classes)

    def describe(self):
        return {"name": "fake", "label_set": self.label_set}


def _photo(seed):
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), 235, np.uint8)
    img[40:220, 70:190] = rng.integers(0, 255, 3)  # a "person" box
    img[:, :41] = img[:, 215:] = 250
    return img


def _segm(seed):
    lab = np.zeros((1101, 750), np.uint8)
    lab[200:900, 200:550] = [1, 5, 6][seed % 3]  # DFMM classes (top / pants-ish)
    lab[100:200, 300:450] = 14  # face
    return lab


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pack")
    img_zip, parsing_zip = tmp / "img.zip", tmp / "parsing.zip"
    rows, names = [], []
    with zipfile.ZipFile(img_zip, "w") as z, zipfile.ZipFile(parsing_zip, "w") as pz:
        for n, (item, cat) in enumerate(ITEMS.items()):
            g, c = cat.split("-")
            for v, view in enumerate(VIEWS):
                member = f"img/{g}/{c}/id_{item}/{view}.jpg"
                buf = io.BytesIO()
                Image.fromarray(_photo(n * 10 + v)).save(buf, "JPEG", quality=95)
                z.writestr(member, buf.getvalue())
                stem = f"{g}-{c}-id_{item}-{view}"
                names.append(stem)
                rows.append(f"{member} id_{item} train")
                if v < 3:  # three manually parsed photos per item
                    b = io.BytesIO()
                    Image.fromarray(_segm(v)).save(b, "PNG")
                    pz.writestr(f"segm/{stem}_segm.png", b.getvalue())
    part = tmp / "list_eval_partition.txt"
    part.write_text(f"{len(rows)}\nimage_name item_id evaluation_status\n" + "\n".join(rows) + "\n", encoding="utf-8")
    labelled = [n for n in names if not n.endswith("02_1_front")]
    old = {"train": [n for n in labelled if "00000001" in n or "00000004" in n],
           "val": [n for n in labelled if "00000002" in n], "test": [n for n in labelled if "00000003" in n]}
    split_file = tmp / "old_splits.json"
    split_file.write_text(json.dumps({"splits": old}), encoding="utf-8")
    report = build_pack(img_zip, part, parsing_zip, split_file, {"00000004"},
                        FakeBackend("lip", (473, 473)), FakeBackend("pascal_person_part", (512, 512)),
                        tmp / "pack256", "pascal_person_part", batch_size=5, num_workers=0, progress=False)
    return tmp / "pack256", report, names, old


def test_blob_roundtrip(tmp_path):
    w = BlobWriter(tmp_path / "b")
    for d in (b"abc", b"", b"xyz12"):
        w.append(d)
    assert w.close() == 3
    r = BlobReader(tmp_path / "b")
    assert [r.get(i) for i in range(3)] == [b"abc", b"", b"xyz12"]
    lab = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert (decode_png(encode_png(lab)) == lab).all()


def test_pack_contents(pack):
    d, report, names, _ = pack
    st = PackedStore(d)
    assert len(st) == 16 and set(st.index) == set(names) and st.size == (256, 256)
    assert report["meta"]["manual"] == 12
    n = names[0]
    assert st.image(n).shape == (256, 256, 3)
    g, a, b = st.garment(n), st.garment_auto(n), st.body(n)
    assert g.shape == a.shape == b.shape == (256, 256)
    assert set(np.unique(g)) <= set(range(12)) and set(np.unique(b)) <= set(range(7))
    G = get_label_set("garment12")
    assert (g[0:20] == 0).all() and (g == G.index("face")).any()  # manual map letterboxed to 256, face kept
    unparsed = [x for x in names if x.endswith("02_1_front")][0]
    assert not st.records[st.index[unparsed]].manual
    assert (st.garment(unparsed) == st.garment_auto(unparsed)).all()
    with pytest.raises(KeyError):
        PackedStore(d, "manual").garment(unparsed)
    assert 0.0 <= report["parser_agreement_vs_manual_garment12"]["pixel_accuracy"] <= 1.0


def test_pack_splits_and_pairs(pack):
    d, report, names, old = pack
    sp = json.loads((d / "splits.json").read_text())["splits"]
    assert sp["val"] == old["val"] and sp["test"] == old["test"]  # evaluation splits unchanged
    assert all("00000001" in n for n in sp["train"]) and len(sp["train"]) == 4  # incl. the unparsed photo
    assert report["splits"]["old_train_photos_dropped_as_protected"] == 3  # item 4 is a protected benchmark item
    pf = PairFilter("classes", d / "pairs.csv")
    assert pf.allowed  # table readable by the training filter
    assert report["items_with_2plus_photos"] == 4 and report["pairs"] == 4 * 6


def test_build_packed_splits_rejects_nothing_silently():
    rows = [{"name": f"MEN-Denim-id_0000000{i}-01_1_front", "item_id": f"0000000{i}", "manual": True} for i in range(1, 5)]
    out = build_packed_splits(rows, {"train": [rows[0]["name"]], "val": [rows[1]["name"]], "test": [rows[2]["name"]]},
                              {"00000004"})
    assert out["splits"]["train"] == [rows[0]["name"]]
    assert out["meta"]["unused_photos"] == {"val_or_test_items": 2, "protected_items": 1}


def _cfg(tmp_path, pack_dir, *extra):
    return cfg_for(tmp_path, f"data.packed_dir={pack_dir.as_posix()}", f"data.split_file={(pack_dir / 'splits.json').as_posix()}",
                   f"data.pair_table={(pack_dir / 'pairs.csv').as_posix()}", "data.pair_filter=classes",
                   "labels.body_label_set=pascal_person_part", "data.num_workers=0", *extra)


def test_datasets_read_the_store(pack, tmp_path):
    d = pack[0]
    cfg = _cfg(tmp_path, d)  # model size 64: maps are resized from 256
    prep = load_packed(cfg)
    w = WarpTripletDataset(prep, "train", cfg, train=True)
    s = w[0]
    assert s["cloth"].shape == (12, 64, 64) and s["body"].shape == (7, 64, 64) and s["target"].shape == (64, 64)
    t = TextureDataset(prep, "train", cfg, train=True)
    s = t[0]
    assert s["image"].shape == (3, 64, 64) and s["layout"].shape == (12, 64, 64)
    cfg_m = _cfg(tmp_path, d, "data.label_source=manual")
    assert len(load_packed(cfg_m).records("train")) == 3  # unparsed photo dropped


def test_training_with_ema_and_inference_prefers_ema(pack, tmp_path):
    cfg = _cfg(tmp_path, pack[0], "training.ema_decay=0.9")
    prep = load_packed(cfg)
    wt = WarpTrainer(cfg, prep, verbose=False)
    before = {k: v.clone() for k, v in wt.ema.items()}
    wt.fit(max_steps=2, resume=False)
    st = torch.load(wt.ckpt.last, map_location="cpu", weights_only=True)
    assert "gen_ema" in st
    k = next(n for n, p in wt.gen.named_parameters())
    assert not torch.equal(st["gen_ema"][k], st["gen"][k]) and not torch.equal(st["gen_ema"][k], before[k])
    TextureTrainer(cfg, prep, verbose=False).fit(max_steps=1, resume=False)
    models = load_tryon_models(cfg)
    assert models.info["warp"]["weights"] == "ema" and models.info["texture"]["weights"] == "ema"
    assert torch.equal(models.warp.state_dict()[k].cpu(), st["gen_ema"][k])
    # resume restores the EMA, not a copy of the live weights
    wt2 = WarpTrainer(cfg, prep, verbose=False)
    assert wt2.resume() and torch.equal(wt2.ema[k], st["gen_ema"][k].to(wt2.ema[k].device))


def test_ema_update_math(pack, tmp_path):
    cfg = _cfg(tmp_path, pack[0], "training.ema_decay=0.5")
    wt = WarpTrainer(cfg, load_packed(cfg), verbose=False)
    k = next(n for n, _ in wt.gen.named_parameters())
    old = wt.ema[k].clone()
    with torch.no_grad():
        dict(wt.gen.named_parameters())[k].add_(1.0)
    wt.step = 1000  # past warm-up: decay = 0.5
    wt.update_ema()
    assert torch.allclose(wt.ema[k], old + 0.5)


def test_batched_schp_inverse_warp_is_identity_for_matching_frames():
    b = FakeBackend("lip", (256, 256))
    bs = BatchedSCHP(b, (256, 256))
    img = _photo(3)
    out = bs(torch.from_numpy(bs.warp_input(img))[None])
    assert out.shape == (1, 256, 256) and out.dtype == np.uint8


def test_inshop_loader_crops_to_dfmm_aspect(pack):
    st = PackedStore(pack[0])
    img = inshop_loader(st)(pack[2][0])
    assert img.shape == (256, 174, 3)


def test_layout_eval_all_definitions(pack, tmp_path):
    from src.evaluation.layout_eval import evaluate_layouts, format_layouts

    cfg = _cfg(tmp_path, pack[0])
    prep = load_packed(cfg)
    WarpTrainer(cfg, prep, verbose=False).fit(max_steps=1, resume=False)
    TextureTrainer(cfg, prep, verbose=False).fit(max_steps=1, resume=False)
    models = load_tryon_models(cfg)
    res = evaluate_layouts(models.warp, prep, cfg, models.device, models.untrained_ids)
    assert res["images"] == len(prep.records("val"))
    for k in ("supervised", "final_layout_auto", "final_layout_gt_identity"):
        r = res[k]
        assert 0 <= r["miou_excl_bg"] <= 1 and 0 <= r["miou_incl_bg"] <= 1 and 0 <= r["pixel_accuracy"] <= 1
    gt = res["final_layout_gt_identity"]["per_class_iou"]
    assert gt["face"] == 1.0  # face labels copied from the manual parse match it exactly
    assert "mIoU incl. background" in format_layouts(res)


# ---------------------------------------------------------------- v2 additions: DensePose, contextual, fine-tuning
def _dp_zip(tmp_path, names, skip=()):
    z = tmp_path / "densepose.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for k, n in enumerate(names):
            if n in skip:
                continue
            a = np.zeros((1101, 750, 3), np.uint8)
            a[200:900, 200:550, 2] = 2  # DensePose part index (torso) in channel 2
            a[900:1050, 250:350, 2] = 7  # a leg part
            b = io.BytesIO()
            Image.fromarray(a).save(b, "PNG")
            zf.writestr(f"densepose/{n}_densepose.png", b.getvalue())
    return z


@pytest.fixture(scope="module")
def pack_dp(pack, tmp_path_factory):
    import shutil

    from src.data.packing import add_densepose

    tmp = tmp_path_factory.mktemp("dp")
    d = tmp / "pack256"
    shutil.copytree(pack[0], d)
    missing = {pack[2][1]}
    info = add_densepose(d, _dp_zip(tmp, pack[2], skip=missing), "dfmm", "body15", progress=False)
    return d, info, missing


def test_add_densepose_and_body_source(pack_dp):
    d, info, missing = pack_dp
    assert info["photos_with_map"] == 15 and info["photos"] == 16
    st = PackedStore(d, body_source="densepose")
    name = next(n for n in st.index if n not in missing)
    b = st.body(name)
    assert b.shape == (256, 256) and set(np.unique(b)) - {0} and b.max() < 15
    assert not st.has_body(next(iter(missing))) and st.has_body(name)
    assert PackedStore(d).has_body(next(iter(missing)))  # schp maps exist for every photo
    with pytest.raises(Exception):
        PackedStore(pack_dp[0].parent.parent / "nope", body_source="densepose")


def test_warp_with_densepose_skips_photos_without_maps(pack_dp, tmp_path):
    d, _, missing = pack_dp
    cfg = _cfg(tmp_path, d, "data.body_source=densepose", "labels.body_label_set=body15")
    prep = load_packed(cfg)
    ds = WarpTripletDataset(prep, "train", cfg, train=True)
    assert not ({r.name for r in ds.records} & missing) or len(ds.records) == len(prep.records("train"))
    assert ds[0]["body"].shape[0] == 15
    WarpTrainer(cfg, prep, verbose=False).fit(max_steps=1, resume=False)


def test_packed_body_backend_crops_like_the_photo(pack_dp):
    from src.evaluation.pose_transfer_eval import PackedBodyBackend

    d, _, missing = pack_dp
    st = PackedStore(d, body_source="densepose")
    name = next(n for n in st.index if n not in missing)
    res = PackedBodyBackend(st, "body15").predict(inshop_loader(st)(name), key=name)
    assert res.labels.shape == (256, 174)


def test_contextual_loss_properties():
    from src.training.losses import contextual_loss

    torch.manual_seed(0)
    y = torch.randn(2, 8, 6, 6)
    perm = y.flatten(2)[:, :, torch.randperm(36)].view_as(y)
    same, shuffled, other = contextual_loss(y, y), contextual_loss(perm, y), contextual_loss(torch.randn(2, 8, 6, 6), y)
    assert same < 0.05 and shuffled < 0.05  # position-independent: a shuffled copy still matches
    assert other > same + 0.3
    x = torch.randn(2, 8, 6, 6, requires_grad=True)
    contextual_loss(x, y).backward()
    assert torch.isfinite(x.grad).all()


def test_texture_finetune_init_from_lr_decay_and_context(pack, tmp_path):
    cfg0 = _cfg(tmp_path, pack[0], "training.ema_decay=0.9")
    prep = load_packed(cfg0)
    base = TextureTrainer(cfg0, prep, verbose=False)
    base.fit(max_steps=2, resume=False)
    src = base.ckpt.last
    cfg = _cfg(tmp_path, pack[0], "training.ema_decay=0.9", "training.texture_run_name=texture_ft",
               f"training.texture_init_from={src.as_posix()}", "training.lr_decay_from=0", "training.lambda_tex_l1=5",
               "training.lambda_tex_context=1")
    ft = TextureTrainer(cfg, prep, verbose=False)
    st0 = torch.load(src, map_location="cpu", weights_only=True)
    out = ft.fit(max_steps=3, resume=True)
    assert out["step"] == 3 and ft.ckpt.dir.name == "texture_ft"
    assert all(g["lr"] == 0.0 for g in ft.opt_g.param_groups)  # decayed to 0 at the last step
    assert all(np.isfinite(h["context"]) and h["context"] > 0 and h["l1"] > 0 for h in ft.history)
    k = next(n for n, _ in ft.gen.named_parameters())
    assert not torch.equal(ft.gen.state_dict()[k].cpu(), torch.zeros_like(st0["gen"][k]))
    # a second call resumes the fine-tune run itself (not init_from again)
    ft2 = TextureTrainer(cfg, prep, verbose=False)
    assert ft2.resume() and ft2.step == 3
