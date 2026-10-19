import numpy as np
import pytest
import torch
from PIL import Image

from src.data import preprocessing as P
from src.data.labels import get_label_set
from src.data.transforms import AffineParams, apply_joint_affine, left_right_swap_lut


def test_load_image_variants(tmp_path):
    rgba = np.zeros((5, 6, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[0, 0, 3] = 0  # transparent pixel -> white
    rgba[1:, :, 3] = 255
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    img = P.load_image(tmp_path / "a.png")
    assert img.shape == (5, 6, 3) and img.dtype == np.uint8
    assert tuple(img[0, 0]) == (255, 255, 255)
    Image.fromarray(np.full((4, 4), 7, np.uint8), "L").save(tmp_path / "g.png")
    assert P.load_image(tmp_path / "g.png").shape == (4, 4, 3)
    with pytest.raises(P.ImageLoadError, match="not found"):
        P.load_image(tmp_path / "missing.jpg")
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    with pytest.raises(P.ImageLoadError, match="Cannot decode"):
        P.load_image(tmp_path / "bad.jpg")


def test_load_label_map_modes(tmp_path):
    lab = np.array([[0, 3], [5, 19]], np.uint8)
    p = Image.fromarray(lab, "L").convert("P")
    p.putpalette([255, 0, 0] * 256)  # display palette must be ignored
    p.save(tmp_path / "p.png")
    np.testing.assert_array_equal(P.load_label_map(tmp_path / "p.png"), lab)
    Image.fromarray(np.stack([lab * 0, lab * 0 + 9, lab], 2)).save(tmp_path / "rgb.png")
    with pytest.raises(P.ImageLoadError, match="multi-channel"):
        P.load_label_map(tmp_path / "rgb.png")
    np.testing.assert_array_equal(P.load_label_map(tmp_path / "rgb.png", channel=2), lab)


def test_letterbox_geometry_and_inverse():
    lab = np.zeros((1101, 750), np.uint8)
    lab[100:200, 300:400] = 4
    out, info = P.resize_labels(lab, (256, 256), "letterbox")
    assert out.shape == (256, 256)
    assert info.content_size == (256, 174) and info.pad_left == 41 and info.pad_top == 0
    assert set(np.unique(out)) == {0, 4}  # nearest neighbour never invents classes
    back = P.undo_resize_labels(out, info)
    assert back.shape == lab.shape
    assert (back == lab).mean() > 0.99


def test_resize_image_pads_white_and_stretch():
    img = np.zeros((40, 20, 3), np.uint8)
    out, info = P.resize_image(img, (32, 32), "letterbox", 255)
    assert out.shape == (32, 32, 3) and out[0, 0].tolist() == [255, 255, 255] and out[16, 16].tolist() == [0, 0, 0]
    out2, info2 = P.resize_image(img, (32, 32), "stretch")
    assert info2.pad_left == 0 and out2.max() == 0


def test_tensor_roundtrip():
    img = np.random.default_rng(0).integers(0, 256, (8, 6, 3)).astype(np.uint8)
    t = P.image_to_tensor(img)
    assert t.shape == (3, 8, 6) and t.dtype == torch.float32 and t.min() >= -1 and t.max() <= 1
    np.testing.assert_array_equal(P.tensor_to_image(t), img)
    assert P.labels_to_tensor(np.zeros((3, 3), np.uint8)).dtype == torch.int64
    with pytest.raises(ValueError):
        P.image_to_tensor(img.astype(np.float32))


def test_joint_affine_keeps_alignment():
    img = np.full((40, 40, 3), 255, np.uint8)
    img[10:20, 5:15] = 0
    lab = np.zeros((40, 40), np.uint8)
    lab[10:20, 5:15] = 1
    out_img, out_lab = apply_joint_affine(img, {"seg": lab}, AffineParams(translate=(0.25, 0.0)))
    ys, xs = np.nonzero(out_lab["seg"])
    assert xs.min() == 15 and xs.max() == 24  # moved right by 10 px
    assert (out_img[out_lab["seg"] == 1] == 0).all()  # image and labels still aligned
    ident_img, ident = apply_joint_affine(img, {"seg": lab}, AffineParams())
    np.testing.assert_array_equal(ident["seg"], lab)


def test_flip_swaps_left_right():
    b = get_label_set("body15")
    lut = left_right_swap_lut(b.names)
    assert lut[b.index("left_hand")] == b.index("right_hand") and lut[b.index("torso")] == b.index("torso")
    lab = np.zeros((10, 10), np.uint8)
    lab[:, :5] = b.index("right_upper_arm")
    _, out = apply_joint_affine(np.zeros((10, 10, 3), np.uint8), {"body": lab}, AffineParams(hflip=True), swap_lr={"body": lut})
    assert (out["body"][:, 5:] == b.index("left_upper_arm")).all()
