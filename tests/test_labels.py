import numpy as np
import pytest

from src.data import labels as L


def test_label_sets_match_verified_sources():
    # Lists copied from SCHP simple_extractor.py and the DFMM README (see module docstring).
    assert L.get_label_set("lip").num_classes == 20
    assert L.get_label_set("atr").num_classes == 18
    assert L.get_label_set("pascal_person_part").num_classes == 7
    assert L.get_label_set("dfmm").num_classes == 24
    assert L.get_label_set("densepose").num_classes == 25
    assert L.get_label_set("dfmm").names[21] == "rompers"
    assert L.get_label_set("lip").names[7] == "coat"


@pytest.mark.parametrize("src,dst", L.available_conversions())
def test_every_source_class_is_mapped(src, dst):
    s = L.get_label_set(src)
    out = L.convert_labels(np.arange(s.num_classes, dtype=np.uint8)[None, :], src, dst)
    assert out.shape == (1, s.num_classes)
    assert out.max() < L.get_label_set(dst).num_classes
    assert out[0, 0] == 0  # background -> background


def test_specific_mappings():
    lip, g = L.get_label_set("lip"), L.get_label_set("garment12")
    m = np.array([[lip.index("coat"), lip.index("left_shoe"), lip.index("right_arm"), lip.index("jumpsuits")]], np.uint8)
    assert L.convert_labels(m, "lip", "garment12").tolist() == [[g.index("outer"), g.index("footwear"), g.index("skin"), g.index("dress")]]
    dp, b = L.get_label_set("densepose"), L.get_label_set("body15")
    m = np.array([[dp.index("torso_1"), dp.index("torso_2"), dp.index("head_2"), dp.index("lower_arm_left_2")]], np.uint8)
    assert L.convert_labels(m, "densepose", "body15").tolist() == [[b.index("torso")] * 2 + [b.index("head"), b.index("left_lower_arm")]]
    # DensePose -> Pascal-Person-Part composes through body15; hands belong to lower arms
    ppp = L.get_label_set("pascal_person_part")
    assert L.convert_labels(np.array([[dp.index("right_hand")]], np.uint8), "densepose", "pascal_person_part")[0, 0] == ppp.index("lower_arms")


def test_invalid_maps_rejected():
    with pytest.raises(L.LabelError, match="outside"):
        L.convert_labels(np.array([[0, 30]], np.uint8), "lip", "garment12")
    with pytest.raises(L.LabelError, match="colour-coded"):
        L.check_label_map(np.zeros((4, 4, 3), np.uint8), L.get_label_set("lip"))
    with pytest.raises(KeyError):
        L.convert_labels(np.zeros((2, 2), np.uint8), "garment12", "lip")


def test_colorize_decode_roundtrip():
    g = L.get_label_set("garment12")
    m = np.random.default_rng(0).integers(0, g.num_classes, (16, 12)).astype(np.uint8)
    rgb = L.colorize(m, g)
    assert rgb.shape == (16, 12, 3)
    np.testing.assert_array_equal(L.decode_color_map(rgb, g), m)
    rgb[0, 0] = (1, 2, 3)
    with pytest.raises(L.LabelError):
        L.decode_color_map(rgb, g)


def test_one_hot_and_keep():
    g = L.get_label_set("garment12")
    m = np.array([[0, g.index("top")], [g.index("face"), g.index("pants")]], np.uint8)
    oh = L.one_hot(m, g.num_classes)
    assert oh.shape == (12, 2, 2) and oh.sum(0).min() == 1
    kept = L.keep_classes(m, g, ["top", "pants"])
    assert kept.tolist() == [[0, g.index("top")], [0, g.index("pants")]]


def test_inject_identity_labels():
    g = L.get_label_set("garment12")
    F, H, T = g.index("face"), g.index("hair"), g.index("top")
    garment = np.array([[F, T], [T, T]], np.uint8)  # reference face must disappear
    target = np.array([[0, H], [0, 0]], np.uint8)
    out = L.inject_identity_labels(garment, target, g, ["hair", "face"])
    assert out.tolist() == [[0, H], [T, T]]


def test_palettes_have_unique_colours_for_project_sets():
    for name in ("garment12", "body15"):
        pal = L.get_label_set(name).palette
        assert len(set(pal)) == len(pal)
