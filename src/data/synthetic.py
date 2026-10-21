"""SYNTHETIC test fixture that mimics the DeepFashion-MultiModal file layout.

!!! These images are procedurally drawn cartoons, NOT data. They exist only to test file
!!! handling, label conversion, shapes and the pipeline plumbing without downloading anything.
!!! Never report metrics computed on them.

Each fake "item" (garment) is drawn on a stick-figure person in several "views" (small shifts /
scales), producing for every image:
* ``images/<name>.jpg``            RGB, white background
* ``segm/<name>_segm.png``         DFMM 24-class label ids (only for "full-body" images)
* ``densepose/<name>_densepose.png`` RGB PNG, part index I in channel 2 (as DFMM per Text2Human)
* ``keypoints/keypoints_loc.txt`` / ``keypoints_vis.txt`` placeholder rows (all -1 / 2)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

from src.data.labels import get_label_set

MARKER_FILE = "SYNTHETIC_FIXTURE_NOT_REAL_DATA.txt"

_DFMM = get_label_set("dfmm")
_DP = get_label_set("densepose")
S = _DFMM.index
D = _DP.index

SKIN = (228, 182, 150)
HAIR = (70, 45, 25)
SHOE = (40, 40, 40)


@dataclass
class FixtureSpec:
    n_items: int = 6
    views_per_item: int = 3
    width: int = 150  # DFMM aspect ratio 750 x 1101 ~ 150 x 220
    height: int = 220
    parsing_fraction: float = 0.75  # share of images that get a manual-parsing file
    seed: int = 0
    corrupt_images: int = 0  # write N undecodable JPGs (validation tests)
    drop_densepose: int = 0  # delete DensePose for N images (missing-file tests)
    orphan_parsing: int = 0  # write N parsing files without an image
    outfit_change_items: int = 0  # first N items: views after the first wear a DIFFERENT outfit (DeepFashion-like noise)


def _rect(draws, box, seg_label, dp_label, color):
    """Paint one box into the RGB, parsing and DensePose canvases (dp_label=None: leave DensePose)."""
    img_d, seg_d, dp_d = draws
    img_d.rectangle(box, fill=color)
    seg_d.rectangle(box, fill=seg_label)
    if dp_label is not None:
        dp_d.rectangle(box, fill=dp_label)


def _draw_person(size, x0: float, scale: float, garment: Dict, rng: np.random.Generator):
    w, h = size
    img = Image.new("RGB", size, (255, 255, 255))
    seg = Image.new("L", size, 0)
    dpi = Image.new("L", size, 0)
    draws = (ImageDraw.Draw(img), ImageDraw.Draw(seg), ImageDraw.Draw(dpi))

    def X(v):
        return x0 + v * scale

    def Y(v):
        return 8 + v * scale

    top_c, low_c = garment["top_color"], garment["lower_color"]
    long_sleeve = garment["long_sleeve"]
    lower = garment["lower"]  # pants | skirt | dress

    # legs (drawn first so the garment can cover them); image-left = person's right
    for side, sx in (("right", -1), ("left", 1)):
        x_in, x_out = X(sx * 2), X(sx * 22)
        box_u = (min(x_in, x_out), Y(112), max(x_in, x_out), Y(150))
        box_l = (min(x_in, x_out), Y(150), max(x_in, x_out), Y(188))
        leg_label = S("pants") if lower == "pants" else S("skin")
        leg_color = low_c if lower == "pants" else SKIN
        _rect(draws, box_u, leg_label, D(f"upper_leg_{side}_1"), leg_color)
        _rect(draws, box_l, leg_label if lower == "pants" else S("skin"), D(f"lower_leg_{side}_1"), leg_color if lower == "pants" else SKIN)
        foot = (min(X(sx * 2), X(sx * 24)), Y(188), max(X(sx * 2), X(sx * 24)), Y(198))
        _rect(draws, foot, S("footwear"), D(f"{side}_foot"), SHOE)
    # arms
    for side, sx in (("right", -1), ("left", 1)):
        up = (min(X(sx * 26), X(sx * 36)), Y(44), max(X(sx * 26), X(sx * 36)), Y(82))
        lo = (min(X(sx * 26), X(sx * 36)), Y(82), max(X(sx * 26), X(sx * 36)), Y(116))
        hand = (min(X(sx * 27), X(sx * 35)), Y(116), max(X(sx * 27), X(sx * 35)), Y(126))
        _rect(draws, up, S("top"), D(f"upper_arm_{side}_1"), top_c)
        _rect(draws, lo, S("top") if long_sleeve else S("skin"), D(f"lower_arm_{side}_1"), top_c if long_sleeve else SKIN)
        _rect(draws, hand, S("skin"), D(f"{side}_hand"), SKIN)
    # torso + lower garment
    # DensePose describes the body surface, so garments covering the legs do not change it.
    if lower == "dress":
        _rect(draws, (X(-25), Y(42), X(25), Y(114)), S("dress"), D("torso_1"), top_c)
        _rect(draws, (X(-26), Y(114), X(26), Y(140)), S("dress"), None, top_c)
    else:
        _rect(draws, (X(-25), Y(42), X(25), Y(114)), S("top"), D("torso_1"), top_c)
        if lower == "skirt":
            _rect(draws, (X(-26), Y(114), X(26), Y(140)), S("skirt"), None, low_c)
    # stripes as a simple texture
    if garment["stripes"]:
        stripe = tuple(int(c * 0.6) for c in top_c)
        for yy in range(int(Y(48)), int(Y(110)), max(3, int(8 * scale))):
            draws[0].line((X(-24), yy, X(24), yy), fill=stripe, width=max(1, int(2 * scale)))
    # neck, head
    _rect(draws, (X(-6), Y(34), X(6), Y(43)), S("skin"), D("torso_1"), SKIN)
    hair_box = (X(-15), Y(0), X(15), Y(26))
    draws[0].ellipse(hair_box, fill=HAIR)
    draws[1].ellipse(hair_box, fill=S("hair"))
    draws[2].ellipse(hair_box, fill=D("head_1"))
    face_box = (X(-11), Y(6), X(11), Y(36))
    draws[0].ellipse(face_box, fill=SKIN)
    draws[1].ellipse(face_box, fill=S("face"))
    draws[2].ellipse(face_box, fill=D("head_2"))
    noise = rng.integers(-6, 7, size=(h, w, 3))
    rgb = np.clip(np.asarray(img).astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return rgb, np.asarray(seg), np.asarray(dpi)


def make_fixture(root: Path, spec: Optional[FixtureSpec] = None) -> Dict:
    """Write the fixture to ``root`` and return a manifest (also saved as JSON)."""
    spec = spec or FixtureSpec()
    root = Path(root)
    for sub in ("images", "segm", "densepose", "keypoints"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / MARKER_FILE).write_text(
        "This directory was generated by src/data/synthetic.py. It contains drawn cartoons, not real "
        "data, and exists only for software tests. Do not report metrics on it.\n",
        encoding="utf-8",
    )
    rng = np.random.default_rng(spec.seed)
    categories = [("WOMEN", "Synthetic_Tees"), ("MEN", "Synthetic_Shirts"), ("WOMEN", "Synthetic_Dresses")]
    names: List[str] = []
    kp_loc, kp_vis = [], []
    for item in range(spec.n_items):
        gender, cat = categories[item % len(categories)]
        garment = {
            "top_color": tuple(int(v) for v in rng.integers(30, 230, 3)),
            "lower_color": tuple(int(v) for v in rng.integers(30, 230, 3)),
            "long_sleeve": bool(rng.random() < 0.5),
            "lower": "dress" if "Dresses" in cat else ("skirt" if rng.random() < 0.3 else "pants"),
            "stripes": bool(rng.random() < 0.5),
        }
        for v in range(spec.views_per_item):
            view = ["1_front", "2_side", "4_full", "7_additional"][v % 4]
            name = f"{gender}-{cat}-id_{90000000 + item:08d}-01_{view}"
            x0 = spec.width / 2 + rng.uniform(-12, 12)
            scale = rng.uniform(0.9, 1.0)
            outfit = garment
            if item < spec.outfit_change_items and v > 0:  # same item ID, other clothes (e.g. the item is an accessory)
                outfit = {**garment, "top_color": tuple(255 - c for c in garment["top_color"]),
                          "lower": "skirt" if garment["lower"] == "pants" else "pants"}
            rgb, seg, dpi = _draw_person((spec.width, spec.height), x0, scale, outfit, rng)
            Image.fromarray(rgb).save(root / "images" / f"{name}.jpg", quality=92)
            if rng.random() < spec.parsing_fraction:
                Image.fromarray(seg).save(root / "segm" / f"{name}_segm.png")
            uv = np.where(dpi > 0, 128, 0).astype(np.uint8)
            Image.fromarray(np.stack([uv, uv, dpi], axis=2)).save(root / "densepose" / f"{name}_densepose.png")
            kp_loc.append(name + ".jpg " + " ".join(["-1 -1"] * 21))
            kp_vis.append(name + ".jpg " + " ".join(["2"] * 21))
            names.append(name)
    (root / "keypoints" / "keypoints_loc.txt").write_text("\n".join(kp_loc) + "\n", encoding="utf-8")
    (root / "keypoints" / "keypoints_vis.txt").write_text("\n".join(kp_vis) + "\n", encoding="utf-8")

    faults = {"corrupt": [], "missing_densepose": [], "orphan_parsing": []}
    for name in names[: spec.corrupt_images]:
        (root / "images" / f"{name}.jpg").write_bytes(b"\xff\xd8\xff\xe0 truncated synthetic jpeg")
        faults["corrupt"].append(name)
    for name in names[spec.corrupt_images : spec.corrupt_images + spec.drop_densepose]:
        (root / "densepose" / f"{name}_densepose.png").unlink()
        faults["missing_densepose"].append(name)
    for k in range(spec.orphan_parsing):
        orphan = f"WOMEN-Synthetic_Orphans-id_{99999900 + k:08d}-01_1_front"
        Image.fromarray(np.zeros((spec.height, spec.width), np.uint8)).save(root / "segm" / f"{orphan}_segm.png")
        faults["orphan_parsing"].append(orphan)
    manifest = {"synthetic": True, "spec": spec.__dict__, "names": names, "faults": faults}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def is_synthetic_root(root: Path) -> bool:
    return (Path(root) / MARKER_FILE).is_file()
