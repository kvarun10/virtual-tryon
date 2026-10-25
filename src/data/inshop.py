"""DeepFashion In-shop (official release): read its annotation lists and compare it with our DFMM data.

Only text annotations are read (no images), so this runs before any large download. The README of the release
defines the formats used here:

* ``Eval/list_eval_partition.txt``: count, header, then ``<image name> <item id> <train|query|gallery>``
* ``Anno/list_bbox_inshop.txt``: count, header, then ``<image name> <clothes type> <pose type> <x1 y1 x2 y2>``
  clothes type 1 upper / 2 lower / 3 full body; pose type 1 frontal, 2 side, 3 back, 4 zoom-out, 5 zoom-in,
  6 stand-alone (garment without a person)
* ``Anno/segmentation/DeepFashion_instances_*.json``: COCO-style clothing masks (mmfashion), optional

An In-shop image ``img/MEN/Denim/id_00000080/01_7_additional.jpg`` is the DFMM image
``MEN-Denim-id_00000080-01_7_additional`` (same photo; DFMM keeps the In-shop path in its file name).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

POSE_TYPES = {1: "frontal", 2: "side", 3: "back", 4: "zoom-out", 5: "zoom-in", 6: "stand-alone"}
CLOTHES_TYPES = {1: "upper", 2: "lower", 3: "full"}
PERSON_POSES = {1, 2, 3, 4}  # a person is visible and mostly in view; 5 = close-up, 6 = garment only


def _rows(path: Path) -> List[List[str]]:
    lines = [l.split() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[2:]  # first row: count, second row: header


def read_eval_partition(path: Path) -> List[Tuple[str, str, str]]:
    return [(r[0], r[1], r[2]) for r in _rows(path)]


def read_bbox(path: Path) -> Dict[str, Tuple[int, int]]:
    return {r[0]: (int(r[1]), int(r[2])) for r in _rows(path)}


def inshop_to_dfmm_stem(image_name: str) -> str:
    """``img/MEN/Denim/id_00000080/01_7_additional.jpg`` -> ``MEN-Denim-id_00000080-01_7_additional``."""
    parts = image_name.replace("\\", "/").split("/")
    if parts and parts[0] in ("img", "img_highres"):
        parts = parts[1:]
    if len(parts) != 4:
        raise ValueError(f"unexpected In-shop image name: {image_name}")
    gender, category, item, file = parts
    return f"{gender}-{category}-{item}-{Path(file).stem}"


def item_number(item_id: str) -> str:
    """``id_00000080`` -> ``00000080`` (the form our DFMM records use)."""
    return item_id.split("_")[-1]


def _seg_stem(file_name: str) -> Optional[str]:
    """Map a segmentation JSON file name to a DFMM-style stem; None if the format is not recognised."""
    f = file_name.replace("\\", "/")
    parts = f.split("/")
    try:
        if len(parts) >= 4:
            return inshop_to_dfmm_stem("/".join(parts[-4:]))
    except ValueError:
        pass
    stem = Path(f).stem
    return stem if stem.count("-") >= 3 else None


def segmentation_coverage(seg_dir: Path, labelled: Optional[set] = None, dfmm_all: Optional[set] = None,
                          protected: Optional[set] = None) -> Dict:
    """Per JSON file: images, masks per category and, if name sets are given, how many images are NOT already
    manually parsed in our DFMM data (i.e. would be new labelled training images) and not of a val/test item."""
    out = {}
    for f in sorted(Path(seg_dir).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        per_cat = Counter(cats.get(a.get("category_id"), str(a.get("category_id"))) for a in d.get("annotations", []))
        img_cats: Dict[int, set] = defaultdict(set)
        for a in d.get("annotations", []):
            img_cats[a.get("image_id")].add(cats.get(a.get("category_id")))
        images = d.get("images", [])
        info = {"images": len(images), "annotations": len(d.get("annotations", [])), "categories": dict(per_cat),
                "example_file_names": [im.get("file_name") for im in images[:3]]}
        if labelled is not None:
            c = Counter()
            for im in images:
                stem = _seg_stem(str(im.get("file_name", "")))
                if stem is None:
                    c["name_not_recognised"] += 1
                    continue
                num = stem.split("-")[2].split("_")[-1] if stem.count("-") >= 3 else ""
                if stem in labelled:
                    c["already_manually_parsed_in_dfmm"] += 1
                elif protected and num in protected:
                    c["of_val_test_items_EXCLUDE"] += 1
                else:
                    c["new_labelled_candidates"] += 1
                    if {"skin", "face", "hair"} & img_cats.get(im.get("id"), set()):
                        c["new_with_skin_face_hair_labels"] += 1
                    if dfmm_all is not None and stem in dfmm_all:
                        c["new_but_image_in_dfmm"] += 1
            info["overlap"] = dict(c)
        out[f.name] = info
    return out


def inspect_inshop(partition_file: Path, bbox_file: Path, split_file: Path,
                   dfmm_names: Optional[Iterable[str]] = None, segmentation_dir: Optional[Path] = None) -> Dict:
    """Compare In-shop with our data.

    ``split_file``: our item-disjoint DFMM splits (train/val/test image names). ``dfmm_names``: names of ALL DFMM
    images (44k; optional, e.g. the file stems of the DFMM image folder); without it, overlap is measured only
    against the 12.7k annotated images in our splits.
    """
    part = read_eval_partition(partition_file)
    bbox = read_bbox(bbox_file)
    splits = json.loads(Path(split_file).read_text(encoding="utf-8"))["splits"]
    labelled = {n: s for s, names in splits.items() for n in names}
    dfmm_all = set(dfmm_names) if dfmm_names is not None else set(labelled)
    # items of our validation/test splits: never train on any photo of them (leakage)
    protected = {n.split("-")[2].split("_")[-1] for s in ("val", "test") for n in splits.get(s, [])}

    rep: Dict = {"inshop_images": len(part), "inshop_items": len({i for _, i, _ in part}),
                 "partition": dict(Counter(s for _, _, s in part)),
                 "pose_types": {POSE_TYPES.get(k, str(k)): v for k, v in sorted(Counter(p for _, p in bbox.values()).items())},
                 "clothes_types": {CLOTHES_TYPES.get(k, str(k)): v for k, v in sorted(Counter(c for c, _ in bbox.values()).items())},
                 "dfmm_reference": "all DFMM images" if dfmm_names is not None else "only our 12.7k annotated split images"}
    in_dfmm = in_labelled = 0
    leak_imgs = 0
    new_person: Dict[str, List[str]] = defaultdict(list)
    new_by_pose = Counter()
    unmatched_bbox = 0
    for img, item, _ in part:
        stem = inshop_to_dfmm_stem(img)
        num = item_number(item)
        in_dfmm += stem in dfmm_all
        in_labelled += stem in labelled
        if num in protected:
            leak_imgs += 1
            continue
        if stem in dfmm_all:
            continue
        if img not in bbox:
            unmatched_bbox += 1
            continue
        _, pose = bbox[img]
        new_by_pose[POSE_TYPES.get(pose, str(pose))] += 1
        if pose in PERSON_POSES:
            new_person[num].append(stem)
    multi = {k: v for k, v in new_person.items() if len(v) >= 2}
    rep.update({
        "also_in_dfmm": in_dfmm, "also_in_our_labelled_splits": in_labelled,
        "images_of_our_val_test_items_EXCLUDE": leak_imgs, "protected_items": len(protected),
        "new_images_by_pose": dict(new_by_pose), "new_person_images": sum(len(v) for v in new_person.values()),
        "new_items_with_2plus_person_images": len(multi),
        "new_same_item_pairs_upper_bound": sum(len(v) * (len(v) - 1) for v in multi.values()),
        "partition_images_without_bbox": unmatched_bbox,
        "note": ("New images have no manual parsing: layouts would come from SCHP-LIP (pseudo-labels). Same-item pairs "
                 "still need the outfit filter (dE <= 15), so the usable pair count will be lower than the upper bound."),
    })
    if segmentation_dir is not None and Path(segmentation_dir).is_dir():
        rep["segmentation_json"] = segmentation_coverage(Path(segmentation_dir), set(labelled), dfmm_all, protected)
    return rep
