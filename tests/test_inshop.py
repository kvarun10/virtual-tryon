"""In-shop annotation reader and overlap/leakage report on a tiny hand-made fixture."""

import json

from scripts.inspect_inshop import main
from src.data.inshop import inshop_to_dfmm_stem, inspect_inshop


def _fixture(tmp_path):
    imgs = [  # (in-shop name, item, status, clothes, pose)
        ("img/MEN/Denim/id_00000080/01_1_front.jpg", "id_00000080", "train", 2, 1),   # in our train split
        ("img/MEN/Denim/id_00000080/01_2_side.jpg", "id_00000080", "train", 2, 2),    # DFMM only (not labelled)
        ("img/WOMEN/Dresses/id_00000002/02_1_front.jpg", "id_00000002", "query", 3, 1),  # item in our TEST split
        ("img/WOMEN/Dresses/id_00000002/02_7_additional.jpg", "id_00000002", "gallery", 3, 5),
        ("img/WOMEN/Tees_Tanks/id_00000500/01_1_front.jpg", "id_00000500", "train", 1, 1),  # new item
        ("img/WOMEN/Tees_Tanks/id_00000500/01_3_back.jpg", "id_00000500", "train", 1, 3),
        ("img/WOMEN/Tees_Tanks/id_00000500/01_6_flat.jpg", "id_00000500", "train", 1, 6),  # garment only
        ("img/MEN/Shirts_Polos/id_00000600/01_5_full.jpg", "id_00000600", "gallery", 1, 4),  # new, single image
    ]
    (tmp_path / "Eval").mkdir(); (tmp_path / "Anno").mkdir()
    (tmp_path / "Eval" / "list_eval_partition.txt").write_text(
        f"{len(imgs)}\nimage_name item_id evaluation_status\n" + "\n".join(f"{a} {b} {c}" for a, b, c, _, _ in imgs))
    (tmp_path / "Anno" / "list_bbox_inshop.txt").write_text(
        f"{len(imgs)}\nimage_name clothes_type pose_type x_1 y_1 x_2 y_2\n"
        + "\n".join(f"{a} {d} {e} 1 2 3 4" for a, _, _, d, e in imgs))
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"meta": {}, "splits": {"train": ["MEN-Denim-id_00000080-01_1_front"], "val": [],
                                                        "test": ["WOMEN-Dresses-id_00000002-02_1_front"]}}))
    seg = tmp_path / "Anno" / "segmentation"; seg.mkdir()
    (seg / "DeepFashion_instances_train.json").write_text(json.dumps(
        {"images": [{"id": 1, "file_name": "img/MEN/Denim/id_00000080/01_1_front.jpg"},
                    {"id": 2, "file_name": "WOMEN/Tees_Tanks/id_00000500/01_1_front.jpg"},
                    {"id": 3, "file_name": "img/WOMEN/Dresses/id_00000002/02_7_additional.jpg"},
                    {"id": 4, "file_name": "weird.jpg"}],
         "annotations": [{"category_id": 1, "image_id": 1}, {"category_id": 2, "image_id": 2},
                         {"category_id": 3, "image_id": 2}],
         "categories": [{"id": 1, "name": "top"}, {"id": 2, "name": "trousers"}, {"id": 3, "name": "skin"}]}))
    dfmm = ["MEN-Denim-id_00000080-01_1_front", "MEN-Denim-id_00000080-01_2_side", "WOMEN-Dresses-id_00000002-02_1_front"]
    return split, dfmm


def test_name_mapping():
    assert inshop_to_dfmm_stem("img/MEN/Denim/id_00000080/01_7_additional.jpg") == "MEN-Denim-id_00000080-01_7_additional"


def test_inspect_counts_and_leakage(tmp_path):
    split, dfmm = _fixture(tmp_path)
    r = inspect_inshop(tmp_path / "Eval" / "list_eval_partition.txt", tmp_path / "Anno" / "list_bbox_inshop.txt",
                       split, dfmm, tmp_path / "Anno" / "segmentation")
    assert r["inshop_images"] == 8 and r["inshop_items"] == 4
    assert r["also_in_dfmm"] == 3 and r["also_in_our_labelled_splits"] == 2
    assert r["images_of_our_val_test_items_EXCLUDE"] == 2  # both photos of the test item, even the one not in DFMM
    assert r["new_person_images"] == 3  # 2 of item 500 (front, back) + 1 of item 600; flat lay excluded
    assert r["new_items_with_2plus_person_images"] == 1 and r["new_same_item_pairs_upper_bound"] == 2
    assert r["new_images_by_pose"]["stand-alone"] == 1
    seg = r["segmentation_json"]["DeepFashion_instances_train.json"]
    assert seg["categories"] == {"top": 1, "trousers": 1, "skin": 1}
    assert seg["overlap"] == {"already_manually_parsed_in_dfmm": 1, "new_labelled_candidates": 1,
                              "new_with_skin_face_hair_labels": 1, "of_val_test_items_EXCLUDE": 1,
                              "name_not_recognised": 1}
    out = tmp_path / "rep.json"
    assert main(["--inshop", str(tmp_path), "--split-file", str(split), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["dfmm_reference"].startswith("only our")
