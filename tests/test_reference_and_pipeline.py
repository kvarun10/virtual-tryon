import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.config import load_config
from src.data.dataset import prepare_dataset
from src.data.labels import get_label_set
from src.data.preprocessing import load_image
from src.data.reference import ReferenceError, build_reference_preprocessor, remove_plain_background
from src.evaluation.evaluate import evaluate_clothing_parser
from src.evaluation.metrics import ConfusionMatrix
from src.inference.pipeline import PreTreatment, StageNotAvailableError, try_on
from src.models.segmentation import build_backend


def _product_photo(bg=(245, 245, 245)):
    """Synthetic 'product photo': a T-shirt shape with a background-coloured logo inside."""
    img = Image.new("RGB", (120, 140), bg)
    d = ImageDraw.Draw(img)
    d.polygon([(30, 20), (90, 20), (115, 50), (95, 60), (95, 125), (25, 125), (25, 60), (5, 50)], fill=(30, 60, 160))
    d.rectangle((50, 60, 70, 80), fill=bg)  # logo with the background colour -> must stay foreground
    return np.asarray(img)


def test_background_removal_keeps_garment_and_holes():
    img = _product_photo()
    mask = remove_plain_background(img)
    assert mask[70, 60]  # logo hole filled
    assert mask[100, 60] and not mask[5, 5] and not mask[135, 2]
    assert 0.3 < mask.mean() < 0.7


def test_background_removal_rejects_busy_background():
    noisy = np.random.default_rng(0).integers(0, 255, (60, 60, 3)).astype(np.uint8)
    with pytest.raises(ReferenceError, match="not a uniform background"):
        remove_plain_background(noisy)


def test_garment_only_branch_is_marked_experimental(smoke_cfg):
    pre = build_reference_preprocessor(smoke_cfg, mode="garment_only")
    with pytest.raises(ReferenceError, match="category"):
        pre.prepare(_product_photo())
    ref = pre.prepare(_product_photo(), garment_category="top")
    g = get_label_set("garment12")
    assert ref.validated is False and ref.warnings
    assert set(np.unique(ref.garment_map)) == {0, g.index("top")}


def test_person_worn_and_pipeline_stop(smoke_cfg, tmp_path):
    prepared = prepare_dataset(smoke_cfg)
    pre = PreTreatment(smoke_cfg)  # annotation backends -> no weights needed
    a, b = prepared.records("train")[:2]
    ref_img = load_image(prepared.index.root / a.image)
    tgt_img = load_image(prepared.index.root / b.image)
    res = try_on(ref_img, tgt_img, smoke_cfg, reference_key=a.name, target_key=b.name,
                 stop_after="pretreatment", pretreatment=pre, save_dir=tmp_path / "out")
    g = get_label_set("garment12")
    assert res.reference.garment_map.shape == ref_img.shape[:2]
    present = {g.names[i] for i in np.unique(res.reference.garment_map)}
    assert present <= {"background", *smoke_cfg.labels.transfer_classes}  # face/hair removed from C_seg
    assert res.target.body_map.max() < 15
    assert (tmp_path / "out" / "C_seg_labels.png").is_file() and (tmp_path / "out" / "pretreatment.json").is_file()
    with pytest.raises(StageNotAvailableError, match="Phase 4"):
        try_on(ref_img, tgt_img, smoke_cfg, reference_key=a.name, target_key=b.name, pretreatment=pre)
    with pytest.raises(StageNotAvailableError, match="experimental"):
        try_on(_product_photo(), tgt_img, smoke_cfg, target_key=b.name, reference_mode="garment_only",
               garment_category="top", pretreatment=pre, stop_after="texture")


def test_person_worn_rejects_empty_reference(smoke_cfg):
    from src.models.segmentation import SegmentationBackend, SegmentationResult

    class Empty(SegmentationBackend):
        name, role, label_set = "empty", "clothing", "lip"

        def predict(self, image, key=None):
            return SegmentationResult(np.zeros(image.shape[:2], np.uint8), "lip", self.name, self.role)

    pre = build_reference_preprocessor(smoke_cfg, Empty())
    with pytest.raises(ReferenceError, match="garment_only"):
        pre.prepare(np.full((20, 20, 3), 255, np.uint8))


def test_confusion_matrix_values():
    cm = ConfusionMatrix(3)
    gt = np.array([[0, 1, 1, 2]])
    pred = np.array([[0, 1, 2, 2]])
    cm.update(pred, gt)
    assert cm.pixel_accuracy() == 0.75
    np.testing.assert_allclose(cm.iou(), [1.0, 0.5, 0.5])
    assert cm.mean_iou(exclude=[0]) == 0.5
    cm2 = ConfusionMatrix(4)
    cm2.update(pred, gt)
    assert np.isnan(cm2.iou()[3]) and cm2.mean_iou() == pytest.approx((1 + 0.5 + 0.5) / 3)


def test_parser_evaluation_code_path(smoke_cfg):
    prepared = prepare_dataset(smoke_cfg)
    gt_backend = build_backend("annotation_parsing", smoke_cfg)
    out = evaluate_clothing_parser(gt_backend, prepared, smoke_cfg, split="train", limit=3, progress=False)
    assert out["pixel_accuracy"] == 1.0 and out["mean_iou"] == 1.0  # GT vs itself
    assert out["synthetic"] is True
