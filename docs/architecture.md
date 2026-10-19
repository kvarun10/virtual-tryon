# Software architecture

```
src/
├── config.py                 typed config (dataclasses) + YAML loading, _base_ inheritance, overrides
├── data/
│   ├── labels.py             label sets, LUT conversions, palettes, identity-label injection
│   ├── preprocessing.py      image/label I/O, letterbox resize, tensor conversion
│   ├── transforms.py         joint affine augmentation (image + labels, left/right swap)
│   ├── deepfashion_mm.py     DFMM index, validation, statistics, same-item pairs
│   ├── splits.py             item-disjoint reproducible splits
│   ├── dataset.py            prepare_dataset(), PreTreatmentDataset, DataLoader factory
│   ├── reference.py          two reference modes (person_worn / garment_only)
│   └── synthetic.py          SYNTHETIC test fixture in DFMM layout
├── models/
│   ├── schp_network.py       SCHP ResNet-101 in pure PyTorch (ABNCompat)
│   ├── segmentation.py       backend interface, SCHP/SegFormer/annotation backends, cache
│   └── weights.py            weight registry, consent-gated download, provenance
├── inference/
│   ├── pipeline.py           PreTreatment, try_on() (stops with StageNotAvailableError after Stage A)
│   └── visualization.py      overlays, legends, panels
├── evaluation/
│   ├── metrics.py            confusion matrix, pixel accuracy, IoU / mIoU
│   └── evaluate.py           parser vs DFMM manual parsing
└── utils/                    device report, seeding, logging, Drive paths
```

Planned for Phase 4+: `models/warp_network.py`, `models/discriminator.py`, `models/alignment_network.py`,
`models/spade_generator.py`, and `training/{losses,checkpointing,train_warp,train_texture}.py`.

## Data flow in Stage A

```
RGB image ──► SegmentationBackend.predict() ──► SegmentationResult(labels, native label set)
                                                         │
                                         to_project_labels()  (LUT)
                                                         ▼
                              garment12 (clothing)   /   body15 or pascal_person_part (body)
                                                         │
   reference: keep_classes(transfer_classes) ──► C_seg   │   target: body map ──► B_seg
```

Swapping a backend means changing `segmentation.clothing_backend` / `body_backend` in the config.
Downstream code only sees project label sets.

## Error philosophy

Every missing resource raises a dedicated exception that says what to do:
`DatasetUnavailableError`, `WeightsUnavailableError`, `CheckpointMismatchError`, `LabelError`,
`ReferenceError`, `StageNotAvailableError`, `GPUUnavailableError`, `ConfigError`. No stage returns
a placeholder output.
