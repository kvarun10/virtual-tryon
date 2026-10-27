# Garment layout accuracy (1271 validation images, pairs {'self': 1156, 'cross_view': 115}, [128, 128] px)

| definition | mIoU excl. background | mIoU incl. background | pixel accuracy |
|---|---|---|---|
| supervised (training metric: garments + skin + background) | 70.8 | 74.3 | 97.2 |
| final layout, face/hair from the automatic parser (fully automatic) | 58.9 | 62.1 | 96.6 |
| final layout, face/hair from the manual parse (uses ground truth) | 62.8 | 65.8 | 97.0 |

Paper Table 2 (definition not published): mIoU 87.6, pixel accuracy 98.9.