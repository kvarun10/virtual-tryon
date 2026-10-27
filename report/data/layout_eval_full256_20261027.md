# Garment layout accuracy (1271 validation images, pairs {'self': 1156, 'cross_view': 115}, [256, 256] px)

| definition | mIoU excl. background | mIoU incl. background | pixel accuracy |
|---|---|---|---|
| supervised (training metric: garments + skin + background) | 77.9 | 80.5 | 98.0 |
| final layout, face/hair from the automatic parser (fully automatic) | 63.2 | 66.2 | 97.4 |
| final layout, face/hair from the manual parse (uses ground truth) | 67.3 | 69.9 | 97.7 |

Paper Table 2 (definition not published): mIoU 87.6, pixel accuracy 98.9.