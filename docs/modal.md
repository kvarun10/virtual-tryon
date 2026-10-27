# Full-data 256 px run on Modal (experiment `full_256`)

Budget: $30 Modal Starter credit; deadline ~1.5 days (from 2026-10-27). B0 (`baseline_128`, tag baseline-v1) stays frozen
and is the comparison point. Code: `modal_app.py`, `scripts/modal_upload.py`, `src/data/packed.py`, `src/data/packing.py`,
`configs/modal_full_256.yaml`.

## What changes vs B0 (decided before training; no tuning on validation)

| | B0 | full_256 |
|---|---|---|
| photos | 10,160 manually parsed DFMM photos | all In-shop photos except val/test items and the standard benchmark test items |
| garment labels | DFMM manual | manual where it exists, SCHP-LIP otherwise (`data.label_source=manual_then_auto`) |
| resolution | 128 px | 256 px (paper) |
| texture widths | corr 32, generator 32 ch | corr 64, generator 64 ch (CoCosNet / SPADE defaults) |
| batch | 8 | 16 |
| weights used | live | EMA of the generator (decay 0.999) |
| losses, learning rates, pair filter, label sets | | unchanged |

Facts behind the data choice (measured 2026-10-27):
* The In-shop `img.zip` photos (256 x 256) are the DFMM photos letterboxed to 174 x 256 at column 41: DFMM manual
  parses resized with our letterbox overlay them exactly. All 44,096 DFMM photos are among the 52,712 In-shop photos;
  8,616 are In-shop only. So no 5.4 GB DFMM image download is needed, and training/benchmark use the benchmark's own images.
* The standard benchmark (PATN pairs) is built from these In-shop photos; the benchmark crops columns 41:215.
* `BatchedSCHP` (GPU batch + `grid_sample` inverse warp) agrees with the original one-image SCHP on 99.99 % of pixels
  (3 photos, both parsers, real weights).

The packing job reports how well SCHP-LIP agrees with the manual labels on the 12,701 parsed photos
(`report.json`, `parser_agreement_vs_manual_garment12`); this is the quality of the automatic labels.

## Napkin maths (before measuring; the probe replaces the assumptions)

Measured on T4 at 128 px, batch 8 (implementation_status.md): warp 0.22 s/step, texture 0.35 s/step.
At 256 px with doubled texture widths: warp ~4x, texture ~16x (4x pixels; 4x conv FLOPs from 2x channels; the
64 x 64 correspondence is 16x the 32 x 32 one). Assumed speed vs T4 for fp16 conv nets: L4 2x, A10 2.5x, L40S 5x,
A100 6x, H100 10x.

Planned length: warp 12k steps, texture 14k steps at batch 16 (B0: 10k / 13k at batch 8 on 1/4 of the data).

| GPU | $/h (+CPU/RAM ~0.4) | warp 12k | texture 14k | GPU $ total | wall time (stages in parallel) |
|---|---|---|---|---|---|
| T4 | 0.59 | 6 h | 43 h | 29 | 43 h: too slow |
| A100 40 GB | 2.10 | 1.0 h | 7.2 h | 17 (+3) | ~7 h |
| L40S | 1.95 | 1.2 h | 8.6 h | 19 (+4) | ~9 h |
| H100 | 3.95 | 0.6 h | 4.3 h | 19.5 (+2) | ~4.5 h |

All fast GPUs cost about the same per step, so the shortest wall time wins: H100 unless the probe shows it is less
than ~1.9x faster than A100 40 GB. Plan: setup (CPU, ~$0.1), pack (L4, ~$0.6), probe on H100/A100/L40S (~$1),
both stages in parallel (~$21), two benchmarks (L4, ~$1); ~$6 reserve. Step counts are fixed from the probe so the
total stays under $25.

## Order of work

1. `python scripts/modal_upload.py` (2.7 GB: In-shop photos as one zip, partition list, cache + pretrained Drive zips,
   B0 generator_only files).
2. `modal run modal_app.py::setup` - unpacks; downloads the two pair lists (approved 2026-10-26) and the DFMM
   parsing archive (90 MB, official link, as in Colab).
3. `modal run modal_app.py::pack` - store + report; `modal run modal_app.py::pack --limit 200` first as a smoke test.
4. `modal run modal_app.py::probe` - seconds/step and memory on three GPUs; choose GPU and steps.
5. `modal run --detach modal_app.py::train --stage warp ...` and `--stage texture ...` in parallel. A short first
   call (e.g. 1,500 steps) is checked (losses, validation grid) before continuing the same run to full length;
   resume is exact, so nothing is wasted.
6. `modal run modal_app.py::benchmark --which b0` and `--which full_256`: same pairs rule, same images.

## Honest limits

* The paper's evaluation protocol is unpublished (paper_comparison.md); matching 0.95 SSIM is not guaranteed.
* Our locked test items stay unscored in the benchmark; the count is in the report.
* Automatic labels carry SCHP errors (e.g. top/outer confusion, P11) into training targets.
