"""Build the packed 256 px store (:mod:`src.data.packed`) from the DeepFashion In-shop photos (one GPU job).

Inputs (all read straight from zip files, nothing is extracted):

* In-shop ``img.zip`` (members ``img/<GENDER>/<Category>/id_<item>/<seq>_<view>.jpg``, 256 x 256) and
  ``Eval/list_eval_partition.txt`` (the list of all 52,712 photos)
* DFMM manual parsing ``parsing.zip`` (``*_segm.png``, 750 x 1101) for the 12,701 manually parsed photos
* our old split file (val / test are kept exactly) and the standard pose-transfer test pairs (their items are
  never trained on)

Steps: (1) one pass over all photos: JPEG bytes copied unchanged, manual parsing letterboxed to 256 (nearest),
SCHP-LIP and SCHP-Pascal on the GPU in batches (garment12 / body maps at 256 directly); (2) garment signatures and
the same-item pair table; (3) splits. ``report.json`` records counts, the agreement of the SCHP maps with the manual
maps (garment12 mIoU) and timings.

:class:`BatchedSCHP` reproduces ``SCHPBackend.predict`` (same affine, same fusion logits), but batches the
network and does the inverse warp of the logits on the GPU with ``grid_sample`` (cv2 used ``warpAffine`` on the CPU,
channel by channel). tests/test_packed.py checks the two agree.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.inshop import inshop_to_dfmm_stem, read_eval_partition
from src.data.labels import convert_labels, get_label_set
from src.data.packed import PACK_VERSION, BlobWriter, PackedStore, build_packed_splits, decode_jpeg, encode_png
from src.data.pair_quality import garment_signature, pair_rows, write_pair_table
from src.data.preprocessing import resize_labels
from src.models.segmentation import SCHP_MEAN_BGR, SCHP_STD_BGR, _schp_affine, _schp_center_scale

SIZE = (256, 256)


def warp_input(rgb: np.ndarray, trans: np.ndarray, input_size) -> np.ndarray:
    """CPU part of SCHP preprocessing: the same cv2 affine as ``SCHPBackend.predict``; returns BGR uint8."""
    import cv2

    in_h, in_w = input_size
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return cv2.warpAffine(bgr, trans, (in_w, in_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


class BatchedSCHP:
    """Batched SCHP inference for images that all have the same size ``src_hw``."""

    def __init__(self, backend, src_hw=SIZE):
        self.b = backend
        self.dev = backend.device
        h, w = src_hw
        in_h, in_w = backend.input_size
        center, scale = _schp_center_scale(w, h, in_w / in_h)
        self.trans = _schp_affine(center, scale, backend.input_size)  # original frame -> network input frame
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        t = self.trans
        sx = t[0, 0] * xs + t[0, 1] * ys + t[0, 2]
        sy = t[1, 0] * xs + t[1, 1] * ys + t[1, 2]
        grid = np.stack([sx / (in_w - 1) * 2 - 1, sy / (in_h - 1) * 2 - 1], -1)  # align_corners=True convention
        self.grid = torch.tensor(grid, dtype=torch.float32, device=self.dev)[None]
        self.mean = torch.tensor(SCHP_MEAN_BGR, device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(SCHP_STD_BGR, device=self.dev).view(1, 3, 1, 1)
        self.label_set = backend.label_set

    @property
    def warp_spec(self):
        """Picklable (matrix, input size) for :func:`warp_input` in DataLoader workers (no model, no CUDA)."""
        return self.trans, tuple(self.b.input_size)

    def warp_input(self, rgb: np.ndarray) -> np.ndarray:
        return warp_input(rgb, *self.warp_spec)

    @torch.no_grad()
    def __call__(self, warped_bgr: torch.Tensor, amp: bool = True) -> np.ndarray:
        """``warped_bgr`` uint8 [B, in_h, in_w, 3] -> labels uint8 [B, H, W] in the backend's label set."""
        in_h, in_w = self.b.input_size
        x = warped_bgr.to(self.dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        with torch.autocast(self.dev.type, dtype=torch.float16, enabled=amp and self.dev.type == "cuda"):
            out = self.b.model(x)
        logits = F.interpolate(out[0][-1].float(), size=(in_h, in_w), mode="bilinear", align_corners=True)
        back = F.grid_sample(logits, self.grid.expand(len(x), -1, -1, -1), mode="bilinear", padding_mode="zeros",
                             align_corners=True)
        return back.argmax(1).to(torch.uint8).cpu().numpy()


class _PhotoSource(Dataset):
    """Worker-side loading: JPEG bytes, both SCHP inputs, and the manual parsing (garment12, 256) if any."""

    def __init__(self, img_zip: Path, members: List[str], stems: List[str], parsing_zip: Optional[Path],
                 parsing_members: Dict[str, str], warp_specs: Sequence):
        self.img_zip, self.members, self.stems = Path(img_zip), members, stems
        self.parsing_zip, self.parsing_members = parsing_zip, parsing_members
        self.warp_specs = list(warp_specs)
        self._z = self._pz = None

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        if self._z is None:  # one ZipFile handle per worker process
            self._z = zipfile.ZipFile(self.img_zip)
            self._pz = zipfile.ZipFile(self.parsing_zip) if self.parsing_zip else None
        data = self._z.read(self.members[i])
        rgb = decode_jpeg(data)
        if rgb.shape[:2] != SIZE:
            raise ValueError(f"{self.members[i]}: size {rgb.shape[:2]}, expected {SIZE} (In-shop img.zip)")
        manual = None
        pm = self.parsing_members.get(self.stems[i])
        if pm is not None:
            lab = np.asarray(Image.open(io.BytesIO(self._pz.read(pm))))
            if lab.ndim != 2:
                raise ValueError(f"{pm}: parsing is not single-channel")
            g = convert_labels(lab.astype(np.uint8), "dfmm", "garment12")
            manual = resize_labels(g, SIZE, "letterbox")[0]
        return {"i": i, "jpeg": data, "warped": [torch.from_numpy(warp_input(rgb, *w)) for w in self.warp_specs],
                "manual": manual}


def _collate(batch):
    return {"i": [b["i"] for b in batch], "jpeg": [b["jpeg"] for b in batch], "manual": [b["manual"] for b in batch],
            "warped": [torch.stack([b["warped"][k] for b in batch]) for k in range(len(batch[0]["warped"]))]}


def _signature_job(args):
    store_dir, names, classes = args
    st = PackedStore(Path(store_dir), "manual_then_auto")
    return {n: garment_signature(st.image(n), st.garment(n), classes) for n in names}


def build_pack(img_zip: Path, partition_file: Path, parsing_zip: Optional[Path], old_split_file: Path,
               protected_items: Set[str], clothing_backend, body_backend, out_dir: Path, body_label_set: str,
               batch_size: int = 32, num_workers: int = 8, limit: Optional[int] = None,
               transfer_classes: Sequence[str] = ("top", "outer", "dress", "skirt", "pants", "footwear"),
               progress: bool = True) -> Dict:
    """Write the store to ``out_dir`` (built in ``out_dir.tmp``, then renamed). Returns the report."""
    t0 = time.time()
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    rows = read_eval_partition(partition_file)
    seen, entries = set(), []
    for name, item_id, _ in rows:
        stem = inshop_to_dfmm_stem(name)
        if stem not in seen:
            seen.add(stem)
            entries.append((name, stem, item_id.split("_")[-1]))
    with zipfile.ZipFile(img_zip) as z:
        members = {m.replace("\\", "/").split("img/", 1)[-1]: m for m in z.namelist() if m.lower().endswith(".jpg")}
    missing = [n for n, _, _ in entries if n.split("img/", 1)[-1] not in members]
    if missing:
        raise FileNotFoundError(f"{len(missing)} partition photos are not in {img_zip}, e.g. {missing[:3]}")
    if limit:
        entries = entries[:limit]
    parsing_members = {}
    if parsing_zip:
        with zipfile.ZipFile(parsing_zip) as pz:
            for m in pz.namelist():
                if m.endswith("_segm.png"):
                    parsing_members[Path(m).name[: -len("_segm.png")]] = m
    stems = [s for _, s, _ in entries]
    lip, pascal = BatchedSCHP(clothing_backend), BatchedSCHP(body_backend)
    src = _PhotoSource(img_zip, [members[n.split("img/", 1)[-1]] for n, _, _ in entries], stems, parsing_zip,
                       parsing_members, [lip.warp_spec, pascal.warp_spec])
    loader = DataLoader(src, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate,
                        pin_memory=lip.dev.type == "cuda", persistent_workers=False, prefetch_factor=4 if num_workers else None)
    writers = {b: BlobWriter(tmp / b) for b in ("images", "garment_manual", "garment_auto", "body")}
    G = get_label_set("garment12")
    conf = np.zeros((G.num_classes, G.num_classes), np.int64)
    names_rows, t_gpu = [], 0.0
    it = loader
    if progress:
        from tqdm.auto import tqdm

        it = tqdm(loader, desc="packing (SCHP-LIP + SCHP-Pascal)", total=len(loader))
    for batch in it:
        tg = time.time()
        lip_labels = lip(batch["warped"][0])
        body_labels = pascal(batch["warped"][1])
        t_gpu += time.time() - tg
        for k, i in enumerate(batch["i"]):
            name, stem, item = entries[i]
            auto = convert_labels(lip_labels[k], lip.label_set, "garment12")
            body = convert_labels(body_labels[k], pascal.label_set, body_label_set)
            manual = batch["manual"][k]
            writers["images"].append(batch["jpeg"][k])
            writers["garment_auto"].append(encode_png(auto))
            writers["body"].append(encode_png(body))
            writers["garment_manual"].append(encode_png(manual) if manual is not None else b"")
            if manual is not None:
                conf += np.bincount(manual.astype(np.int64).ravel() * G.num_classes + auto.ravel(),
                                    minlength=G.num_classes ** 2).reshape(G.num_classes, G.num_classes)
            parts = stem.split("-")[-1].split("_")
            names_rows.append({"name": stem, "item_id": item, "seq": parts[0], "view": "_".join(parts[1:]),
                               "manual": manual is not None, "inshop": name})
    counts = {b: w.close() for b, w in writers.items()}
    if len(set(counts.values())) != 1 or counts["images"] != len(entries):
        raise RuntimeError(f"blob counts differ: {counts} for {len(entries)} photos")
    (tmp / "names.json").write_text(json.dumps(names_rows), encoding="utf-8")
    tp = np.diag(conf).astype(float)
    denom = conf.sum(0) + conf.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    agreement = {"pixel_accuracy": float(tp.sum() / max(conf.sum(), 1)),
                 "miou_excl_background": float(np.nanmean(iou[1:])) if np.isfinite(iou[1:]).any() else float("nan"),
                 "iou": {G.names[k]: (None if np.isnan(v) else round(float(v), 4)) for k, v in enumerate(iou)},
                 "photos": int(sum(r["manual"] for r in names_rows))}
    meta = {"version": PACK_VERSION, "size": list(SIZE), "garment_label_set": "garment12", "body_label_set": body_label_set,
            "photos": len(names_rows), "manual": agreement["photos"], "sources": {
                "images": f"{Path(img_zip).name} (In-shop, JPEG bytes unchanged)", "partition": Path(partition_file).name,
                "manual_parsing": Path(parsing_zip).name if parsing_zip else None,
                "garment_auto": clothing_backend.describe(), "body": body_backend.describe()},
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    t_pass1 = time.time() - t0

    # ---- garment signatures + same-item pair table (CPU) ----
    classes = G.indices(list(transfer_classes))
    by_item: Dict[str, List] = defaultdict(list)
    for r in names_rows:
        by_item[r["item_id"]].append(r)
    multi = {k: v for k, v in by_item.items() if len(v) > 1}
    todo = [r["name"] for v in multi.values() for r in v]
    chunks = [(str(tmp), todo[i:i + 500], classes) for i in range(0, len(todo), 500)]
    sigs: Dict = {}
    if num_workers > 0 and len(chunks) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(num_workers) as pool:
            for part in pool.imap_unordered(_signature_job, chunks):
                sigs.update(part)
    else:
        for c in chunks:
            sigs.update(_signature_job(c))

    class _R:  # pair_rows needs name / seq / view attributes
        def __init__(self, d):
            self.name, self.seq, self.view = d["name"], d["seq"], d["view"]

    pair_table = pair_rows({k: [_R(d) for d in v] for k, v in multi.items()}, sigs)
    write_pair_table(pair_table, tmp / "pairs.csv")

    # ---- splits ----
    old = json.loads(Path(old_split_file).read_text(encoding="utf-8"))["splits"]
    split_data = build_packed_splits(names_rows, old, set(protected_items))
    (tmp / "splits.json").write_text(json.dumps(split_data, indent=1), encoding="utf-8")

    report = {"meta": meta, "parser_agreement_vs_manual_garment12": agreement, "pairs": len(pair_table),
              "items": len(by_item), "items_with_2plus_photos": len(multi), "splits": split_data["meta"],
              "seconds": {"pass1": round(t_pass1, 1), "gpu": round(t_gpu, 1), "total": round(time.time() - t0, 1)}}
    (tmp / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return report
