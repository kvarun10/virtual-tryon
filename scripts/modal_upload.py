"""Upload the files the Modal jobs need from report/private/kaggle_upload/ to the private volume "tryon-vol".

    python scripts/modal_upload.py            # build the zip, show the plan, upload
    python scripts/modal_upload.py --dry-run  # only show what would be uploaded

Uploaded (about 4.5 GB):
* In-shop photos as ONE stored zip (52,712 small files upload slowly one by one) -> /raw/inshop/img.zip
* Eval/list_eval_partition.txt                                                      -> /raw/inshop/
* the Drive zips of cache/ (old splits) and pretrained/ (SCHP, VGG, LPIPS, Inception) -> /raw/drive_zips/
* only generator_only.pt of the frozen B0 runs (warp_pf, texture_pf), extracted from the Drive zips
                                                                   -> /project/checkpoints/baseline_128/<run>/
Nothing here is public: a Modal volume is visible only to your workspace.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "private" / "kaggle_upload"
WORK = ROOT / "report" / "private" / "modal_upload"


def build_img_zip(img_root: Path, out: Path) -> Path:
    """Zip every In-shop jpg as ``img/<GENDER>/<Category>/id_x/<file>.jpg`` (stored, no recompression)."""
    files = sorted(img_root.rglob("*.jpg"))
    if out.is_file():
        with zipfile.ZipFile(out) as z:
            if len(z.namelist()) == len(files):
                return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            rel = f.relative_to(img_root).as_posix()  # MEN/Denim/id_.../01_1_front.jpg
            z.write(f, "img/" + rel)
    tmp.replace(out)
    return out


def find_img_root(base: Path) -> Path:
    for cand in sorted(base.rglob("MEN"), key=lambda p: len(p.parts)):
        if cand.is_dir() and (cand.parent / "WOMEN").is_dir():
            return cand.parent
    raise FileNotFoundError(f"no MEN/ and WOMEN/ folders under {base}")


def extract_generator(zips, run: str, out_dir: Path) -> Path:
    member = f"{run}/generator_only.pt"
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            if member in zf.namelist():
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir / run / "generator_only.pt"
                if not target.is_file():
                    zf.extract(member, out_dir)
                return target
    raise FileNotFoundError(f"{member} not in {[Path(z).name for z in zips]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-models", action="store_true",
                    help="also upload the trained full_256 / v2_256 models from report/private/models (moving workspaces)")
    a = ap.parse_args(argv)
    img_zip = build_img_zip(find_img_root(SRC / "inshop" / "img"), WORK / "img.zip")
    ck = sorted((SRC / "checkpoints" / "baseline_128").glob("*.zip"))
    gens = {r: extract_generator([z for z in ck if z.name.startswith(r)], r, WORK / "b0") for r in ("warp_pf", "texture_pf")}
    plan = [(img_zip, "/raw/inshop/img.zip"),
            (SRC / "inshop" / "Eval" / "list_eval_partition.txt", "/raw/inshop/list_eval_partition.txt"),
            *[(z, f"/raw/drive_zips/{z.name}") for z in sorted((SRC / "cache").glob("*.zip"))],
            *[(z, f"/raw/drive_zips/{z.name}") for z in sorted((SRC / "pretrained").glob("*.zip"))],
            *[(p, f"/project/checkpoints/baseline_128/{r}/generator_only.pt") for r, p in gens.items()]]
    if a.with_models:
        models = ROOT / "report" / "private" / "models"
        for f in sorted(models.rglob("*.pt")):  # e.g. full_256/texture/last.pt -> /project/checkpoints/full_256/texture/last.pt
            plan.append((f, "/project/checkpoints/" + f.relative_to(models).as_posix()))
    total = 0
    for local, remote in plan:
        if not Path(local).is_file():
            raise FileNotFoundError(local)
        total += Path(local).stat().st_size
        print(f"{Path(local).stat().st_size / 1e9:6.2f} GB  {remote}")
    print(f"total {total / 1e9:.2f} GB")
    if a.dry_run:
        return 0
    import time

    import modal

    vol = modal.Volume.from_name("tryon-vol", create_if_missing=True)

    def remote_size(path: str):
        try:
            entries = vol.listdir(str(Path(path).parent.as_posix()))
        except Exception:  # noqa: BLE001  (folder does not exist yet)
            return None
        for e in entries:
            if "/" + e.path.lstrip("/") == path:
                return getattr(e, "size", None)
        return None

    done = 0
    for local, remote in plan:  # one file per batch: each file is committed as soon as it is uploaded
        size = Path(local).stat().st_size
        if remote_size(remote) == size:
            done += size
            print(f"skip (already uploaded) {remote}", flush=True)
            continue
        t = time.time()
        with vol.batch_upload(force=True) as batch:
            batch.put_file(str(local), remote)
        done += size
        dt = max(time.time() - t, 1e-3)
        print(f"uploaded {remote}: {size / 1e9:.2f} GB in {dt / 60:.1f} min ({size / 1e6 / dt:.2f} MB/s); "
              f"total {done / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
    print("all files are on volume tryon-vol")
    return 0


if __name__ == "__main__":
    sys.exit(main())
