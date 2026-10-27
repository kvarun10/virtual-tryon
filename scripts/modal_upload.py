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
    a = ap.parse_args(argv)
    img_zip = build_img_zip(find_img_root(SRC / "inshop" / "img"), WORK / "img.zip")
    ck = sorted((SRC / "checkpoints" / "baseline_128").glob("*.zip"))
    gens = {r: extract_generator([z for z in ck if z.name.startswith(r)], r, WORK / "b0") for r in ("warp_pf", "texture_pf")}
    plan = [(img_zip, "/raw/inshop/img.zip"),
            (SRC / "inshop" / "Eval" / "list_eval_partition.txt", "/raw/inshop/list_eval_partition.txt"),
            *[(z, f"/raw/drive_zips/{z.name}") for z in sorted((SRC / "cache").glob("*.zip"))],
            *[(z, f"/raw/drive_zips/{z.name}") for z in sorted((SRC / "pretrained").glob("*.zip"))],
            *[(p, f"/project/checkpoints/baseline_128/{r}/generator_only.pt") for r, p in gens.items()]]
    total = 0
    for local, remote in plan:
        if not Path(local).is_file():
            raise FileNotFoundError(local)
        total += Path(local).stat().st_size
        print(f"{Path(local).stat().st_size / 1e9:6.2f} GB  {remote}")
    print(f"total {total / 1e9:.2f} GB")
    if a.dry_run:
        return 0
    import modal

    vol = modal.Volume.from_name("tryon-vol", create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        for local, remote in plan:
            batch.put_file(str(local), remote)
    print("uploaded to volume tryon-vol")
    return 0


if __name__ == "__main__":
    sys.exit(main())
