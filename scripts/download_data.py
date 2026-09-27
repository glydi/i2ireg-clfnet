#!/usr/bin/env python
"""Step 1 - download the datasets into ./data/raw.

  kidneystonetr  KidneyStoneTR (the paper's own NCCT dataset, CC BY 4.0, ~550 MB)
                 https://doi.org/10.6084/m9.figshare.31156783
  totalseg       TotalSegmentator small subset (CC BY 4.0). Only ct.nii.gz + kidney masks
                 of subjects that actually contain kidneys are fetched (HTTP range requests),
                 ~2 GB instead of 3.2 GB. Used as the kidney-ROI ground truth source, because
                 KidneyStoneTR's public release ships without the radiologists' kidney masks.
                 https://zenodo.org/records/10047263
  ctkidney       (optional, external test) Islam et al. CT KIDNEY DATASET from Kaggle.
                 Needs a Kaggle account -> prints manual instructions.

Usage:  python scripts/download_data.py --all
        python scripts/download_data.py kidneystonetr totalseg --max-subjects 20
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

KSTR_URL = "https://ndownloader.figshare.com/files/62547886"
KSTR_MD5 = "10e749f4afa041c59ebf4514e69b6d9e"
TS_URL = "https://zenodo.org/records/10047263/files/Totalsegmentator_dataset_small_v201.zip?download=1"


def fetch(url: str, dst: Path, md5: str | None = None):
    if dst.exists() and (md5 is None or _md5(dst) == md5):
        print(f"  already downloaded: {dst.name}")
        return
    from tqdm import tqdm
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "i2ireg-clfnet"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f, tqdm(
            total=int(r.headers.get("Content-Length", 0)), unit="B", unit_scale=True, desc=dst.name) as bar:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    if md5 and _md5(tmp) != md5:
        sys.exit(f"MD5 mismatch for {dst.name}; delete it and retry.")
    tmp.rename(dst)


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_rar(archive: Path, out: Path):
    """RAR5 needs an external tool: unar, 7z, unrar or bsdtar (or python rarfile + unrar)."""
    out.mkdir(parents=True, exist_ok=True)
    tools = [
        ("unar", ["unar", "-q", "-f", "-o", str(out), str(archive)]),
        ("7z", ["7z", "x", "-y", f"-o{out}", str(archive)]),
        ("7zz", ["7zz", "x", "-y", f"-o{out}", str(archive)]),
        ("unrar", ["unrar", "x", "-o+", str(archive), str(out) + "/"]),
        ("bsdtar", ["bsdtar", "-xf", str(archive), "-C", str(out)]),
    ]
    import os
    for exe in (Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "7-Zip" / "7z.exe",
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "7-Zip" / "7z.exe"):
        if exe.exists():                                   # Windows 7-Zip is usually not on PATH
            tools.insert(0, (str(exe), [str(exe), "x", "-y", f"-o{out}", str(archive)]))
    for name, cmd in tools:
        if shutil.which(name):
            print(f"  extracting with {name} ...")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            return
    sys.exit("No RAR extractor found. Install one, e.g.\n"
             "  Ubuntu/Debian: sudo apt install unar      Fedora: sudo dnf install unar\n"
             "  macOS: brew install unar                   Windows: install 7-Zip from https://www.7-zip.org")


def get_kidneystonetr():
    print("[KidneyStoneTR]")
    RAW.mkdir(parents=True, exist_ok=True)
    rar = RAW / "KidneyStoneTR.rar"
    out = RAW / "KidneyStoneTR"
    if (out / "Dataset" / "dataset_integrated_task" / "holdout").exists():
        print("  already extracted")
        return
    fetch(KSTR_URL, rar, KSTR_MD5)
    extract_rar(rar, out)
    print(f"  -> {out}")


def get_totalseg(max_subjects: int | None):
    """Fetch masks first (tiny), then CT volumes only for subjects whose kidneys are present."""
    import nibabel as nib
    from remotezip import RemoteZip
    from tqdm import tqdm

    print("[TotalSegmentator small subset]")
    out = RAW / "totalseg"
    out.mkdir(parents=True, exist_ok=True)
    with RemoteZip(TS_URL) as z:
        names = set(z.namelist())
        subjects = sorted({n.split("/")[0] for n in names if n.endswith("/ct.nii.gz")})
        (out / "meta.csv").write_bytes(z.read("meta.csv"))
        kept = 0
        for s in tqdm(subjects, desc="subjects"):
            if max_subjects and kept >= max_subjects:
                break
            sdir = out / s
            if (sdir / "ct.nii.gz").exists():
                kept += 1
                continue
            (sdir / "segmentations").mkdir(parents=True, exist_ok=True)
            has_kidney = False
            for side in ("left", "right"):
                member = f"{s}/segmentations/kidney_{side}.nii.gz"
                dst = sdir / "segmentations" / f"kidney_{side}.nii.gz"
                dst.write_bytes(z.read(member))
                has_kidney |= bool(nib.load(str(dst)).dataobj[...].any())
            if not has_kidney:
                shutil.rmtree(sdir)
                continue
            tmp = sdir / "ct.nii.gz.part"
            tmp.write_bytes(z.read(f"{s}/ct.nii.gz"))
            tmp.rename(sdir / "ct.nii.gz")
            kept += 1
    print(f"  {kept} subjects with kidneys -> {out}")


def ctkidney_instructions():
    dst = RAW / "ctkidney"
    print(f"""[CT KIDNEY DATASET - optional external test set]
  Requires a Kaggle account. Either:
    pip install kaggle   # put your API token at ~/.kaggle/kaggle.json
    kaggle datasets download -d nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone -p {dst} --unzip
  or download it in the browser from
    https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone
  and unzip it into {dst}  (the 'Normal' and 'Stone' folders are used).""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="*", choices=["kidneystonetr", "totalseg", "ctkidney"])
    ap.add_argument("--all", action="store_true", help="kidneystonetr + totalseg (+ ctkidney instructions)")
    ap.add_argument("--max-subjects", type=int, default=40,
                    help="TotalSegmentator subjects with kidneys to fetch (default 40, ~1.3 GB; 0 = all)")
    a = ap.parse_args()
    what = set(a.what) | ({"kidneystonetr", "totalseg", "ctkidney"} if a.all else set())
    if not what:
        ap.error("choose datasets or --all")
    if "kidneystonetr" in what:
        get_kidneystonetr()
    if "totalseg" in what:
        get_totalseg(a.max_subjects)
    if "ctkidney" in what:
        ctkidney_instructions()


if __name__ == "__main__":
    main()
