"""Datasets, preprocessing and augmentation."""
from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KSTR_RAW = DATA / "raw" / "KidneyStoneTR" / "Dataset" / "dataset_integrated_task"
KSTR_DIR = DATA / "kidneystonetr"          # index.csv + masks/<md5>.png (pseudo ROI masks)
TS2D_DIR = DATA / "totalseg_2d"            # index.csv + images/ + masks/
IMG_SIZE = 224


# ----------------------------------------------------------------------------- I/O helpers
def load_gray(path, size: int = IMG_SIZE) -> np.ndarray:
    """Any image file (PNG/JPG/... or DICOM) -> float32 HxW in [0,1], resized to size x size."""
    if str(path).lower().endswith((".dcm", ".dicom")):
        import pydicom
        ds = pydicom.dcmread(str(path))
        hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + \
            float(getattr(ds, "RescaleIntercept", 0))
        im = Image.fromarray((window_hu(hu) * 255).astype(np.uint8))
    else:
        im = Image.open(path).convert("L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_mask(path, size: int = IMG_SIZE) -> np.ndarray:
    im = Image.open(path).convert("L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.NEAREST)
    return (np.asarray(im) > 127).astype(np.float32)


def file_md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def window_hu(hu: np.ndarray, level: float = 40, width: float = 400) -> np.ndarray:
    lo = level - width / 2
    return np.clip((hu - lo) / width, 0, 1).astype(np.float32)


def read_csv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows: list[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- augmentation
class Augment:
    """Light, label-preserving augmentation applied identically to image and ROI mask:
    random affine (rot/scale/shift), horizontal flip is NOT used (left/right kidneys are
    anatomically distinct), plus contrast/brightness/gamma jitter on the image only."""

    def __init__(self, rot=10, scale=(0.9, 1.1), shift=0.06, intensity=True):
        self.rot, self.scale, self.shift, self.intensity = rot, scale, shift, intensity

    def __call__(self, img: torch.Tensor, mask: torch.Tensor | None):
        import torchvision.transforms.v2.functional as TF
        ang = random.uniform(-self.rot, self.rot)
        sc = random.uniform(*self.scale)
        h = img.shape[-1]
        tx, ty = (int(random.uniform(-self.shift, self.shift) * h) for _ in range(2))
        img = TF.affine(img, angle=ang, translate=[tx, ty], scale=sc, shear=[0.0],
                        interpolation=TF.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.affine(mask, angle=ang, translate=[tx, ty], scale=sc, shear=[0.0],
                             interpolation=TF.InterpolationMode.NEAREST)
        if self.intensity:
            c = random.uniform(0.85, 1.15)
            b = random.uniform(-0.05, 0.05)
            g = random.uniform(0.85, 1.15)
            img = ((img * c + b).clamp(0, 1)) ** g
        return img, mask


# ----------------------------------------------------------------------------- datasets
def to_tensor3(img: np.ndarray) -> torch.Tensor:
    """HxW [0,1] -> 3xHxW (grayscale replicated, as in the paper's 224x224x3 input)."""
    return torch.from_numpy(img)[None].repeat(3, 1, 1)


class SliceDataset(Dataset):
    """Generic slice dataset.

    rows: dicts with 'path', optional 'label' (0/1) and optional 'mask' (path).
    Returns dict(image 3xHxW, label, mask 1xHxW (zeros if absent), has_mask, roi 3xHxW).
    roi = mask * image is the regression target (Sec. 3.1, Fig. 3.1d)."""

    def __init__(self, rows, augment: Augment | None = None, size: int = IMG_SIZE):
        self.rows, self.augment, self.size = rows, augment, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = to_tensor3(load_gray(r["path"], self.size))
        has_mask = bool(r.get("mask")) and Path(r["mask"]).exists()
        mask = torch.from_numpy(load_mask(r["mask"], self.size))[None] if has_mask \
            else torch.zeros(1, self.size, self.size)
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return {
            "image": img,
            "mask": mask,
            "roi": img * mask,
            "has_mask": torch.tensor(float(has_mask)),
            "label": torch.tensor(int(r.get("label", -1))),
            "path": str(r["path"]),
        }


# ----------------------------------------------------------------------------- KidneyStoneTR
def build_kstr_index() -> list[dict]:
    """Indexes the official patient-level splits. One row per (file, split); `md5` identifies
    duplicated images so each unique slice gets one pseudo-mask and leakage can be removed."""
    if not KSTR_RAW.exists():
        raise SystemExit(f"KidneyStoneTR not found at {KSTR_RAW}. Run scripts/download_data.py first.")
    rows = []
    for split_dir, split, fold in (
            [(KSTR_RAW / "cross_validation" / "Training" / f"Fold_{k}", "train", k) for k in range(1, 6)]
            + [(KSTR_RAW / "cross_validation" / "Test" / f"Fold_{k}", "test", k) for k in range(1, 6)]
            + [(KSTR_RAW / "holdout", "holdout", 0)]):
        for cls, label in (("Negative", 0), ("Positive", 1)):
            for p in sorted((split_dir / cls).glob("*.jpg"), key=lambda p: (len(p.stem), p.stem)):
                rows.append({"path": str(p.relative_to(ROOT)), "md5": file_md5(p),
                             "label": label, "split": split, "fold": fold})
    return rows


def kstr_rows(split: str, fold: int = 0, drop_holdout_dups: bool = True, masks: bool = True):
    """split in {'train','test','holdout','trainall'}; 'trainall' = all CV images (for final
    training before hold-out evaluation). Paths are absolute; `mask` points to the pseudo-mask."""
    idx = KSTR_DIR / "index.csv"
    if not idx.exists():
        write_csv(idx, build_kstr_index())
    rows = read_csv(idx)
    hold = {r["md5"] for r in rows if r["split"] == "holdout"}
    if split == "trainall":
        sel = [r for r in rows if r["split"] == "test"]            # test folds 1-5 ...
        extra = {r["md5"] for r in sel}
        sel += [r for r in rows if r["split"] == "train" and r["md5"] not in extra]  # ... + train-only
        seen, uniq = set(), []
        for r in sel:
            if r["md5"] not in seen:
                seen.add(r["md5"])
                uniq.append(r)
        sel = uniq
    else:
        sel = [r for r in rows if r["split"] == split and (split == "holdout" or int(r["fold"]) == fold)]
    if drop_holdout_dups and split in ("train", "trainall"):
        sel = [r for r in sel if r["md5"] not in hold]
    out = []
    for r in sel:
        m = KSTR_DIR / "masks" / f"{r['md5']}.png"
        out.append({"path": str(ROOT / r["path"]), "label": int(r["label"]), "md5": r["md5"],
                    "mask": str(m) if masks and m.exists() else ""})
    return out


def kstr_unique_rows():
    rows = read_csv(KSTR_DIR / "index.csv") if (KSTR_DIR / "index.csv").exists() else build_kstr_index()
    seen, out = set(), []
    for r in rows:
        if r["md5"] not in seen:
            seen.add(r["md5"])
            out.append({"path": str(ROOT / r["path"]), "md5": r["md5"], "label": int(r["label"])})
    return out


# ----------------------------------------------------------------------------- TotalSegmentator 2D
def ts2d_rows(val_fraction: float = 0.15, seed: int = 0):
    """Subject-level train/val split of the 2D kidney slices."""
    rows = read_csv(TS2D_DIR / "index.csv")
    subjects = sorted({r["subject"] for r in rows})
    random.Random(seed).shuffle(subjects)
    n_val = max(1, round(len(subjects) * val_fraction))
    val_s = set(subjects[:n_val])
    fix = lambda r: {"path": str(TS2D_DIR / r["image"]), "mask": str(TS2D_DIR / r["mask"])}
    return [fix(r) for r in rows if r["subject"] not in val_s], [fix(r) for r in rows if r["subject"] in val_s]


# ----------------------------------------------------------------------------- CT KIDNEY (external)
def ctkidney_rows(root: Path | None = None, n_per_class: int | None = 848, seed: int = 0):
    """Normal (0) vs Stone (1) from Islam et al.'s CT KIDNEY DATASET (external, no masks).
    The paper kept only axial stone slices and a balanced random set of normals; axial/coronal
    is not labelled in the release, so we sample a balanced subset (see README)."""
    root = root or DATA / "raw" / "ctkidney"
    found = {}
    for cls in ("Normal", "Stone"):
        cands = [p for p in root.rglob(cls) if p.is_dir()]
        if not cands:
            raise SystemExit(f"Could not find a '{cls}' folder under {root}. See scripts/download_data.py ctkidney.")
        found[cls] = sorted(p for p in cands[0].iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    rng = random.Random(seed)
    rows = []
    for cls, label in (("Normal", 0), ("Stone", 1)):
        files = found[cls]
        if n_per_class and len(files) > n_per_class:
            files = rng.sample(files, n_per_class)
        rows += [{"path": str(p), "label": label, "mask": ""} for p in files]
    return rows


# ----------------------------------------------------------------------------- any folder
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm")
NEG_NAMES = {"negative", "normal", "stone-", "stone_negative", "no_stone", "0"}
POS_NAMES = {"positive", "stone", "stone+", "stone_positive", "1"}


def folder_rows(root) -> tuple[list[dict], bool]:
    """Images from a folder. Labels come from the name of any parent directory
    (Positive/Stone/1 -> 1, Negative/Normal/0 -> 0), searched recursively. Returns (rows,
    labelled); unlabelled images get label -1 (predictions only, no accuracy)."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"not a folder: {root}")
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        label = -1
        for part in reversed(p.relative_to(root.parent).parts[:-1]):   # includes root itself
            k = part.lower()
            if k in POS_NAMES:
                label = 1
                break
            if k in NEG_NAMES:
                label = 0
                break
        rows.append({"path": str(p), "label": label, "mask": ""})
    return rows, bool(rows) and all(r["label"] >= 0 for r in rows)
