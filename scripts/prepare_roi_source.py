#!/usr/bin/env python
"""Step 2 - turn TotalSegmentator CT volumes into 2D axial kidney-ROI training pairs.

For every subject: reorient to radiological axial view (patient right on image left,
anterior up, as in KidneyStoneTR), apply an abdominal window (L40/W400 by default, which
matches the KidneyStoneTR JPEG appearance), pad to a square field of view, resize to 224,
and save image + binary kidney mask (left U right) for slices containing kidney.
TotalSegmentator kidney masks exclude the renal sinus / pelvis (where many stones sit),
while the paper's radiologists traced the *outer* kidney contour; each kidney is therefore
replaced by its 2D convex hull per slice.
A few kidney-free slices just above/below the kidneys are kept so the regressor learns to
output an empty ROI.

Usage:  python scripts/prepare_roi_source.py [--stride 2] [--fov-mm 420]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DATA, TS2D_DIR, window_hu, write_csv  # noqa: E402


def to_radiological(sl: np.ndarray) -> np.ndarray:
    """RAS (x,y) slice -> image rows anterior->posterior, cols patient-right->left."""
    return sl.T[::-1, ::-1]


def square_fov(img: np.ndarray, spacing: tuple[float, float], fov_mm: float, fill: float) -> np.ndarray:
    """Pad (never crop) to a square physical field of view, then return the padded array
    together with isotropic pixel spacing so a plain resize keeps aspect ratio."""
    from scipy.ndimage import zoom
    sy, sx = spacing
    if abs(sy - sx) > 1e-3:           # make pixels square
        img = zoom(img, (sy / sx, 1.0), order=1 if img.dtype != bool else 0)
        sy = sx
    h, w = img.shape
    side = max(h, w, int(round(fov_mm / sx)))
    out = np.full((side, side), fill, dtype=img.dtype)
    t, l = (side - h) // 2, (side - w) // 2
    out[t:t + h, l:l + w] = img
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA / "raw" / "totalseg"))
    ap.add_argument("--stride", type=int, default=2, help="keep every n-th axial slice")
    ap.add_argument("--min-area", type=int, default=80, help="min kidney voxels in a slice")
    ap.add_argument("--fov-mm", type=float, default=420.0)
    ap.add_argument("--level", type=float, default=40)
    ap.add_argument("--width", type=float, default=400)
    ap.add_argument("--neg-margin", type=int, default=12, help="slices beyond kidneys used as empty ROIs")
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()

    import nibabel as nib
    from skimage.morphology import convex_hull_image
    src = Path(a.src)
    subjects = sorted(p for p in src.iterdir() if (p / "ct.nii.gz").exists())
    if not subjects:
        sys.exit(f"No subjects in {src}. Run: python scripts/download_data.py totalseg")
    (TS2D_DIR / "images").mkdir(parents=True, exist_ok=True)
    (TS2D_DIR / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for s in tqdm(subjects, desc="volumes"):
        ct = nib.as_closest_canonical(nib.load(str(s / "ct.nii.gz")))
        hu = np.asarray(ct.dataobj, dtype=np.float32)
        sides = [np.asarray(nib.as_closest_canonical(
            nib.load(str(s / "segmentations" / f"kidney_{side}.nii.gz"))).dataobj) > 0 for side in ("left", "right")]
        kid = sides[0] | sides[1]
        sx, sy = ct.header.get_zooms()[:2]
        area = kid.sum((0, 1))
        zs = np.where(area >= a.min_area)[0]
        if len(zs) == 0:
            continue
        pos = set(zs[::a.stride].tolist())
        lo, hi = zs.min(), zs.max()
        neg = [z for z in range(lo - a.neg_margin, lo, 4)] + [z for z in range(hi + 4, hi + a.neg_margin + 1, 4)]
        neg = {z for z in neg if 0 <= z < hu.shape[2] and area[z] == 0}
        for z in sorted(pos | neg):
            img = square_fov(to_radiological(window_hu(hu[:, :, z], a.level, a.width)), (sy, sx), a.fov_mm, 0.0)
            outline = np.zeros(kid.shape[:2], dtype=bool)
            for m in sides:
                if m[:, :, z].sum() >= 10:
                    outline |= convex_hull_image(m[:, :, z])
            msk = square_fov(to_radiological(outline).astype(np.float32), (sy, sx), a.fov_mm, 0.0)
            name = f"{s.name}_z{z:04d}.png"
            Image.fromarray((img * 255).astype(np.uint8)).resize((a.size, a.size), Image.BILINEAR) \
                .save(TS2D_DIR / "images" / name)
            Image.fromarray(((msk > 0.5) * 255).astype(np.uint8)).resize((a.size, a.size), Image.NEAREST) \
                .save(TS2D_DIR / "masks" / name)
            rows.append({"subject": s.name, "z": z, "image": f"images/{name}", "mask": f"masks/{name}",
                         "has_kidney": int(z in pos)})
    write_csv(TS2D_DIR / "index.csv", rows)
    print(f"{len(rows)} slices from {len({r['subject'] for r in rows})} subjects -> {TS2D_DIR}")


if __name__ == "__main__":
    main()
