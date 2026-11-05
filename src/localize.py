"""Approximate stone localisation: bounding boxes from the Grad-CAM map.

No stone-level labels exist in KidneyStoneTR, so this is weakly supervised: the boxes mark where the
classifier looked, not a segmentation of the stone. Treat them as a hint.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def localize(cam: np.ndarray, roi: np.ndarray | None = None, rel_thr: float = 0.6,
             min_area: int = 12, max_boxes: int = 3) -> list[dict]:
    """cam: HxW in [0,1]; roi: optional HxW predicted kidney image (non-zero inside kidneys).

    Returns boxes sorted by score: dict(box=(x0,y0,x1,y1), center=(x,y), score, side).
    `side` is the patient side assuming radiological display (image left = patient right).
    """
    h, w = cam.shape
    c = cam.copy()
    if roi is not None:
        c = c * ndi.binary_dilation(roi > 0.05, iterations=3)
    if c.max() <= 0:
        return []
    c = c / c.max()
    lab, n = ndi.label(c >= rel_thr)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < min_area:
            continue
        ys, xs = np.where(m)
        cy, cx = ndi.center_of_mass(c * m)
        out.append({"box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    "center": (float(cx), float(cy)),
                    "score": float(c[m].mean()),
                    "side": "right kidney" if cx < w / 2 else "left kidney"})
    out.sort(key=lambda d: -d["score"])
    return out[:max_boxes]


def draw_boxes(img: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """img: HxWx3 uint8 or float [0,1] -> uint8 with red boxes and centre crosses."""
    a = img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8)
    a = a.copy()
    h, w = a.shape[:2]
    for b in boxes:
        x0, y0, x1, y1 = b["box"]
        x0, y0, x1, y1 = max(x0 - 2, 0), max(y0 - 2, 0), min(x1 + 2, w - 1), min(y1 + 2, h - 1)
        for t in (0, 1):
            a[y0 + t:y1 - t + 1, [x0 + t, x1 - t]] = (255, 0, 0)
            a[[y0 + t, y1 - t], x0 + t:x1 - t + 1] = (255, 0, 0)
        cx, cy = int(b["center"][0]), int(b["center"][1])
        a[cy, max(cx - 3, 0):cx + 4] = (255, 255, 0)
        a[max(cy - 3, 0):cy + 4, cx] = (255, 255, 0)
    return a
