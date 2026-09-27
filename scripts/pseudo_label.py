#!/usr/bin/env python
"""Step 4 - create kidney ROI masks for KidneyStoneTR with the pre-trained I2IRegNet.

The public KidneyStoneTR release does not include the radiologists' kidney masks, so the
regression targets of the cascade (ROI = mask * image) are built from pseudo-masks:
  I2IRegNet weighting map -> threshold -> keep the <=2 largest plausible blobs
  -> per-blob convex hull (outer kidney contour, like the radiologists' tracing).
Masks are stored as data/kidneystonetr/masks/<md5>.png (one per unique image) and can be
reviewed / rejected in the app ("Pseudo-mask review" tab). Rejected or empty masks simply
switch off the regression loss for that image.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.engine import ROOT, pick_device  # noqa: E402
from src.data import KSTR_DIR, load_gray, kstr_unique_rows, to_tensor3  # noqa: E402
from src.models import I2IRegNet  # noqa: E402


def clean_mask(prob: np.ndarray, thr: float, min_area: int, max_blobs: int = 2) -> np.ndarray:
    from scipy import ndimage as ndi
    from skimage.morphology import convex_hull_image
    m = ndi.binary_opening(prob > thr, iterations=1)
    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros_like(m)
    areas = ndi.sum(m, lab, range(1, n + 1))
    keep = [i + 1 for i in np.argsort(areas)[::-1][:max_blobs] if areas[i] >= min_area]
    out = np.zeros_like(m)
    for k in keep:
        out |= convex_hull_image(lab == k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "i2ireg" / "best.pt"))
    ap.add_argument("--thr", type=float, default=0.3, help="threshold on the regression weighting map")
    ap.add_argument("--min-area", type=int, default=60, help="min blob area (pixels @224)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    dev = pick_device(a.device)
    net = I2IRegNet(pretrained=False).to(dev).eval()
    sd = torch.load(a.ckpt, map_location=dev)
    net.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("reg.")})
    rows = kstr_unique_rows()
    out_dir = KSTR_DIR / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if a.overwrite or not (out_dir / f"{r['md5']}.png").exists()]
    print(f"{len(rows)} unique images, {len(todo)} to label, device={dev}")
    stats = {"empty": 0, "one_blob": 0, "two_blobs": 0}
    from scipy import ndimage as ndi
    with torch.no_grad():
        for i in tqdm(range(0, len(todo), a.batch_size), desc="pseudo-labelling"):
            chunk = todo[i:i + a.batch_size]
            x = torch.stack([to_tensor3(load_gray(r["path"])) for r in chunk]).to(dev)
            _, w = net(x)
            w = w.mean(1).cpu().numpy()
            for r, wm in zip(chunk, w):
                m = clean_mask(wm, a.thr, a.min_area)
                n = ndi.label(m)[1]
                stats["empty" if n == 0 else "one_blob" if n == 1 else "two_blobs"] += 1
                if n == 0:        # no confident kidney: leave unlabelled (no regression loss)
                    continue
                Image.fromarray((m * 255).astype(np.uint8)).save(out_dir / f"{r['md5']}.png")
    (KSTR_DIR / "pseudo_label_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()
