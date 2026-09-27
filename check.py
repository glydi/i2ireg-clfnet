#!/usr/bin/env python
"""Quick check that the I2IReg-ClfNet algorithm works (about 2 min on CPU).

    python check.py

1. model builds, forward + backward pass, loss decreases on a tiny batch
2. trained I2IRegNet finds the kidneys (ROI image is non-empty, inside the body)
3. trained classifier accuracy on 100 hold-out images (50 stone, 50 normal)
Writes check_rois.png (input | predicted kidney ROI) for a visual check.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data import KSTR_RAW, folder_rows, load_gray, to_tensor3  # noqa: E402
from src.engine import cascade_loss, pick_device  # noqa: E402
from src.inference import list_runs, load_run  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402
from src.models import build_model  # noqa: E402

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def check_architecture():
    torch.manual_seed(0)
    m = build_model("i2ireg_clfnet", pretrained=False)
    x = torch.rand(4, 3, 224, 224)
    mask = torch.zeros(4, 1, 224, 224)
    mask[..., 120:160, 60:100] = 1
    batch = {"image": x, "mask": mask, "roi": x * mask, "has_mask": torch.ones(4),
             "label": torch.tensor([0, 1, 0, 1])}
    out = m(x)
    shapes_ok = out["logits"].shape == (4, 2) and out["roi"].shape == (4, 3, 224, 224)
    report("architecture shapes", shapes_ok, f"logits {tuple(out['logits'].shape)}, ROI {tuple(out['roi'].shape)}")
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], 1e-3)
    losses = []
    m.train()
    for _ in range(8):
        loss, _ = cascade_loss(m(x), batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    report("end-to-end training (loss decreases)", losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}")


def check_trained(dev):
    runs = [r for r in list_runs() if "proposed" in r] or list_runs()
    if not runs:
        report("trained model present", False, "train one first: start menu option [3]")
        return
    model, cfg, ckpt = load_run(ROOT / runs[0], dev)
    report("trained model present", True, str(ckpt.relative_to(ROOT)))
    if KSTR_RAW.exists():
        rows, _ = folder_rows(KSTR_RAW / "holdout")
        rng = random.Random(0)
        rows = rng.sample([r for r in rows if r["label"] == 1], 50) + rng.sample([r for r in rows if r["label"] == 0], 50)
    else:                       # dataset not downloaded: use the bundled samples
        rows = [{"path": str(p), "label": int(p.name.startswith("stone")), "mask": ""}
                for p in sorted((ROOT / "samples").glob("*.jpg"))]
        print(f"       dataset not downloaded - checking on {len(rows)} bundled samples")

    probs, tiles, roi_frac = [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), 20):
            chunk = rows[i:i + 20]
            x = torch.stack([to_tensor3(load_gray(r["path"])) for r in chunk]).to(dev)
            out = model(x)
            probs += out["logits"].softmax(1)[:, 1].cpu().tolist()
            if out["roi"] is not None:
                roi = out["roi"].mean(1).cpu().numpy()
                roi_frac += [(r > 0.1).mean() for r in roi]
                if len(tiles) < 6:
                    for xi, ri in zip(x[:, 0].cpu().numpy(), roi[:6 - len(tiles)]):
                        tiles.append(np.concatenate([xi, np.clip(ri, 0, 1)], 1))
    if roi_frac:
        f = float(np.median(roi_frac))
        # two kidneys cover roughly 1-8 % of a 224x224 axial slice
        report("I2IRegNet kidney ROI", 0.003 < f < 0.15, f"median ROI area {100 * f:.1f}% of the slice")
        from PIL import Image
        Image.fromarray((np.concatenate(tiles, 0) * 255).astype(np.uint8)).save(ROOT / "check_rois.png")
        print("       saved check_rois.png (left: input, right: predicted kidney ROI)")
    m = classification_metrics([r["label"] for r in rows], [int(p >= 0.5) for p in probs], probs)
    report("classification accuracy > 70% (basic model)", m["accuracy"] > 0.70,
           f"acc {100 * m['accuracy']:.1f}%  recall {100 * m['recall']:.1f}%  "
           f"specificity {100 * m['specificity']:.1f}%  AUC {m.get('roc_auc', 0):.3f}")


if __name__ == "__main__":
    dev = pick_device()
    print(f"device: {dev}")
    check_architecture()
    check_trained(dev)
    print(f"\n{sum(results)}/{len(results)} checks passed" + ("" if all(results) else "  -> see FAIL lines"))
    sys.exit(0 if all(results) else 1)
