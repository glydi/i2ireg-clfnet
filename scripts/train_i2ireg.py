#!/usr/bin/env python
"""Step 3 - pre-train the image-to-image regression subnetwork (I2IRegNet) on the
TotalSegmentator kidney slices (Sec. 3.4.1: half-MSE, Adam, lr 1e-4).

Output: runs/i2ireg/best.pt  (used for pseudo-labelling and to initialise the cascade)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.engine import ROOT, fit, evaluate, loader, pick_device, seed_everything  # noqa: E402
from src.data import Augment, SliceDataset, ts2d_rows  # noqa: E402
from src.models import I2IRegNet, count_params  # noqa: E402


class RegOnly(nn.Module):
    uses_roi = True

    def __init__(self, head="regression"):
        super().__init__()
        self.reg = I2IRegNet(pretrained=True, head=head)

    def forward(self, x):
        roi, raw = self.reg(x)
        return {"logits": None, "roi": roi, "raw": raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head", choices=["regression", "segmentation"], default="regression")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seed_everything(a.seed)
    dev = pick_device(a.device)
    tr, va = ts2d_rows()
    print(f"device={dev}  train slices={len(tr)}  val slices={len(va)}")
    # wide scale range: KidneyStoneTR bodies often sit smaller in the FOV than TotalSegmentator ones
    aug = Augment(rot=12, scale=(0.6, 1.15), shift=0.1)
    tr_dl = loader(SliceDataset(tr, aug), a.batch_size, True, a.workers)
    va_dl = loader(SliceDataset(va), a.batch_size, False, a.workers)
    model = RegOnly(a.head).to(dev)
    print("I2IRegNet params: %.2fM" % (count_params(model)[0] / 1e6))
    out = Path(a.out) if a.out else ROOT / "runs" / ("i2ireg" if a.head == "regression" else "i2ireg_seg")
    fit(model, tr_dl, va_dl, dev, a.epochs, a.lr, out, w_reg=1.0, w_clf=0.0, head=a.head,
        monitor="reg_iou", mode="max")
    m, _ = evaluate(model, va_dl, dev)
    (out / "val_metrics.json").write_text(json.dumps(m, indent=1))
    print("validation:", json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
