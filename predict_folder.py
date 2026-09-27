#!/usr/bin/env python
"""Predict every image in a folder and report accuracy.

    python predict_folder.py /path/to/folder                 # uses the proposed model
    python predict_folder.py /path/to/folder --run runs/clfnet_only/final --thr 0.5

Labels come from sub-folder names: Positive/Stone/1 = stone, Negative/Normal/0 = no stone
(any nesting depth). Without labels you still get predictions (CSV), just no accuracy.
Supports PNG/JPG/BMP/TIF and DICOM (.dcm).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data import SliceDataset, folder_rows  # noqa: E402
from src.engine import pick_device  # noqa: E402
from src.inference import list_runs, load_run  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402


def default_run():
    runs = list_runs()
    for pref in ("runs/proposed/final", "runs/proposed"):
        for r in runs:
            if r.startswith(pref):
                return r
    return runs[0] if runs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--run", default=None, help="trained run dir (default: runs/proposed/final)")
    ap.add_argument("--thr", type=float, default=0.5, help="decision threshold on P(stone)")
    ap.add_argument("--out", default=None, help="CSV path (default: <folder>_predictions.csv here)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    run = a.run or default_run()
    if not run:
        sys.exit("No trained model in runs/. Train one first (README step 5).")
    dev = pick_device(a.device)
    model, cfg, ckpt = load_run(ROOT / run if not Path(run).is_absolute() else run, dev)
    rows, labelled = folder_rows(a.folder)
    if not rows:
        sys.exit("No images found.")
    print(f"model: {ckpt}\nimages: {len(rows)}  device: {dev}")

    probs = []
    with torch.no_grad():
        for b in tqdm(DataLoader(SliceDataset(rows), batch_size=16, num_workers=2), desc="predicting"):
            probs += model(b["image"].to(dev))["logits"].softmax(1)[:, 1].cpu().tolist()
    preds = [int(p >= a.thr) for p in probs]

    out = Path(a.out or f"{Path(a.folder).resolve().name}_predictions.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "pred", "prob_stone"])
        w.writerows([[r["path"], r["label"], q, round(p, 5)] for r, q, p in zip(rows, preds, probs)])
    print(f"predictions -> {out}")

    idx = [i for i, r in enumerate(rows) if r["label"] >= 0]
    if not idx:
        print(f"Predicted Stone+: {sum(preds)}  Stone-: {len(preds) - sum(preds)}  "
              "(no Positive/Negative sub-folders -> no accuracy)")
        return
    m = classification_metrics([rows[i]["label"] for i in idx], [preds[i] for i in idx], [probs[i] for i in idx])
    print(f"\nACCURACY: {100 * m['accuracy']:.2f}%   ({len(idx)} labelled images)")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, indent=1))


if __name__ == "__main__":
    main()
