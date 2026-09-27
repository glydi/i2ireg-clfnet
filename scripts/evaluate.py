#!/usr/bin/env python
"""Step 6 - evaluate a trained run, compare runs statistically, export Grad-CAM figures.

  python scripts/evaluate.py --run runs/proposed --data holdout
  python scripts/evaluate.py --run runs/proposed --data ctkidney            # external, no fine-tuning
  python scripts/evaluate.py --run runs/proposed --data folder:/path/with/Positive_Negative
  python scripts/evaluate.py --mcnemar runs/proposed/final/holdout_preds.csv runs/clfnet_only/final/holdout_preds.csv
  python scripts/evaluate.py --run runs/proposed --data holdout --gradcam 12 --timing
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.engine import ROOT, evaluate, loader, pick_device  # noqa: E402
from src.data import SliceDataset, ctkidney_rows, folder_rows, kstr_rows  # noqa: E402
from src.inference import load_run, predict  # noqa: E402
from src.metrics import mcnemar  # noqa: E402


def rows_for(spec: str):
    if spec == "holdout":
        return kstr_rows("holdout")
    if spec.startswith("test:"):
        return kstr_rows("test", int(spec.split(":")[1]))
    if spec == "ctkidney":
        return ctkidney_rows()
    if spec.startswith("folder:"):
        rows, labelled = folder_rows(spec.split(":", 1)[1])
        if not labelled:
            print("note: some images have no Positive/Negative parent folder -> metrics use labelled ones only")
        return rows
    raise SystemExit(f"unknown --data {spec}")


def confusion_png(m: dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Negative", "Positive"])
    ax.set_yticks([0, 1], ["Negative", "Positive"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gradcam_panels(model, rows, n, dev, out: Path):
    from PIL import Image
    from src.gradcam import overlay
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pick = [rows[i] for i in rng.choice(len(rows), min(n, len(rows)), replace=False)]
    for i, r in enumerate(pick):
        p = predict(model, r["path"], dev)
        g = lambda a: (np.repeat(a[..., None], 3, -1) * 255).astype(np.uint8)
        tiles = [g(p["image"])]
        if p["roi"] is not None:
            tiles.append(g(p["roi"]))
        tiles += [p["cam_overlay"], overlay(np.repeat(p["image"][..., None], 3, -1), p["cam"])]
        name = f"{i:02d}_true{r['label']}_pred{p['pred']}_p{p['prob_stone']:.2f}.png"
        Image.fromarray(np.concatenate(tiles, 1)).save(out / name)
    print(f"Grad-CAM panels (image | ROI | CAM on ROI | CAM on image) -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run")
    ap.add_argument("--data", default="holdout")
    ap.add_argument("--mcnemar", nargs=2, metavar=("PREDS_A", "PREDS_B"))
    ap.add_argument("--gradcam", type=int, default=0, help="export N Grad-CAM panels")
    ap.add_argument("--timing", action="store_true", help="measure inference ms/slice")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    if a.mcnemar:
        load = lambda p: {r["path"]: r for r in csv.DictReader(open(p))}
        A, B = map(load, a.mcnemar)
        common = sorted(set(A) & set(B))
        res = mcnemar([int(A[k]["label"]) for k in common], [int(A[k]["pred"]) for k in common],
                      [int(B[k]["pred"]) for k in common])
        print(f"n={len(common)}  A={a.mcnemar[0]}\n            B={a.mcnemar[1]}")
        print(json.dumps(res, indent=1))
        return
    if not a.run:
        ap.error("--run is required")

    dev = pick_device(a.device)
    model, cfg, ckpt = load_run(ROOT / a.run if not Path(a.run).is_absolute() else a.run, dev)
    rows = rows_for(a.data)
    tag = a.data.replace(":", "_").replace("/", "_")[:60]
    out = ckpt.parent / f"eval_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint={ckpt}  images={len(rows)}  device={dev}")
    m, preds = evaluate(model, loader(SliceDataset(rows), a.batch_size, False), dev, a.data)
    if a.timing:
        x = torch.rand(1, 3, 224, 224, device=dev)
        with torch.no_grad():
            for _ in range(5):
                model(x)
            t = time.perf_counter()
            for _ in range(50):
                model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        m["inference_ms_per_slice"] = (time.perf_counter() - t) / 50 * 1000
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    with open(out / "preds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "pred", "prob"])
        w.writeheader()
        w.writerows(preds)
    if "tp" in m:
        confusion_png(m, out / "confusion.png", f"{Path(a.run).name} on {a.data}")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, indent=1))
    if a.gradcam:
        gradcam_panels(model, rows, a.gradcam, dev, out / "gradcam")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
