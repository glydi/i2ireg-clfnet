#!/usr/bin/env python
"""Step 5 - train I2IReg-ClfNet end-to-end (or a baseline) on KidneyStoneTR.

  --folds 1 2 3 4 5   five-fold patient-level cross-validation (official splits, "CV5")
  --final             train on all CV images, evaluate on the independent hold-out set

Examples
  python scripts/train.py --name proposed --folds 1 2 3 4 5 --final
  python scripts/train.py --name clfnet_only --arch full_image --final          # no ROI stage
  python scripts/train.py --name squeezenet --arch full_image --no-shallow --no-cbam --final
  python scripts/train.py --name seg_head --head segmentation --final          # ablation 3.4.3.2
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.engine import ROOT, evaluate, fit, loader, pick_device, seed_everything  # noqa: E402
from src.data import Augment, SliceDataset, kstr_rows  # noqa: E402
from src.models import build_model, count_params  # noqa: E402


def save_preds(path: Path, preds):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "pred", "prob"])
        w.writeheader()
        w.writerows(preds)


def make_model(cfg, dev):
    m = build_model(cfg["arch"], cfg["deep"], cfg["shallow"], cfg["cbam"], cfg["head"]).to(dev)
    init = cfg.get("init_reg")
    if cfg["arch"] == "i2ireg_clfnet" and init and Path(init).exists():
        sd = torch.load(init, map_location=dev)
        missing = m.load_state_dict({k: v for k, v in sd.items() if k.startswith("reg.")}, strict=False)
        print(f"  initialised I2IRegNet from {init} ({len(sd)} tensors; "
              f"{len([k for k in missing.missing_keys if k.startswith('reg.')])} reg keys missing)")
    return m


def run_split(cfg, train_rows, test_rows, out_dir: Path, dev, test_name: str):
    rng = random.Random(cfg["seed"])
    rows = train_rows[:]
    rng.shuffle(rows)
    if cfg["limit"]:
        rows = rows[:cfg["limit"]]
        test_rows = rng.sample(test_rows, min(len(test_rows), max(cfg["limit"] // 4, 16)))
    n_val = int(len(rows) * cfg["val_fraction"])
    val_rows, tr_rows = rows[:n_val], rows[n_val:]
    print(f"  train={len(tr_rows)} val={len(val_rows)} {test_name}={len(test_rows)} "
          f"(ROI masks on {sum(bool(r['mask']) for r in tr_rows)} train images)")
    aug = Augment() if cfg["augment"] else None
    tr_dl = loader(SliceDataset(tr_rows, aug), cfg["batch_size"], True, cfg["workers"])
    va_dl = loader(SliceDataset(val_rows), cfg["batch_size"], False, cfg["workers"]) if val_rows else None
    te_dl = loader(SliceDataset(test_rows), cfg["batch_size"], False, cfg["workers"])
    seed_everything(cfg["seed"])
    model = make_model(cfg, dev)
    tot, trn = count_params(model)
    print(f"  params: {tot / 1e6:.2f}M total, {trn / 1e6:.2f}M trainable")
    fit(model, tr_dl, va_dl, dev, cfg["epochs"], cfg["lr"], out_dir, cfg["w_reg"], cfg["w_clf"],
        cfg["head"], monitor="f1", mode="max")
    m, preds = evaluate(model, te_dl, dev, test_name)
    (out_dir / f"{test_name}_metrics.json").write_text(json.dumps(m, indent=1))
    save_preds(out_dir / f"{test_name}_preds.csv", preds)
    print(f"  {test_name}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))
    return m


def summarise(results: list[dict]) -> dict:
    keys = [k for k in results[0] if isinstance(results[0][k], float)]
    return {k: {"mean": float(np.mean([r[k] for r in results])), "std": float(np.std([r[k] for r in results]))}
            for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="proposed")
    ap.add_argument("--arch", choices=["i2ireg_clfnet", "full_image"], default="i2ireg_clfnet")
    ap.add_argument("--deep", choices=["squeezenet", "resnet18", "efficientnet_v2_s"], default="squeezenet")
    ap.add_argument("--no-shallow", action="store_true", help="single-encoder ClfNet")
    ap.add_argument("--no-cbam", action="store_true")
    ap.add_argument("--head", choices=["regression", "segmentation"], default="regression")
    ap.add_argument("--init-reg", default=None, help="I2IRegNet checkpoint (default runs/i2ireg/best.pt)")
    ap.add_argument("--folds", type=int, nargs="*", default=[])
    ap.add_argument("--final", action="store_true", help="train on all CV data, test on hold-out")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-reg", type=float, default=0.8)
    ap.add_argument("--w-clf", type=float, default=0.2)
    ap.add_argument("--val-fraction", type=float, default=0.1, help="random val split for LR schedule/checkpoint")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--keep-holdout-dups", action="store_true",
                    help="keep the 17 hold-out images that also appear in CV training folds")
    ap.add_argument("--limit", type=int, default=0, help="subsample for smoke tests")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.folds and not a.final:
        ap.error("give --folds and/or --final")

    head = a.head
    init = a.init_reg or str(ROOT / "runs" / ("i2ireg" if head == "regression" else "i2ireg_seg") / "best.pt")
    cfg = {"arch": a.arch, "deep": a.deep, "shallow": not a.no_shallow, "cbam": not a.no_cbam, "head": head,
           "init_reg": init, "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr, "w_reg": a.w_reg,
           "w_clf": a.w_clf, "val_fraction": a.val_fraction, "augment": not a.no_augment, "limit": a.limit,
           "workers": a.workers, "seed": a.seed}
    dev = pick_device(a.device)
    run_dir = ROOT / "runs" / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"run={run_dir} device={dev}\nconfig={cfg}")
    if a.arch == "i2ireg_clfnet" and not any(r["mask"] for r in kstr_rows("trainall")):
        print("WARNING: no KidneyStoneTR ROI masks found - run scripts/pseudo_label.py first "
              "(otherwise the regression loss is off and the ROI stage is unsupervised).")
    drop = not a.keep_holdout_dups

    cv = []
    for k in a.folds:
        print(f"\n=== fold {k} ===")
        cv.append(run_split(cfg, kstr_rows("train", k, drop), kstr_rows("test", k), run_dir / f"fold_{k}", dev,
                            "test"))
    if cv:
        s = summarise(cv)
        (run_dir / "cv_summary.json").write_text(json.dumps(s, indent=1))
        print("\nCV5 summary:\n" + "\n".join(f"  {k:14s} {v['mean']:.4f} ± {v['std']:.4f}" for k, v in s.items()))
    if a.final:
        print("\n=== final model (all CV data) -> hold-out ===")
        run_split(cfg, kstr_rows("trainall", 0, drop), kstr_rows("holdout"), run_dir / "final", dev, "holdout")


if __name__ == "__main__":
    main()
