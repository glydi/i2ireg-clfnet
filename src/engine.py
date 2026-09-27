"""Training / evaluation loops shared by the scripts."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import RegressionMeter, classification_metrics

ROOT = Path(__file__).resolve().parents[1]
# keep ImageNet weights inside the project folder (self-contained)
os.environ.setdefault("TORCH_HOME", str(ROOT / "weights" / "torch_hub"))


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loader(ds, batch_size, shuffle, workers=None):
    workers = min(2, os.cpu_count() or 1) if workers is None else workers
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle and len(ds) > batch_size,
                      persistent_workers=workers > 0)


def half_mse(pred, target):
    """L_reg = 1/(2N) * sum (ROI - ROI_hat)^2   (Eq. 2.1)"""
    return 0.5 * F.mse_loss(pred, target)


def focal_bce_with_logits(logits, target, alpha=0.25, gamma=2.0):
    """Segmentation-head ablation loss (Sec. 3.4.3.2)."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    a = alpha * target + (1 - alpha) * (1 - target)
    return (a * (1 - pt) ** gamma * bce).mean()


def cascade_loss(out, batch, w_reg=0.8, w_clf=0.2, head="regression"):
    """L_total = w_reg * L_reg + w_clf * L_clf   (Eq. 2.2). Regression term only on images
    that have an ROI mask; classification term only on labelled images."""
    logs = {}
    total = 0.0
    if out["roi"] is not None and batch["has_mask"].sum() > 0:
        sel = batch["has_mask"] > 0
        if head == "regression":
            l_reg = half_mse(out["roi"][sel], batch["roi"][sel])
        else:
            l_reg = focal_bce_with_logits(out["raw"][sel], batch["mask"][sel])
        total = total + w_reg * l_reg
        logs["l_reg"] = l_reg.item()
    lab = batch["label"] >= 0
    if lab.any():
        l_clf = F.cross_entropy(out["logits"][lab], batch["label"][lab])
        total = total + w_clf * l_clf
        logs["l_clf"] = l_clf.item()
    return total, logs


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, dl, dev, desc="eval"):
    """Returns (metrics dict, per-image predictions list)."""
    model.eval()
    ys, ps, probs, paths = [], [], [], []
    head = getattr(getattr(model, "reg", None), "head", "regression")
    reg = RegressionMeter(thr=0.5 if head == "segmentation" else 0.1)
    for b in tqdm(dl, desc=desc, leave=False):
        b = to_dev(b, dev)
        out = model(b["image"])
        if "logits" in out and out["logits"] is not None:
            pr = out["logits"].softmax(1)[:, 1]
            probs += pr.cpu().tolist()
            ps += out["logits"].argmax(1).cpu().tolist()
            ys += b["label"].cpu().tolist()
        if out.get("roi") is not None and b["has_mask"].sum() > 0:
            sel = b["has_mask"] > 0
            reg.update(out["roi"][sel], b["roi"][sel], b["mask"][sel])
        paths += b["path"]
    m = {}
    if ys and min(ys) >= 0:
        m.update(classification_metrics(ys, ps, probs))
    m.update({f"reg_{k}": v for k, v in reg.result().items()})
    preds = [{"path": p, "label": y, "pred": q, "prob": r} for p, y, q, r in zip(paths, ys, ps, probs)] if ys else []
    return m, preds


def fit(model, train_dl, val_dl, dev, epochs, lr, out_dir: Path, w_reg=0.8, w_clf=0.2,
        head="regression", monitor="f1", mode="max", log_every=0):
    """Adam + ReduceLROnPlateau ('dynamic learning rate'); keeps the best checkpoint on val."""
    out_dir.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=mode, factor=0.5, patience=4)
    best = -float("inf") if mode == "max" else float("inf")
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, run = time.time(), {}
        n = 0
        for b in tqdm(train_dl, desc=f"epoch {ep}/{epochs}", leave=False):
            b = to_dev(b, dev)
            out = model(b["image"])
            loss, logs = cascade_loss(out, b, w_reg, w_clf, head)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n += 1
            for k, v in logs.items():
                run[k] = run.get(k, 0.0) + v
        tr = {k: v / max(n, 1) for k, v in run.items()}
        vm, _ = evaluate(model, val_dl, dev, "val") if val_dl is not None else ({}, None)
        score = vm.get(monitor, -tr.get("l_reg", 0) if mode == "max" else tr.get("l_reg", 0))
        sched.step(score)
        rec = {"epoch": ep, "time_s": round(time.time() - t0, 1), "lr": opt.param_groups[0]["lr"], **tr,
               **{f"val_{k}": v for k, v in vm.items() if isinstance(v, float)}}
        history.append(rec)
        improved = score > best if mode == "max" else score < best
        if improved:
            best = score
            torch.save(model.state_dict(), out_dir / "best.pt")
        torch.save(model.state_dict(), out_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
        msg = " ".join(f"{k}={v:.4f}" for k, v in rec.items() if isinstance(v, float) and k != "lr")
        print(f"[ep {ep:3d}] {msg} lr={rec['lr']:.1e}{'  *best*' if improved else ''}", flush=True)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev))
    return history
