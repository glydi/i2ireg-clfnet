"""Evaluation metrics used in the paper (Eqs. 3.1-3.4) plus McNemar's test."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a) / b if b else 0.0
    rec, prec = div(tp, tp + fn), div(tp, tp + fp)
    out = {
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "recall": rec,
        "specificity": div(tn, tn + fp),
        "precision": prec,
        "f1": div(2 * prec * rec, prec + rec),
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    if y_prob is not None and len(set(y_true.tolist())) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


class RegressionMeter:
    """Accumulates MSE / RMSE / MAE (Eq. 3.3) and IoU / Dice (Eq. 3.4) of ROI estimates.
    IoU/Dice binarise both ROI images at `thr` (pixel intensity in [0,1])."""

    def __init__(self, thr: float = 0.1):
        self.thr = thr
        self.se = self.ae = self.n = 0.0
        self.tp = self.fp = self.fn = 0.0

    def update(self, pred, gt, gt_mask=None):
        pred, gt = pred.detach().float(), gt.detach().float()
        self.se += ((pred - gt) ** 2).sum().item()
        self.ae += (pred - gt).abs().sum().item()
        self.n += gt.numel()
        pm = pred.mean(1) > self.thr
        gm = (gt_mask[:, 0] > 0.5) if gt_mask is not None else gt.mean(1) > self.thr
        self.tp += (pm & gm).sum().item()
        self.fp += (pm & ~gm).sum().item()
        self.fn += (~pm & gm).sum().item()

    def result(self) -> dict:
        if not self.n:
            return {}
        mse = self.se / self.n
        return {
            "mse": mse, "rmse": math.sqrt(mse), "mae": self.ae / self.n,
            "iou": self.tp / max(self.tp + self.fp + self.fn, 1),
            "dice": 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1),
        }


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemar's test with continuity correction. b = A right & B wrong, c = A wrong & B right."""
    y_true, a, b_ = map(np.asarray, (y_true, pred_a, pred_b))
    ra, rb = a == y_true, b_ == y_true
    b = int((ra & ~rb).sum())
    c = int((~ra & rb).sum())
    both = int((ra & rb).sum())
    neither = int((~ra & ~rb).sum())
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    p = math.erfc(math.sqrt(chi2 / 2))           # survival fn of chi-square(1)
    phi = math.sqrt(chi2 / (b + c)) if (b + c) else 0.0
    return {"both_correct": both, "a_only": b, "b_only": c, "both_wrong": neither,
            "chi2": chi2, "p_value": p, "phi": phi}
