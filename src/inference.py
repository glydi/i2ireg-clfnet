"""Loading trained runs and single-image inference (used by evaluate.py and the app)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import IMG_SIZE, load_gray, to_tensor3
from .gradcam import GradCAM, overlay
from .models import build_model

ROOT = Path(__file__).resolve().parents[1]


def find_checkpoint(run: Path) -> tuple[Path, dict]:
    """run may be runs/<name>, runs/<name>/final or runs/<name>/fold_k."""
    run = Path(run)
    for d in (run, run / "final", *sorted(run.glob("fold_*"))):
        if (d / "best.pt").exists():
            cfg_path = d / "config.json" if (d / "config.json").exists() else d.parent / "config.json"
            return d / "best.pt", json.loads(cfg_path.read_text())
    raise FileNotFoundError(f"no best.pt under {run}")


def load_run(run, device="cpu"):
    ckpt, cfg = find_checkpoint(run)
    model = build_model(cfg["arch"], cfg["deep"], cfg["shallow"], cfg["cbam"], cfg["head"], pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval(), cfg, ckpt


def list_runs() -> list[str]:
    out = []
    for p in sorted((ROOT / "runs").glob("*/**/best.pt")):
        d = p.parent
        if (d / "config.json").exists() or (d.parent / "config.json").exists():
            out.append(str(d.relative_to(ROOT)))
    return out


def predict(model, image_path_or_array, device="cpu", with_cam=True):
    """Returns dict(prob_stone, pred, image, roi, cam_overlay)."""
    if isinstance(image_path_or_array, np.ndarray):
        from PIL import Image
        a = image_path_or_array
        if a.ndim == 3:
            a = a[..., :3].mean(-1)
        a = a.astype(np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8) if a.max() > 1 else a
        img = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE)),
                         dtype=np.float32) / 255
    else:
        img = load_gray(image_path_or_array)
    x = to_tensor3(img)[None].to(device)
    if with_cam:
        cam, out, cam_cls = GradCAM(model)(x)           # explains the predicted class
    else:
        with torch.no_grad():
            out = model(x)
        cam, cam_cls = None, None
    prob = float(out["logits"].softmax(1)[0, 1])
    roi = out["roi"][0].mean(0).clamp(0, 1).cpu().numpy() if out.get("roi") is not None else None
    base = np.repeat((roi if roi is not None else img)[..., None], 3, -1)
    return {
        "prob_stone": prob,
        "pred": int(prob >= 0.5),
        "image": img,
        "roi": roi,
        "cam": cam,
        "cam_class": cam_cls,
        "cam_overlay": overlay(base, cam) if cam is not None else None,
    }
