"""Grad-CAM (Selvaraju et al., 2017) on the attention-refined ClfNet features."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.clf = model.clf
        self._acts = None

    def _hook(self, _m, _i, out):
        self._acts = out
        if out.requires_grad:
            out.retain_grad()

    def __call__(self, x: torch.Tensor, target: int | None = None):
        """x: (1,3,H,W) in [0,1]. Returns (cam HxW in [0,1], model output dict, target class)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)       # frozen backbones still need a grad path
        handle = self.clf.cbam.register_forward_hook(self._hook)
        try:
            with torch.enable_grad():
                out = self.model(x)
                logits = out["logits"]
                if target is None:
                    target = int(logits.argmax(1))
                self.model.zero_grad(set_to_none=True)
                logits[0, target].backward()
        finally:
            handle.remove()
        acts, grads = self._acts, self._acts.grad
        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}, target


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """img: HxWx3 float [0,1]; cam: HxW [0,1] -> uint8 jet overlay."""
    import matplotlib.cm as cm
    heat = cm.jet(cam)[..., :3]
    return (np.clip((1 - alpha) * img + alpha * heat, 0, 1) * 255).astype(np.uint8)
