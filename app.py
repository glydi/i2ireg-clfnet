#!/usr/bin/env python
"""I2IReg-ClfNet research UI.

    python app.py            # then open http://127.0.0.1:7860
    python app.py --share    # temporary public link (Gradio)

Tabs: Detect | Evaluate folder | ROI-aware vs full-image | Results | Pseudo-mask review | About
Research prototype - NOT for clinical use.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / "weights" / "torch_hub"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from src.data import KSTR_DIR, KSTR_RAW, kstr_unique_rows, load_gray, load_mask, window_hu  # noqa: E402
from src.engine import pick_device  # noqa: E402
from src.gradcam import overlay  # noqa: E402
from src.inference import list_runs, load_run, predict  # noqa: E402

DEV = pick_device()
_cache: dict[str, tuple] = {}


def get_model(run: str):
    if run not in _cache:
        _cache[run] = load_run(ROOT / run, DEV)
    return _cache[run]


def read_input(file_path):
    """PNG/JPG or DICOM (HU -> abdominal window L40/W400, matching the training data)."""
    if file_path is None:
        return None
    p = str(file_path)
    if p.lower().endswith((".dcm", ".dicom")) or not Path(p).suffix:
        import pydicom
        ds = pydicom.dcmread(p)
        hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1)) + \
            float(getattr(ds, "RescaleIntercept", 0))
        return window_hu(hu)
    return load_gray(p, 512)


def rgb(a):
    return (np.repeat(a[..., None], 3, -1) * 255).astype(np.uint8)


def examples():
    return [[str(p)] for p in sorted((ROOT / "samples").glob("*.jpg"))]


# ------------------------------------------------------------------------------ Detect tab
def detect(file_path, run, thr):
    if file_path is None:
        raise gr.Error("Upload a CT slice first.")
    if not run:
        raise gr.Error("No trained model found. Train one first (see the About tab / README).")
    model, cfg, _ = get_model(run)
    img = read_input(file_path)
    p = predict(model, img, DEV)
    prob = p["prob_stone"]
    verdict = "Stone +" if prob >= thr else "Stone −"
    cam_img = overlay(np.repeat(p["image"][..., None], 3, -1), p["cam"])
    roi = rgb(p["roi"]) if p["roi"] is not None else None
    explained = "Stone +" if p["cam_class"] == 1 else "Stone −"
    info = f"**{verdict}**  P(stone) = {prob:.3f}"
    return {"Stone +": prob, "Stone −": 1 - prob}, info, rgb(p["image"]), roi, p["cam_overlay"], cam_img


# ------------------------------------------------------------------------------ Compare tab
def compare(file_path, run_a, run_b):
    if file_path is None or not run_a or not run_b:
        raise gr.Error("Upload an image and choose two runs.")
    img = read_input(file_path)
    outs = []
    for run in (run_a, run_b):
        model, cfg, _ = get_model(run)
        p = predict(model, img, DEV)
        cam = overlay(np.repeat(p["image"][..., None], 3, -1), p["cam"])
        outs += [cam, f"**{run}** ({cfg['arch']}): P(stone) = {p['prob_stone']:.3f} → "
                      f"{'Stone +' if p['pred'] else 'Stone −'}"]
    return outs


# ------------------------------------------------------------------------------ Results tab
def results_table():
    rows = []
    for d in sorted((ROOT / "runs").glob("*")):
        if not d.is_dir():
            continue
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        cv = json.loads((d / "cv_summary.json").read_text()) if (d / "cv_summary.json").exists() else {}
        ho = d / "final" / "holdout_metrics.json"
        if not ho.exists():
            ho = d / "final" / "eval_holdout" / "metrics.json"
        ho = json.loads(ho.read_text()) if ho.exists() else {}
        val = d / "val_metrics.json"
        val = json.loads(val.read_text()) if val.exists() else {}
        if not (cv or ho or val):
            continue
        f = lambda x: f"{x:.3f}" if isinstance(x, float) else ""
        cvf = lambda k: f"{cv[k]['mean']:.3f}±{cv[k]['std']:.3f}" if k in cv else ""
        rows.append([d.name, cfg.get("arch", "i2ireg (stage 1)"), cvf("accuracy"), cvf("recall"), cvf("f1"),
                     cvf("kappa"), f(ho.get("accuracy")), f(ho.get("recall")), f(ho.get("specificity")),
                     f(ho.get("f1")), f(ho.get("kappa")), f(ho.get("roc_auc")),
                     f(ho.get("reg_iou", val.get("reg_iou"))), f(ho.get("reg_mse", val.get("reg_mse")))])
    return rows


RESULT_COLS = ["run", "arch", "CV acc", "CV recall", "CV F1", "CV kappa", "HO acc", "HO recall",
               "HO spec", "HO F1", "HO kappa", "HO AUC", "IoU", "MSE"]


def history_plot(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not run_dir:
        return None
    hist = [h for h in sorted((ROOT / "runs" / run_dir).rglob("history.json"))]
    if not hist:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for h in hist:
        H = json.loads(h.read_text())
        ep = [r["epoch"] for r in H]
        lab = h.parent.name
        for k, ax in (("l_reg", axes[0]), ("l_clf", axes[0])):
            if k in H[0]:
                ax.plot(ep, [r[k] for r in H], label=f"{lab} {k}")
        for k in ("val_f1", "val_reg_iou"):
            if k in H[0]:
                axes[1].plot(ep, [r[k] for r in H], label=f"{lab} {k}")
    axes[0].set_title("training loss")
    axes[1].set_title("validation")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------ Folder tab
def evaluate_folder(folder_path, uploaded, run, thr, progress=gr.Progress()):
    """Batch prediction on a folder. Labels come from parent folder names
    (Positive/Stone/1 vs Negative/Normal/0); with labels, accuracy and the paper's metrics are reported."""
    import csv
    import tempfile

    import torch
    from torch.utils.data import DataLoader

    from src.data import IMG_EXT, NEG_NAMES, POS_NAMES, SliceDataset, folder_rows
    from src.metrics import classification_metrics

    if not run:
        raise gr.Error("No trained model found. Train one first.")
    if folder_path and folder_path.strip():
        try:
            rows, _ = folder_rows(folder_path.strip())
        except FileNotFoundError as e:
            raise gr.Error(str(e))
    elif uploaded:
        rows = []
        for p in uploaded:
            p = Path(p)
            if p.suffix.lower() not in IMG_EXT:
                continue
            parts = [x.lower() for x in p.parts[:-1]]
            lab = next((1 if x in POS_NAMES else 0 for x in reversed(parts) if x in POS_NAMES | NEG_NAMES), -1)
            rows.append({"path": str(p), "label": lab, "mask": ""})
    else:
        raise gr.Error("Enter a folder path or upload a folder.")
    if not rows:
        raise gr.Error("No images (png/jpg/bmp/tif/dcm) found.")

    model, cfg, _ = get_model(run)
    dl = DataLoader(SliceDataset(rows), batch_size=16, shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for b in progress.tqdm(dl, desc="predicting"):
            probs += model(b["image"].to(DEV))["logits"].softmax(1)[:, 1].cpu().tolist()
    preds = [int(p >= thr) for p in probs]
    table = [[Path(r["path"]).name, {1: "Stone +", 0: "Stone −"}.get(r["label"], "?"),
              "Stone +" if q else "Stone −", round(p, 4),
              "" if r["label"] < 0 else ("yes" if q == r["label"] else "no")]
             for r, q, p in zip(rows, preds, probs)]
    out_csv = Path(tempfile.gettempdir()) / "i2ireg_folder_predictions.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "pred", "prob_stone"])
        w.writerows([[r["path"], r["label"], q, p] for r, q, p in zip(rows, preds, probs)])

    lab_idx = [i for i, r in enumerate(rows) if r["label"] >= 0]
    n_pos = sum(preds)
    if not lab_idx:
        md = (f"**{len(rows)} images** · predicted Stone +: **{n_pos}**, Stone −: **{len(rows) - n_pos}**\n\n"
              "No labels found, so accuracy can't be computed. To get accuracy, put the images in sub-folders named "
              "`Positive`/`Negative` (or `Stone`/`Normal`).")
        return md, None, table, str(out_csv)
    y = [rows[i]["label"] for i in lab_idx]
    m = classification_metrics(y, [preds[i] for i in lab_idx], [probs[i] for i in lab_idx])
    pct = lambda k: f"{100 * m[k]:.2f}%" if k in m else "n/a"
    md = (f"## Accuracy: {pct('accuracy')}\n"
          f"{len(lab_idx)} labelled images ({sum(y)} Stone +, {len(y) - sum(y)} Stone −)"
          f"{f', {len(rows) - len(lab_idx)} unlabelled ignored' if len(lab_idx) < len(rows) else ''} · "
          f"model `{run}` · threshold {thr:.2f}\n\n"
          "| Recall (sens.) | Specificity | Precision | F1 | Cohen κ | ROC-AUC |\n|---|---|---|---|---|---|\n"
          f"| {pct('recall')} | {pct('specificity')} | {pct('precision')} | {pct('f1')} | {m['kappa']:.3f} | "
          f"{pct('roc_auc')} |\n\nTP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
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
    fig.tight_layout()
    return md, fig, table, str(out_csv)


# ------------------------------------------------------------------------------ Review tab
def review_items():
    if not (KSTR_DIR / "masks").exists():
        return []
    return [r for r in kstr_unique_rows() if (KSTR_DIR / "masks" / f"{r['md5']}.png").exists()]


def review_show(i):
    items = review_items()
    if not items:
        return None, "No pseudo-masks yet. Run `python scripts/pseudo_label.py`.", 0
    i = int(max(0, min(i, len(items) - 1)))
    r = items[i]
    img = load_gray(r["path"])
    m = load_mask(KSTR_DIR / "masks" / f"{r['md5']}.png") > 0
    vis = rgb(img)
    vis[m, 0] = np.clip(vis[m, 0].astype(int) + 90, 0, 255)
    return vis, f"{i + 1}/{len(items)} · label={'Stone +' if r['label'] else 'Stone −'} · `{Path(r['path']).name}`", i


def review_reject(i):
    items = review_items()
    if items:
        r = items[int(i)]
        rej = KSTR_DIR / "masks_rejected"
        rej.mkdir(exist_ok=True)
        shutil.move(KSTR_DIR / "masks" / f"{r['md5']}.png", rej / f"{r['md5']}.png")
    return review_show(i)


# ------------------------------------------------------------------------------ UI
def build():
    runs = list_runs()
    roi_runs = [r for r in runs if (get_cfg(r) or {}).get("arch") == "i2ireg_clfnet"] or runs
    with gr.Blocks(title="I2IReg-ClfNet · kidney stone detection") as demo:
        gr.Markdown("## I2IReg-ClfNet kidney stone detection")
        with gr.Tab("Detect"):
            with gr.Row():
                with gr.Column(scale=1):
                    f = gr.Image(label="CT slice (click a sample below or upload)", type="filepath", height=300)
                    run = gr.Dropdown(runs, value=roi_runs[0] if roi_runs else None, label="Model run",
                                      allow_custom_value=False)
                    thr = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="Decision threshold")
                    btn = gr.Button("Detect", variant="primary")
                    ex = examples()
                    if ex:
                        gr.Examples(ex, inputs=[f], label="Samples (stone_* / normal_*)", examples_per_page=8)
                with gr.Column(scale=2):
                    lab = gr.Label(label="Prediction", num_top_classes=2)
                    info = gr.Markdown()
                    with gr.Row():
                        o1 = gr.Image(label="Input (224×224)", height=260)
                        o2 = gr.Image(label="Predicted kidney ROI (I2IRegNet)", height=260)
                    with gr.Row():
                        o3 = gr.Image(label="Grad-CAM on ClfNet input (ROI)", height=260)
                        o4 = gr.Image(label="Grad-CAM on original slice", height=260)
            btn.click(detect, [f, run, thr], [lab, info, o1, o2, o3, o4])
        with gr.Tab("Evaluate folder"):
            with gr.Row():
                with gr.Column(scale=1):
                    fpath = gr.Textbox(label="Folder path (sub-folders Positive/Negative)",
                                       placeholder=str(KSTR_RAW / "holdout") if KSTR_RAW.exists() else "/path/to/folder")
                    fup = gr.File(label="…or upload a folder", file_count="directory", type="filepath")
                    frun = gr.Dropdown(runs, value=roi_runs[0] if roi_runs else None, label="Model run")
                    fthr = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="Decision threshold")
                    fbtn = gr.Button("Evaluate", variant="primary")
                with gr.Column(scale=2):
                    fmd = gr.Markdown()
                    fplot = gr.Plot(label="Confusion matrix")
            ftab = gr.Dataframe(headers=["file", "true", "predicted", "P(stone)", "correct"], label="Per-image results")
            fcsv = gr.File(label="Download predictions (CSV)")
            fbtn.click(evaluate_folder, [fpath, fup, frun, fthr], [fmd, fplot, ftab, fcsv])
        with gr.Tab("ROI-aware vs full-image"):
            with gr.Row():
                cf = gr.Image(label="CT slice", type="filepath", height=300)
                ra = gr.Dropdown(runs, value=roi_runs[0] if roi_runs else None, label="Run A")
                rb = gr.Dropdown(runs, value=next((r for r in runs if r not in roi_runs), None), label="Run B")
            cbtn = gr.Button("Compare", variant="primary")
            with gr.Row():
                with gr.Column():
                    ia = gr.Image(label="A", height=300)
                    ta = gr.Markdown()
                with gr.Column():
                    ib = gr.Image(label="B", height=300)
                    tb = gr.Markdown()
            cbtn.click(compare, [cf, ra, rb], [ia, ta, ib, tb])
        with gr.Tab("Results"):
            tbl = gr.Dataframe(results_table(), headers=RESULT_COLS, label="CV5 (mean±std) and hold-out (HO)")
            with gr.Row():
                rsel = gr.Dropdown(sorted({r.split("/")[1] for r in runs}) if runs else [], label="Training curves")
                refresh = gr.Button("Refresh")
            plot = gr.Plot()
            rsel.change(history_plot, rsel, plot)
            refresh.click(lambda: results_table(), None, tbl)
        with gr.Tab("Pseudo-mask review"):
            idx = gr.Number(0, visible=False, precision=0)
            with gr.Row():
                prev_b, next_b, rej_b = gr.Button("Prev"), gr.Button("Next"), gr.Button("Reject mask", variant="stop")
            rimg = gr.Image(label="Image + pseudo-mask (red)", height=420)
            rtxt = gr.Markdown()
            demo.load(review_show, idx, [rimg, rtxt, idx])
            prev_b.click(lambda i: review_show(i - 1), idx, [rimg, rtxt, idx])
            next_b.click(lambda i: review_show(i + 1), idx, [rimg, rtxt, idx])
            rej_b.click(review_reject, idx, [rimg, rtxt, idx])
    return demo


def get_cfg(run):
    for p in (ROOT / run / "config.json", (ROOT / run).parent / "config.json"):
        if p.exists():
            return json.loads(p.read_text())
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    print(f"device: {DEV}")
    theme = gr.themes.Base(radius_size=gr.themes.sizes.radius_none, font=["system-ui", "sans-serif"])
    css = "* { border-radius: 0 !important; } footer { display: none !important; }"
    demo = build()
    try:                       # Gradio >= 6: theme/css are launch() arguments
        demo.launch(server_name=a.host, server_port=a.port, share=a.share, theme=theme, css=css)
    except TypeError:          # Gradio 4/5
        demo.theme, demo.css = theme, css
        demo.launch(server_name=a.host, server_port=a.port, share=a.share)
