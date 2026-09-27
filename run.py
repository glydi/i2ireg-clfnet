#!/usr/bin/env python
"""Cross-platform launcher menu (started by start.sh / start.bat).

    python run.py              interactive menu
    python run.py check        quick check that the algorithm works
    python run.py ui           open the web UI
    python run.py download     download datasets
    python run.py quick        prepare data + train a basic model (~2-3 h on CPU, ~15 min on GPU)
    python run.py full         full paper pipeline (GPU recommended)
    python run.py folder PATH  accuracy on a folder of images
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def sh(*args):
    print("\n>>", " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([PY, *map(str, args)], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"step failed (exit {r.returncode}): {' '.join(map(str, args))}")


def have_data():
    return (ROOT / "data/raw/KidneyStoneTR/Dataset/dataset_integrated_task/holdout").exists()


def have_model():
    return any((ROOT / "runs").glob("*/final/best.pt")) or any((ROOT / "runs").glob("*/fold_*/best.pt"))


def download(subjects=40):
    sh("scripts/download_data.py", "kidneystonetr", "totalseg", "--max-subjects", subjects)


def prepare(roi_epochs):
    if not (ROOT / "data/totalseg_2d/index.csv").exists():
        sh("scripts/prepare_roi_source.py")
    if not (ROOT / "runs/i2ireg/best.pt").exists():
        sh("scripts/train_i2ireg.py", "--epochs", roi_epochs)
    if not (ROOT / "data/kidneystonetr/masks").exists():
        sh("scripts/pseudo_label.py")


def quick():
    download(20)
    prepare(12)
    sh("scripts/train.py", "--name", "proposed", "--final", "--epochs", 4, "--limit", 1000)


def full():
    download(40)
    prepare(30)
    sh("scripts/train.py", "--name", "proposed", "--folds", 1, 2, 3, 4, 5, "--final")
    sh("scripts/train.py", "--name", "clfnet_only", "--arch", "full_image", "--final")
    sh("scripts/evaluate.py", "--run", "runs/proposed", "--data", "holdout", "--gradcam", 12, "--timing")
    sh("scripts/evaluate.py", "--mcnemar", "runs/proposed/final/holdout_preds.csv",
       "runs/clfnet_only/final/holdout_preds.csv")


def folder(path=None):
    path = path or input("Folder with images (sub-folders Positive/Negative give labels): ").strip().strip('"')
    sh("predict_folder.py", path)


def ui():
    if not have_model():
        print("NOTE: no trained model yet - the UI opens, but Detect/Evaluate need a model (menu option 3).")
    sh("app.py")


def check():
    sh("check.py")


MENU = [("Check the algorithm works (about 2 min)", check),
        ("Open the web UI (http://127.0.0.1:7860)", ui),
        ("Download datasets", download),
        ("Train a basic model (~2-3 h on CPU, ~15 min on a GPU)", quick),
        ("Full paper pipeline: CV5 + hold-out + baseline (GPU recommended)", full),
        ("Accuracy on a folder of images", folder)]


def main():
    cmds = {"check": check, "ui": ui, "download": download, "quick": quick, "full": full, "folder": folder}
    if len(sys.argv) > 1:
        fn = cmds.get(sys.argv[1])
        if not fn:
            sys.exit(__doc__)
        return fn(*sys.argv[2:])
    print("\n  I2IReg-ClfNet · kidney stone detection  (research use only)")
    print(f"  data: {'ready' if have_data() else 'not downloaded'} · model: {'trained' if have_model() else 'none yet'}\n")
    for i, (label, _) in enumerate(MENU, 1):
        print(f"  [{i}] {label}")
    print("  [q] Quit")
    c = input("\nChoose: ").strip().lower()
    if c.isdigit() and 1 <= int(c) <= len(MENU):
        MENU[int(c) - 1][1]()


if __name__ == "__main__":
    main()
