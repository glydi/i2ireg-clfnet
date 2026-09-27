# I2IReg–ClfNet: ROI-aware kidney stone detection on non-contrast CT

A self-contained research reproduction of

> C. Öksüz, A. Narter, B. Ece, M. Koyun, İ. Taşkent, M. K. Güllü,
> **"I2IReg–ClfNet: a cascaded multi-task deep learning framework for ROI-aware kidney stone detection in abdominal CT images"**,
> *Biomedical Signal Processing and Control* 119 (2026) 109857. https://doi.org/10.1016/j.bspc.2026.109857

It includes PyTorch code, data download and preparation, training, evaluation, and a web UI.
Everything (virtual env, datasets, ImageNet weights, checkpoints) stays inside this folder.

> Note: this is a research prototype. It is **not** a medical device and must not be used for clinical decisions.

---

## 1. The idea

```
CT slice ──► I2IRegNet (DeepLabv3+, ResNet18) ──► ROI_hat = W(x) ⊙ x ──► ClfNet ──► Stone + / Stone −
                         │                                   (kidneys only)        │
                  L_reg = ½·MSE(ROI, ROI_hat)                               L_clf = CE
                                   L_total = 0.8·L_reg + 0.2·L_clf   (trained end-to-end)
```

* **I2IRegNet** regresses an image of the kidney region rather than a binary mask. Pixels inside the ROI
  keep their original intensities, so stones stay visible. Everything else is suppressed, including
  calcifications in the gallbladder, liver or vessels, which fool whole-slice classifiers.
* **ClfNet** is a dual encoder. A trainable shallow CNN (kernel counts 128/32/96/64/256, from the
  paper's Bayesian search) runs alongside a *frozen* ImageNet SqueezeNet. The two are fused at 7×7, then
  BN → 1×1 conv → **CBAM** → ReLU → Dropout → GAP → FC → softmax.
* **Grad-CAM** on the CBAM output shows *why* a slice was called stone-positive.

## 2. Datasets

| role | dataset | licence | size | why |
|---|---|---|---|---|
| **main** (train / CV5 / hold-out) | **KidneyStoneTR**, the paper's own NCCT dataset, [figshare 10.6084/m9.figshare.31156783](https://doi.org/10.6084/m9.figshare.31156783) | CC BY 4.0 | 550 MB | Same data, same official patient-level splits, so results compare directly with the paper |
| kidney-ROI ground truth | **TotalSegmentator** small subset, [zenodo 10047263](https://zenodo.org/records/10047263) | CC BY 4.0 | ~1.3 GB (40 subjects; only CT + kidney masks are fetched) | The public KidneyStoneTR release **does not include the radiologists' kidney masks**, and the regression stage needs them |
| optional external test | CT KIDNEY DATASET (Islam et al.), [Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) | see Kaggle | 1.5 GB | The external set used in the paper (Sec. 3.4.4.2) |

**How the missing masks are handled (the main deviation from the paper):**
1. Axial slices and kidney masks are extracted from TotalSegmentator (abdominal window L40/W400, radiological
   orientation, so they look like KidneyStoneTR). Each kidney is replaced by its convex hull, which matches the
   radiologists' outer-contour tracing and keeps the renal pelvis, where stones sit, inside the ROI.
2. I2IRegNet is pre-trained on these slices.
3. It then pseudo-labels every KidneyStoneTR slice, giving the kidney ROI targets. You can review and
   reject masks in the UI.
4. The full cascade is trained end-to-end on KidneyStoneTR, as in the paper.

Regression metrics (IoU, MSE) on KidneyStoneTR are therefore measured against *pseudo* ground truth, so they
are not directly comparable to the paper's 83% IoU. Classification metrics use the real expert labels.

Data notes found while building this:
* Training folds are not exactly the union of the other test folds (5,039 unique images vs 4,184). The
  official folders are used as-is.
* 17 hold-out images also appear in CV training folds 1, 3, 4 and 5. They are **removed from training by
  default** to avoid leakage (`--keep-holdout-dups` turns this off).

## 0. Quick start (one click)

| | first time | afterwards |
|---|---|---|
| **Windows** | install [Python 3.10+](https://www.python.org/downloads/) (tick *Add python.exe to PATH*) and [7-Zip](https://www.7-zip.org), then double-click **`setup.bat`** | double-click **`start.bat`** |
| **Linux / macOS** | `sudo apt install unar` (Fedora: `dnf`, macOS: `brew`), then `./setup.sh` | `./start.sh` |

`start.*` also runs the setup automatically if it hasn't been done. It then shows a menu:

```
[1] Open the web UI (http://127.0.0.1:7860)
[2] Download datasets
[3] Train a basic model (~2-3 h on CPU, ~15 min on a GPU)
[4] Full paper pipeline: CV5 + hold-out + baseline (GPU recommended)
[5] Accuracy on a folder of images
```
You can skip the menu: `start.bat folder D:\ct\test`, `./start.sh ui`, `./start.sh quick`.
A typical first session is **[3]** (downloads ~1 GB and trains: ~2–3 h on CPU, much faster on a GPU) then **[1]** or **[5]**.

## 3. Install

Requirements: Python ≥ 3.10, ~6 GB free disk, and a RAR extractor (`unar`, `7z`, `unrar` or `bsdtar`) for the
KidneyStoneTR archive. An NVIDIA GPU is strongly recommended for training. CPU works but is slow
(the UI and inference run fine on CPU).

```bash
cd i2ireg_clfnet
./setup.sh                 # creates .venv, installs PyTorch (CUDA if nvidia-smi found, else CPU) + requirements
source .venv/bin/activate
```
Windows: `setup.bat` (or `setup.bat cpu`), then use `.venv\Scripts\python.exe` for the scripts below.

## 4. Download data

```bash
python scripts/download_data.py --all            # KidneyStoneTR + 40 TotalSegmentator subjects + Kaggle instructions
# options: --max-subjects 0 (all 102, ~3 GB)   |   python scripts/download_data.py kidneystonetr
```
Downloads resume: finished subjects are skipped on re-run. Zenodo can be slow (~0.3–2 MB/s).

## 5. Run

One command:
```bash
./run_pipeline.sh              # full: CV5 + hold-out for the proposed model and the ClfNet-only baseline
QUICK=1 ./run_pipeline.sh      # smoke run on a subset, a few epochs (~30–60 min on CPU)
```

Or step by step:
```bash
python scripts/prepare_roi_source.py                     # TotalSegmentator → data/totalseg_2d
python scripts/train_i2ireg.py --epochs 30               # runs/i2ireg/best.pt
python scripts/pseudo_label.py                           # data/kidneystonetr/masks/<md5>.png
python scripts/train.py --name proposed --folds 1 2 3 4 5 --final
python scripts/evaluate.py --run runs/proposed --data holdout --gradcam 12 --timing
```

### Accuracy on any folder of images

```bash
python predict_folder.py /path/to/folder            # prints ACCURACY + recall/specificity/F1/κ/AUC, writes a CSV
python predict_folder.py data/raw/KidneyStoneTR/Dataset/dataset_integrated_task/holdout
```
Labels are read from sub-folder names at any depth: `Positive` / `Stone` / `1` means stone, and
`Negative` / `Normal` / `0` means no stone. PNG/JPG/BMP/TIF and DICOM (`.dcm`) are supported. Images without a
labelled parent folder still get predictions (in the CSV) but are left out of the accuracy. The same is available
in the UI under **Evaluate folder**.

### Paper experiments

| paper | command |
|---|---|
| Proposed (I2IRegNet RN18 + ShN+SqN+CBAM) | `train.py --name proposed --folds 1 2 3 4 5 --final` |
| ClfNet-only, no ROI stage (Fig. 3.7/3.9b) | `train.py --name clfnet_only --arch full_image --final` |
| SqueezeNet baseline (Fig. 3.9a) | `train.py --name squeezenet --arch full_image --no-shallow --no-cbam --final` |
| Table 3.4 variants | add `--deep resnet18`, `--no-shallow`, `--no-cbam` |
| Regression vs segmentation head (Sec. 3.4.3.2) | `train_i2ireg.py --head segmentation` then `train.py --name seg_head --head segmentation --final` |
| McNemar test | `evaluate.py --mcnemar runs/A/final/holdout_preds.csv runs/B/final/holdout_preds.csv` |
| External CT KIDNEY, no fine-tuning (Table 3.10) | `evaluate.py --run runs/proposed --data ctkidney` |
| Any folder of images | `evaluate.py --run runs/proposed --data folder:/path` (sub-folders `Positive`/`Negative` or `Stone`/`Normal`) |

Hyper-parameters follow the paper: 224×224×3 input, Adam lr 1e-4, 50 epochs, loss weights (0.8, 0.2),
with ReduceLROnPlateau for the "dynamic learning rate". Checkpoints are selected on a 10% random validation
split of the training images, never on the test fold.

Outputs go to `runs/<name>/{fold_k,final}/` (`best.pt`, `history.json`, `*_metrics.json`, `*_preds.csv`),
`runs/<name>/cv_summary.json` (mean ± std over folds), and `eval_*/` (confusion matrix, Grad-CAM panels).

## 6. UI

```bash
python app.py        # http://127.0.0.1:7860   (--share for a temporary public link)
```
* **Detect**: upload a PNG/JPG/DICOM slice and see the prediction, the predicted kidney ROI, and Grad-CAM on the
  ROI and on the original slice. Hold-out examples are one click away.
* **Evaluate folder**: type a folder path (or upload a folder) to get accuracy, the full metric set, a confusion
  matrix, a per-image table and a CSV of predictions.
* **ROI-aware vs full-image**: the same slice through two models, e.g. the proposed model vs `clfnet_only` on slices
  with gallbladder calcifications.
* **Results**: CV5 and hold-out table for every run, plus training curves.
* **Pseudo-mask review**: browse the kidney ROI pseudo-masks and reject bad ones before (re-)training.

## 7. Layout

```
app.py                    Gradio UI
predict_folder.py         folder of images → accuracy + CSV
start.sh / start.bat      one-click launcher (setup on first run + menu) → run.py
setup.sh / setup.bat      create .venv and install packages
run.py                    cross-platform menu / commands (ui, download, quick, full, folder)
run_pipeline.sh
scripts/
  download_data.py        step 1  datasets → data/raw
  prepare_roi_source.py   step 2  TotalSegmentator NIfTI → 2D kidney ROI pairs
  train_i2ireg.py         step 3  pre-train I2IRegNet
  pseudo_label.py         step 4  KidneyStoneTR kidney ROI masks
  train.py                step 5  cascade / baselines, CV5 + hold-out
  evaluate.py             step 6  metrics, McNemar, external test, Grad-CAM, timing
src/
  models.py               I2IRegNet, ShallowEncoder, DeepEncoder, CBAM, ClfNet, I2IRegClfNet
  data.py                 datasets, splits, augmentation
  engine.py               losses (½MSE, CE, focal BCE), training loop
  metrics.py              Acc/Recall/Spec/Prec/F1/κ/AUC, MSE/RMSE/MAE/IoU/Dice, McNemar
  gradcam.py, inference.py
data/  runs/  weights/    created at run time
```

## 8. Implementation notes and other deviations

* **Reference-guided weighting.** I2IRegNet outputs a 3-channel map `W`, and `ROI_hat = W ⊙ x`. This is how the
  "reference line for weighting" in Fig. 2.1 is read here: values inside the ROI are tied to the original
  intensities. The regression loss is still ½MSE against `ROI = mask ⊙ x`.
* **ASPP.** Depthwise dilated 3×3 convolutions (rates 1/6/12/18) → pointwise 256 → BN/ReLU, concatenated to
  14×14×1024. Up-sampling units follow Fig. 2.2 (8×8 transposed convolutions with stride 4, then cropping).
  I2IRegNet has ≈9.8 M parameters here (the paper reports ≈11.3 M).
* **Shallow encoder.** The paper found N=6 blocks with kernels 128/32/96/64/256/224 and pruned layers below 7×7.
  For a 224 input that leaves five blocks (7×7×256). The paper's ≈0.9 M parameter figure could not be matched
  exactly; this version has ≈1.2 M.
* **SqueezeNet** (v1.1, as in MATLAB) ends at 13×13, so it is average-pooled to 7×7 for fusion.
* **Light augmentation** (small affine, intensity jitter, no left/right flip) is on by default because the training
  set is small. Turn it off with `--no-augment` for a stricter paper setting.
* Pseudo-masks, rather than expert masks, are the regression targets (see §2).

## 9. Citation

If you use this code or data, cite the paper above and the datasets:
KidneyStoneTR (Öksüz, 2026, doi:10.6084/m9.figshare.31156783); TotalSegmentator (Wasserthal et al.,
*Radiology: AI* 2023, doi:10.1148/ryai.230024); CT KIDNEY DATASET (Islam et al., *Sci. Rep.* 2022).
