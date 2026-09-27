#!/usr/bin/env bash
# End-to-end reproduction. Every step can also be run on its own (see README).
#   ./run_pipeline.sh              full run (GPU recommended)
#   QUICK=1 ./run_pipeline.sh      small smoke run (a few epochs on a subset; CPU OK, ~30-60 min)
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

if [ "${QUICK:-0}" = "1" ]; then
  SUBJ=10; E1=5; E2=3; LIMIT="--limit 600"; FOLDS=""
else
  SUBJ=${SUBJECTS:-40}; E1=${EPOCHS_ROI:-30}; E2=${EPOCHS:-50}; LIMIT=""; FOLDS="--folds 1 2 3 4 5"
fi

echo "== 1/6 download";           python scripts/download_data.py kidneystonetr totalseg --max-subjects "$SUBJ"
echo "== 2/6 kidney ROI slices";  python scripts/prepare_roi_source.py
echo "== 3/6 pre-train I2IRegNet"; python scripts/train_i2ireg.py --epochs "$E1"
echo "== 4/6 pseudo ROI masks";   python scripts/pseudo_label.py
echo "== 5/6 train";
python scripts/train.py --name proposed    $FOLDS --final --epochs "$E2" $LIMIT
python scripts/train.py --name clfnet_only --arch full_image --final --epochs "$E2" $LIMIT
echo "== 6/6 evaluate"
python scripts/evaluate.py --run runs/proposed --data holdout --gradcam 12 --timing
python scripts/evaluate.py --mcnemar runs/proposed/final/holdout_preds.csv runs/clfnet_only/final/holdout_preds.csv
echo "done - launch the UI with:  python app.py"
