#!/usr/bin/env bash
# Creates ./.venv and installs everything this project needs (nothing is installed globally).
#   ./setup.sh            auto: CUDA wheels if an NVIDIA GPU is present, else CPU wheels
#   ./setup.sh cpu        force CPU-only PyTorch (smaller download)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 required"'

if [ ! -d .venv ]; then
  echo ">> creating virtual environment .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

MODE=${1:-auto}
if [ "$MODE" = "cpu" ] || { [ "$MODE" = "auto" ] && [ "$(uname)" = "Linux" ] && ! command -v nvidia-smi >/dev/null; }; then
  echo ">> installing CPU-only PyTorch"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  echo ">> installing PyTorch (default wheels: CUDA on Linux/Windows, MPS on Apple silicon)"
  pip install torch torchvision
fi
pip install -r requirements.txt

python - <<'EOF'
import torch
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\n>> ready. torch {torch.__version__}, device = {dev}")
EOF

for t in unar 7z 7zz unrar bsdtar; do command -v $t >/dev/null && { RAR_OK=1; break; }; done
if [ -z "${RAR_OK:-}" ]; then
  echo ">> NOTE: no RAR extractor found (needed once, for the KidneyStoneTR .rar)."
  echo "   Ubuntu/Debian: sudo apt install unar | Fedora: sudo dnf install unar | macOS: brew install unar"
fi
echo ">> next:  source .venv/bin/activate && python scripts/download_data.py --all"
