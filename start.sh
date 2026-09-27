#!/usr/bin/env bash
# Linux / macOS one-click start:  ./start.sh   (or: bash start.sh)
# First run creates .venv and installs everything; then shows the menu.
# Pass-through: ./start.sh ui | quick | full | download | folder PATH
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import torch, gradio" 2>/dev/null; then
  echo ">> first run: setting up the environment (a few minutes)..."
  bash setup.sh
fi
exec .venv/bin/python run.py "$@"
