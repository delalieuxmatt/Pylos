#!/usr/bin/env bash


set -euo pipefail

# ====== Instellingen ======
PYTHON_BIN=${PYTHON_BIN:-python3.11}   # of absolute pad naar je Python 3.11
VENV_DIR=${VENV_DIR:-.venv311}
SCRIPT_PATH=${1:-./train.py}             # geef pad naar je script mee als 1e arg (default: train.py)
DATASET_PATH=${DATASET_PATH:-resources/games/0.json}

# ====== Checks ======a
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Kon $PYTHON_BIN niet vinden. Installeer Python 3.11 of zet PYTHON_BIN=/pad/naar/python3.11"
  exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Kon trainingsscript niet vinden: $SCRIPT_PATH"
  echo "Geef het bestand door als argument, bv.: ./run_training.sh pad/naar/jouw_script.py"
  exit 1
fi

# Dataset en modelmap aanleggen (als ze nog niet bestaan)
mkdir -p resources/games resources/models
if [ ! -f "$DATASET_PATH" ]; then
  echo "WAARSCHUWING: dataset ontbreekt op $DATASET_PATH"
  echo "Zet je JSON games-bestand daar neer voordat je traint."
fi

# ====== Virtuele omgeving ======
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# ====== TensorFlow + deps installeren op basis van OS/arch ======
OS="$(uname -s || echo Unknown)"
ARCH="$(uname -m || echo Unknown)"

echo "Detectie: OS=$OS ARCH=$ARCH"

pip install "tensorflow[and-cuda]==2.20.*"

# Overige libs (json is stdlib)
pip install "numpy>=1.26,<3.0"

# (Optioneel) laat TF minder threads gebruiken; jouw script stuurt dit zelf ook al aan
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-8}"

echo "== Start training =="
python "$SCRIPT_PATH"
echo "== Klaar! =="

