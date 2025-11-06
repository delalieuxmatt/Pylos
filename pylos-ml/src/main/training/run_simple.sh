#!/usr/bin/env bash

set -euo pipefail

# ====== Instellingen ======
PYTHON_BIN=${PYTHON_BIN:-python3.11}   # Gebruik python3.11 tenzij anders opgegeven
VENV_DIR=${VENV_DIR:-.venv311}         # Naam van je virtuele omgeving
SCRIPT_PATH=${1:-./train.py}           # Eerste argument = pad naar je Python script

# ====== Controleer of script bestaat ======
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Kon script niet vinden: $SCRIPT_PATH"
  echo "Gebruik: ./run_simple.sh pad/naar/jouw_script.py"
  exit 1
fi

# ====== Controleer of venv bestaat ======
if [ ! -d "$VENV_DIR" ]; then
  echo "Virtuele omgeving niet gevonden: $VENV_DIR"
  echo "Maak er eerst een aan met: $PYTHON_BIN -m venv $VENV_DIR"
  exit 1
fi

# ====== Activeer de virtuele omgeving ======
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# ====== Start het Python script ======
echo "== Start: $SCRIPT_PATH =="
python "$SCRIPT_PATH"
echo "== Klaar! =="
