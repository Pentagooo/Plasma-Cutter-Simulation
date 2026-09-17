#!/usr/bin/env bash
# Einrichtung auf Ubuntu (Threadripper-Box) -- einmalig.
#
#   Voraussetzung: dieser Ordner heisst "plasma_cutter" (interne Importe
#   "plasma_cutter....") und liegt in einem Arbeitsordner $WORK, aus dem
#   alle Kommandos laufen:
#       $WORK/plasma_cutter/tools/setup_ubuntu.sh
#
#   Python 3.12 empfohlen (reine Python-Aufzaehlung ~25 % schneller als 3.10;
#   3.10 laeuft ebenfalls). GPU wird nicht benutzt (Shapely + Held-Karp).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(dirname "$HERE")"
if [[ "$(basename "$HERE")" != "plasma_cutter" ]]; then
    echo "FEHLER: Repo-Ordner muss 'plasma_cutter' heissen (ist: $(basename "$HERE"))" >&2
    exit 1
fi

PY="${PYTHON:-}"
for cand in python3.12 python3.13 python3.11 python3; do
    if [[ -z "$PY" ]] && command -v "$cand" >/dev/null 2>&1; then PY="$cand"; fi
done
echo "Python: $($PY --version)  ($(command -v "$PY"))"

if [[ "${INSTALL_APT:-0}" == "1" ]]; then
    sudo apt-get update
    sudo apt-get install -y python3-venv python3-tk      # tk nur fuer Dialoge/Simulator
fi

cd "$HERE"
if [[ ! -d .venv ]]; then
    "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

cd "$WORK"
echo "== Syntax aller Module =="
python - <<'EOF'
import glob, py_compile
for f in glob.glob("plasma_cutter/**/*.py", recursive=True):
    if ".venv" in f:
        continue
    py_compile.compile(f, doraise=True)
print("ok")
EOF

echo "== Parameterstempel (muss zu den finalen Werten passen) =="
python -m plasma_cutter.segment_simulation.surrogate.params

echo "== Tests (schnell) =="
python -m pytest plasma_cutter/segment_simulation/surrogate/tests -q -m "not slow"

echo "== Smoke-Label (3 Instanzen, k_max 14) =="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m plasma_cutter.segment_simulation.surrogate.dataset \
    --n 3 --seed 42 --n-jobs 3 --k-max 14 \
    --out plasma_cutter/segment_simulation/surrogate/artifacts/runs/smoke
echo "Setup fertig. Naechster Schritt: tools/run_final_training.sh phase1"
