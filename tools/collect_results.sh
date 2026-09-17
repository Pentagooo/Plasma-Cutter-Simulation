#!/usr/bin/env bash
# Ergebnisse der Box einsammeln: alle Laeufe (inkl. Labels) + Logs als tar.gz.
#   plasma_cutter/tools/collect_results.sh            (aus $WORK)
# Zurueck auf Windows: entpacken nach plasma_cutter/segment_simulation/surrogate/artifacts/
# und das gewaehlte Modell (runs/<name>/surrogate_model.joblib + model_meta.json)
# nach artifacts/ kopieren -- dann laeuft Taste S im Simulator damit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(dirname "$HERE")"
cd "$WORK"
ART="plasma_cutter/segment_simulation/surrogate/artifacts"
OUT="results_final_$(date +%F_%H%M).tar.gz"
tar czf "$OUT" -C "$ART" runs -C "$WORK" logs
ls -lh "$OUT"
echo "scp $(hostname):$WORK/$OUT  ->  Windows"
