#!/usr/bin/env bash
# Abschluss-Trainingslauf auf der Ubuntu-Box, phasenweise und resumebar.
#
#   Start (aus $WORK, dem Elternordner von plasma_cutter), z.B. in tmux:
#       nohup plasma_cutter/tools/run_final_training.sh phase1 > logs/phase1.out 2>&1 &
#       nohup plasma_cutter/tools/run_final_training.sh phase2 > logs/phase2.out 2>&1 &
#       plasma_cutter/tools/run_final_training.sh phase3
#
#   phase1  Testsaetze (seed 7, Standard 12/4 und fein 18/3) labeln (~1 h),
#           dann Hauptlauf: N_MAIN Katalog-Instanzen mit gemischter
#           Segmentierung (SEG_MIX, k 5..20) bis BUDGET1_MIN, Training,
#           Benchmark auf beiden Testsaetzen, Lernkurve.   ~18 h gesamt
#   phase2  Hauptlauf FORTSETZEN (Cache, weitere BUDGET2_MIN), neu trainieren,
#           Benchmark + Lernkurve neu (nur wenn noch Zeit ist)
#   phase3  nur Benchmarks + Lernkurve (nach Bedarf), dann collect_results.sh
#
#   Jede Phase ist idempotent: Labels liegen je Instanz im Cache. Sanfter
#   Stopp eines Label-Laufs: touch STOP (laufende Instanzen rechnen zu Ende).
#   Zeitbudget stoppt nur das Einreichen neuer Instanzen; die bis dahin
#   fertigen Labels sind eine zufaellige Teilmenge (Instanzreihenfolge ist
#   ueber Familien und Segmentierungen gemischt).
#
#   Kosten (CPU-s je Instanz, pessimistisch): k12 13, k14 64, k16 364, k18 1170,
#   k20 ~4200, k21 ~7900. Grosser Lauf (24.09.2026, Katalog SHAPE_VERSION 7 mit
#   Familie "assembly", 10 000 Instanzen, k_max 21): Dry-Run ~87 % <= k 21
#   (~8 700 Labels, assembly ~1 870), 1 100-2 250 CPU-h -> 25-51 h bei 44
#   Workern. Alle Ausgaben unter runs/*_$TAG (alte Laeufe bleiben unberuehrt).
set -euo pipefail

# ------------------------------------------------------------ Konfiguration
NJOBS="${NJOBS:-$(( $(nproc) > 4 ? $(nproc) - 4 : $(nproc) ))}"   # 4 Threads fuer System/Assembly frei
TAG="${TAG:-v7}"                   # Suffix aller Laufordner (Katalog SHAPE_VERSION 7)
N_MAIN="${N_MAIN:-10000}"          # 10 Slots: 7 Familien x 1000 + assembly 3000
N_EVAL="${N_EVAL:-90}"             # Testinstanzen je Seed + reale Geometrien
SEED_TRAIN=42
SEED_EVAL=7
SEED_EVAL2=11                      # zweiter, unabhaengiger Testsatz
KMAX_MAIN="${KMAX_MAIN:-21}"
SEG_MIX="${SEG_MIX:-12/4:0.50,14/4:0.25,16/3:0.20,18/3:0.05}"
SEG_DIV_FINE="${SEG_DIV_FINE:-18}"  # feiner Testsatz
SEG_MIN_FINE="${SEG_MIN_FINE:-3}"
BUDGET1_MIN="${BUDGET1_MIN:-2700}"  # 45 h Labeln in phase1
BUDGET2_MIN="${BUDGET2_MIN:-600}"   # 10 h Labeln in phase2 (optional)
N_LIST="${N_LIST:-250,1000,2500,5000,10000}"
STOP_FILE="STOP"
# ---------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(dirname "$HERE")"
cd "$WORK"
# shellcheck disable=SC1091
source "$HERE/.venv/bin/activate"
mkdir -p logs
ART="plasma_cutter/segment_simulation/surrogate/artifacts"
RUNS="$ART/runs"
MOD="plasma_cutter.segment_simulation.surrogate"
MAIN="$RUNS/main_$TAG"

label_env()  { export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; }
train_env()  { export OMP_NUM_THREADS="$NJOBS" OPENBLAS_NUM_THREADS="$NJOBS"; }
stamp()      { python -m "$MOD.params" | head -4; }

label_main() {   # $1 = Budget [min]
    label_env
    python -m "$MOD.dataset" --n "$N_MAIN" --seed $SEED_TRAIN --k-max "$KMAX_MAIN" \
        --seg-mix "$SEG_MIX" --n-jobs "$NJOBS" --max-minutes "$1" \
        --stop-file "$STOP_FILE" --out "$MAIN" 2>&1 | tee -a logs/main_labels_$TAG.log
}

label_eval() {
    label_env
    for sd in $SEED_EVAL $SEED_EVAL2; do
        # Standard-Segmentierung (wie Simulator), Optimum bis k_max
        python -m "$MOD.dataset" --n "$N_EVAL" --seed "$sd" --k-max "$KMAX_MAIN" \
            --n-jobs "$NJOBS" --include-real --out "$RUNS/eval_s${sd}_$TAG" \
            2>&1 | tee -a logs/eval_labels_$TAG.log
        # feine Segmentierung, Optimum bis k_max
        python -m "$MOD.dataset" --n "$N_EVAL" --seed "$sd" --k-max "$KMAX_MAIN" \
            --seg-divisor "$SEG_DIV_FINE" --seg-min-spacings "$SEG_MIN_FINE" \
            --n-jobs "$NJOBS" --include-real --out "$RUNS/fine_eval_s${sd}_$TAG" \
            2>&1 | tee -a logs/fine_eval_labels_$TAG.log
    done
}

train_main() {
    train_env
    python -m "$MOD.model" --train --out "$MAIN" 2>&1 | tee -a logs/train_$TAG.log
    cp "$MAIN/surrogate_model.joblib" "$ART/surrogate_model.joblib"
    cp "$MAIN/model_meta.json" "$ART/model_meta.json"
}

benchmarks() {
    train_env
    for sd in $SEED_EVAL $SEED_EVAL2; do
        python -m "$MOD.benchmark" --n "$N_EVAL" --seed "$sd" --reps 3 \
            --model "$MAIN/surrogate_model.joblib" --out "$MAIN" \
            --stem "benchmark_s$sd" --k-max "$KMAX_MAIN" \
            --opt-labels "$RUNS/eval_s${sd}_$TAG/labels" \
            2>&1 | tee -a logs/benchmark_$TAG.log
        python -m "$MOD.benchmark" --n "$N_EVAL" --seed "$sd" --reps 3 \
            --model "$MAIN/surrogate_model.joblib" --out "$MAIN" \
            --stem "benchmark_fine_s$sd" --k-max "$KMAX_MAIN" \
            --seg-divisor "$SEG_DIV_FINE" --seg-min-spacings "$SEG_MIN_FINE" \
            --opt-labels "$RUNS/fine_eval_s${sd}_$TAG/labels" \
            2>&1 | tee -a logs/benchmark_fine_$TAG.log
    done
    python -m "$MOD.learning_curve" --dataset "$MAIN" --n-eval "$N_EVAL" \
        --seed $SEED_EVAL --n-list "$N_LIST" \
        --opt-labels "$RUNS/eval_s${SEED_EVAL}_$TAG/labels" \
        2>&1 | tee -a logs/learning_curve_$TAG.log
}

phase1() {
    echo "== phase1: Testsaetze, dann Hauptlauf (seg-mix $SEG_MIX, k_max $KMAX_MAIN, n=$N_MAIN, Budget $BUDGET1_MIN min) =="; stamp
    label_eval
    label_main "$BUDGET1_MIN"
    train_main
    benchmarks
    echo "== phase1 fertig =="
}

phase2() {
    echo "== phase2: Hauptlauf fortsetzen (Budget $BUDGET2_MIN min), neu trainieren =="
    label_main "$BUDGET2_MIN"
    train_main
    benchmarks
    echo "== phase2 fertig =="
}

phase3() {
    echo "== phase3: Benchmarks + Lernkurve =="
    benchmarks
    echo "== phase3 fertig -> tools/collect_results.sh =="
}

snapshot() {
    # Zwischenstand WAEHREND des Laufs: aktuelle Labels einsammeln, Modell
    # trainieren, Benchmark auf dem Standard-Testsatz (falls schon gelabelt).
    # Stoert den laufenden Label-Prozess nicht (liest nur den Cache).
    SNAP="$RUNS/snapshot_${TAG}_$(date +%H%M)"
    echo "== snapshot -> $SNAP =="
    train_env
    python -m "$MOD.dataset" --n 0 --seed $SEED_TRAIN --k-max "$KMAX_MAIN" \
        --seg-mix "$SEG_MIX" --extra-labels "$MAIN/labels" --out "$SNAP"
    python -m "$MOD.model" --train --out "$SNAP"
    if [[ -d "$RUNS/eval_s${SEED_EVAL}_$TAG/labels" ]]; then
        python -m "$MOD.benchmark" --n "$N_EVAL" --seed $SEED_EVAL --reps 1 \
            --model "$SNAP/surrogate_model.joblib" --out "$SNAP" --k-max "$KMAX_MAIN" \
            --opt-labels "$RUNS/eval_s${SEED_EVAL}_$TAG/labels"
        echo "-> $SNAP/benchmark.md"
    fi
}

case "${1:-}" in
    phase1|phase2|phase3|snapshot) "$1" ;;
    *) echo "usage: $0 {phase1|phase2|phase3|snapshot}" >&2; exit 2 ;;
esac
