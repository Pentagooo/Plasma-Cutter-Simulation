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
#   k20 ~4200. Voreinstellung (4200 Instanzen, k_max 20): ~700 CPU-h ->
#   ~16 h auf einem Threadripper 5965WX (24 Kerne / 48 Threads, 44 Worker).
set -euo pipefail

# ------------------------------------------------------------ Konfiguration
NJOBS="${NJOBS:-$(( $(nproc) > 4 ? $(nproc) - 4 : $(nproc) ))}"   # 4 Threads fuer System/Assembly frei
N_MAIN="${N_MAIN:-4200}"           # 7 Familien x 600 (24-h-Ziel, 18.09.)
N_EVAL="${N_EVAL:-60}"             # Testinstanzen (seed 7) + reale Geometrien
SEED_TRAIN=42
SEED_EVAL=7
KMAX_MAIN="${KMAX_MAIN:-20}"
SEG_MIX="${SEG_MIX:-12/4:0.50,14/4:0.25,16/3:0.20,18/3:0.05}"
SEG_DIV_FINE="${SEG_DIV_FINE:-18}"  # feiner Testsatz
SEG_MIN_FINE="${SEG_MIN_FINE:-3}"
BUDGET1_MIN="${BUDGET1_MIN:-900}"   # 15 h Labeln in phase1
BUDGET2_MIN="${BUDGET2_MIN:-600}"   # 10 h Labeln in phase2 (optional)
N_LIST="${N_LIST:-100,250,500,1000,2000,4200}"
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

label_env()  { export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; }
train_env()  { export OMP_NUM_THREADS="$NJOBS" OPENBLAS_NUM_THREADS="$NJOBS"; }
stamp()      { python -m "$MOD.params" | head -4; }

label_main() {   # $1 = Budget [min]
    label_env
    python -m "$MOD.dataset" --n "$N_MAIN" --seed $SEED_TRAIN --k-max "$KMAX_MAIN" \
        --seg-mix "$SEG_MIX" --n-jobs "$NJOBS" --max-minutes "$1" \
        --stop-file "$STOP_FILE" --out "$RUNS/main" 2>&1 | tee -a logs/main_labels.log
}

label_eval() {
    label_env
    # Standard-Segmentierung (wie Simulator), Optimum bis k 18
    python -m "$MOD.dataset" --n "$N_EVAL" --seed $SEED_EVAL --k-max 18 \
        --n-jobs "$NJOBS" --include-real --out "$RUNS/main_eval" \
        2>&1 | tee -a logs/eval_labels.log
    # feine Segmentierung, Optimum bis k 22
    python -m "$MOD.dataset" --n "$N_EVAL" --seed $SEED_EVAL --k-max "$KMAX_MAIN" \
        --seg-divisor "$SEG_DIV_FINE" --seg-min-spacings "$SEG_MIN_FINE" \
        --n-jobs "$NJOBS" --include-real --out "$RUNS/fine_eval" \
        2>&1 | tee -a logs/fine_eval_labels.log
}

train_main() {
    train_env
    python -m "$MOD.model" --train --out "$RUNS/main" 2>&1 | tee -a logs/train.log
    cp "$RUNS/main/surrogate_model.joblib" "$ART/surrogate_model.joblib"
    cp "$RUNS/main/model_meta.json" "$ART/model_meta.json"
}

benchmarks() {
    train_env
    python -m "$MOD.benchmark" --n "$N_EVAL" --seed $SEED_EVAL --reps 3 \
        --model "$RUNS/main/surrogate_model.joblib" --out "$RUNS/main" \
        --stem benchmark --opt-labels "$RUNS/main_eval/labels" \
        2>&1 | tee -a logs/benchmark.log
    python -m "$MOD.benchmark" --n "$N_EVAL" --seed $SEED_EVAL --reps 3 \
        --model "$RUNS/main/surrogate_model.joblib" --out "$RUNS/main" \
        --stem benchmark_fine --k-max "$KMAX_MAIN" \
        --seg-divisor "$SEG_DIV_FINE" --seg-min-spacings "$SEG_MIN_FINE" \
        --opt-labels "$RUNS/fine_eval/labels" 2>&1 | tee -a logs/benchmark_fine.log
    python -m "$MOD.learning_curve" --dataset "$RUNS/main" --n-eval "$N_EVAL" \
        --seed $SEED_EVAL --n-list "$N_LIST" --opt-labels "$RUNS/main_eval/labels" \
        2>&1 | tee -a logs/learning_curve.log
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

case "${1:-}" in
    phase1|phase2|phase3) "$1" ;;
    *) echo "usage: $0 {phase1|phase2|phase3}" >&2; exit 2 ;;
esac
