#!/usr/bin/env bash
# Runs every experiment phase, JOBS training runs at a time.
#   bash run.sh [PHASE ...]   (default: all phases)
# Environment overrides: PYTHON SEEDS JOBS THREADS_PER_JOB
set -e
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10"}
JOBS=${JOBS:-5}
NPROC=$(nproc 2>/dev/null || echo 1)
if [ -z "${THREADS_PER_JOB:-}" ]; then
  THREADS_PER_JOB=$(( NPROC / JOBS )); [ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1
fi

if [ "$#" -gt 0 ]; then PHASES="$*"; else PHASES=${PHASES:-"baselines main-ano main-ci arch lookback"}; fi

ETT="ETTh1 ETTh2 ETTm1 ETTm2"
BASELINE_MODELS="LSTM Linear DLinear NLinear MTSF-PZ QFWP QLSTM QuLTSF"
ETT_PLS="1 5 16 32 48"

running=0
throttle() {
  running=$((running + 1))
  if (( running >= JOBS )); then wait -n 2>/dev/null || true; running=$((running - 1)); fi
}

# launch MODEL DATASET [VAR=value ...]
launch() {
  local model="$1" data="$2"; shift 2
  env "$@" PYTHON="$PYTHON" \
    OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
    bash "$BASE_DIR/scripts/train.sh" "$model" "$data" &
  throttle
}

phase_banner() { echo; echo "########## phase: $1 ##########"; }

# Baselines: 8 models x 4 datasets x 5 horizons.
phase_baselines() {
  phase_banner "baselines"
  local m d pl seed
  for m in $BASELINE_MODELS; do
    for d in $ETT; do
      for pl in $ETT_PLS; do
        for seed in $SEEDS; do
          launch "$m" "$d" "SEEDS=$seed" "PRED_LEN=$pl"
        done
      done
    done
  done
  wait || true; running=0
}

# MTSF-ANO: k = 1..7 x 4 datasets x 5 horizons.
phase_main_ano() {
  phase_banner "main MTSF-ANO"
  local k d pl seed
  for k in 1 2 3 4 5 6 7; do
    for d in $ETT; do
      for pl in $ETT_PLS; do
        for seed in $SEEDS; do
          launch MTSF-ANO "$d" "SEEDS=$seed" "PRED_LEN=$pl" "N_LOCAL=$k"
        done
      done
    done
  done
  wait || true; running=0
}

# MTSF-ANO-CI: k = 3, 5 on ETTh1, H = 48.
phase_main_ci() {
  phase_banner "main MTSF-ANO-CI"
  local k seed
  for k in 3 5; do
    for seed in $SEEDS; do
      launch MTSF-ANO-CI ETTh1 "SEEDS=$seed" "PRED_LEN=48" "N_LOCAL=$k"
    done
  done
  wait || true; running=0
}

# Architecture ablation on ETTh1: layers, scaling and entanglement for k = 3, 5 and H = 1, 48.
phase_arch() {
  phase_banner "arch ablation (L / scale / entanglement)"
  local ARMS=(
    "1 1 cnot"  "1 1 none"  "1 0 cnot"  "1 0 none"
    "3 0 none"  "3 1 none"  "3 0 cnot"
    "5 1 cnot"  "5 1 none"  "5 0 cnot"  "5 0 none"
  )
  local arm nl ls ent k pl seed
  for arm in "${ARMS[@]}"; do
    read -r nl ls ent <<< "$arm"
    for k in 3 5; do
      for pl in 1 48; do
        for seed in $SEEDS; do
          launch MTSF-ANO ETTh1 "SEEDS=$seed" "PRED_LEN=$pl" "N_LOCAL=$k" \
                 "NUM_LAYERS=$nl" "LEARNABLE_SCALING=$ls" "ENTANGLEMENT=$ent"
        done
      done
    done
  done
  wait || true; running=0
}

# Lookback ablation on ETTh1, H = 48: MTSF-ANO and MTSF-ANO-CI (k = 3, 5) and all baselines.
phase_lookback() {
  phase_banner "lookback ablation (L sweep @ H=48)"
  local SEQ_LENS="8 24 32 48 72 96"  # L = 16 is covered by the main phases
  local L k seed m
  for L in $SEQ_LENS; do
    for k in 3 5; do
      for seed in $SEEDS; do
        launch MTSF-ANO ETTh1 "SEEDS=$seed" "PRED_LEN=48" "N_LOCAL=$k" "SEQ_LEN=$L"
      done
    done
    for k in 3 5; do
      for seed in $SEEDS; do
        launch MTSF-ANO-CI ETTh1 "SEEDS=$seed" "PRED_LEN=48" "N_LOCAL=$k" "SEQ_LEN=$L"
      done
    done
    for m in $BASELINE_MODELS; do
      for seed in $SEEDS; do
        launch "$m" ETTh1 "SEEDS=$seed" "PRED_LEN=48" "SEQ_LEN=$L"
      done
    done
  done
  wait || true; running=0
}

echo "run.sh: JOBS=$JOBS THREADS_PER_JOB=$THREADS_PER_JOB (cores=$NPROC)  phases=[$PHASES]"
for phase in $PHASES; do
  case "$phase" in
    baselines) phase_baselines ;;
    main-ano)  phase_main_ano ;;
    main-ci)   phase_main_ci ;;
    arch)      phase_arch ;;
    lookback)  phase_lookback ;;
    *) echo "!! unknown phase: '$phase' (valid: baselines main-ano main-ci arch lookback)" >&2; exit 1 ;;
  esac
done

echo
echo "=== run.sh complete: phases [$PHASES] ==="