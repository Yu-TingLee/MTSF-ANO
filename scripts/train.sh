#!/usr/bin/env bash
# Trains and tests one model on one ETT dataset, once per seed.
#
#   scripts/train.sh MODEL DATASET
#
# Environment overrides:
#   SEEDS ("1 2 3 4 5"), PRED_LEN (1), SEQ_LEN (16), PYTHON (python), DRY_RUN (0)
#   MTSF-ANO, MTSF-ANO-CI: N_LOCAL (1), NUM_LAYERS (3), LEARNABLE_SCALING (1), ENTANGLEMENT (cnot)
#   baselines: MODEL_TAG (MODEL), LSTM_HIDDEN (7)
#
# Results go to results/<run>/<DATASET>_L<SEQ_LEN>_H<PRED_LEN>_s<seed>/, where <run>
# is the model name plus a suffix for every non-default setting.
set -e
model=${1:?usage: scripts/train.sh MODEL DATASET}
data=${2:?usage: scripts/train.sh MODEL DATASET}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS=${SEEDS:-"1 2 3 4 5"}
PRED_LEN=${PRED_LEN:-1}
SEQ_LEN=${SEQ_LEN:-16}
PYTHON=${PYTHON:-python}

case "$data" in
  ETTh1|ETTh2) freq=h ;;
  ETTm1|ETTm2) freq=t ;;
  *) echo "train.sh: unknown dataset '$data'" >&2; exit 1 ;;
esac
enc_in=7

suffix=""
case "$model" in
  MTSF-ANO|MTSF-ANO-CI)
    N_LOCAL=${N_LOCAL:-1}
    NUM_LAYERS=${NUM_LAYERS:-3}
    LEARNABLE_SCALING=${LEARNABLE_SCALING:-1}
    ENTANGLEMENT=${ENTANGLEMENT:-cnot}
    suffix="-comb${N_LOCAL}local"
    [ "$NUM_LAYERS" != "3" ]       && suffix="${suffix}-L${NUM_LAYERS}"
    [ "$LEARNABLE_SCALING" = "0" ] && suffix="${suffix}-fixscale"
    [ "$ENTANGLEMENT" = "none" ]   && suffix="${suffix}-noent"
    model_args=(--num_layers "$NUM_LAYERS" --n_local "$N_LOCAL"
                --learnable_scaling "$LEARNABLE_SCALING" --entanglement "$ENTANGLEMENT"
                --QML_device default.qubit) ;;
  MTSF-PZ) model_args=(--num_layers 3 --QML_device default.qubit) ;;
  LSTM)    model_args=(--lstm_hidden "${LSTM_HIDDEN:-7}" --lstm_layers 3 --lstm_dropout 0.05
                       --no_checkpoint 1) ;;
  QuLTSF)  model_args=(--num_layers 3 --QML_device default.qubit --no_checkpoint 1) ;;
  QLSTM)   model_args=(--num_layers 3 --qlstm_input 3 --qlstm_hidden 4
                       --QML_device default.qubit --no_checkpoint 1) ;;
  QFWP)    model_args=(--qfwp_qubits 7 --qfwp_depth 3 --qfwp_a 7 --no_checkpoint 1) ;;
  Linear|DLinear|NLinear) model_args=(--individual 0 --no_checkpoint 1) ;;
  *) echo "train.sh: unknown model '$model'" >&2; exit 1 ;;
esac
[ "$SEQ_LEN" != "16" ] && suffix="${suffix}-lookback${SEQ_LEN}"

run="${MODEL_TAG:-$model}${suffix}"
mkdir -p "logs/$run"
for seed in $SEEDS; do
  out_dir="results/$run/${data}_L${SEQ_LEN}_H${PRED_LEN}_s${seed}"
  if [ -f "$out_dir/metrics.npz" ]; then echo "skip: $out_dir"; continue; fi

  cmd=("$PYTHON" -u run.py
    --random_seed "$seed"
    --model "$model"
    --save_suffix="${run#"$model"}"
    --data "$data"
    --root_path "$ROOT_DIR/dataset/ETT-small/"
    --data_path "${data}.csv"
    --features M
    --freq "$freq"
    --seq_len "$SEQ_LEN"
    --label_len 0
    --pred_len "$PRED_LEN"
    --enc_in "$enc_in"
    --num_qubits "$enc_in"
    --use_revin 1
    "${model_args[@]}"
    --batch_size 1024
    --num_workers 0
    --train_epochs 100
    --patience 10
    --learning_rate 0.001
    --lradj constant
    --checkpoints "./checkpoints/$run/")

  if [ "${DRY_RUN:-0}" = "1" ]; then printf '%s\n' "${cmd[*]}"; continue; fi
  "${cmd[@]}" > "logs/$run/${model}_${data}_${SEQ_LEN}_${PRED_LEN}_s${seed}.log" 2>&1 \
    || echo "WARN $run $data H=$PRED_LEN seed=$seed exited non-zero"
  echo "done -> $out_dir"
done
