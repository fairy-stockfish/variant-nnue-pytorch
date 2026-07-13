#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 TRAIN.bin VALIDATION.bin [extra train.py arguments...]" >&2
  exit 2
fi

train_data=$1
validation_data=$2
seed=${NNUE_SEED:-42}
shift 2

if [[ -e "$train_data" && -e "$validation_data" && "$train_data" -ef "$validation_data" ]]; then
  echo "Training and validation data must be separate files." >&2
  exit 2
fi

python train.py \
 "$train_data" \
 "$validation_data" \
 --accelerator gpu \
 --devices 1 \
 --threads 2 \
 --batch-size 8192 \
 --random-fen-skipping 10 \
 --features=HalfKAv2^ \
 --lambda=1.0 \
 --seed "$seed" \
 --max_epochs=300 \
 "$@"
