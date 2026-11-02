#!/bin/bash
# End-to-end smoke test of the training workflow on CPU:
# data generation -> training -> serialization -> loading the net in the engine.
# The resulting net is meaningless; only the functionality is verified.
#
# Usage: scripts/cpu_smoke_test.sh <path to variant-nnue-tools stockfish binary> [work dir]
# Requires the data loader to be built (compile_data_loader.bat) and the
# Python dependencies to be installed (CuPy is not needed).

set -euo pipefail

ENGINE=$(realpath "$1")
WORKDIR=$(realpath -m "${2:-smoke_test}")
VARIANT=chess
PYTHON=${PYTHON:-python}
TRAINER_DIR=$(cd "$(dirname "$0")/.." && pwd)

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

generate() {
    local name=$1 count=$2 seed=$3
    printf '%s\n' \
        "setoption name Threads value 2" \
        "setoption name UCI_Variant value $VARIANT" \
        "setoption name Use NNUE value false" \
        "setoption name PruneAtShallowDepth value false" \
        "generate_training_data depth 3 count $count random_multi_pv 4 random_multi_pv_diff 100 random_move_count 8 random_move_max_ply 20 write_min_ply 5 eval_limit 10000 seed $seed output_file_name $WORKDIR/$name" \
        "quit" | "$ENGINE"
    test -s "$WORKDIR/$name.bin"
}

echo "=== Generating training data ==="
generate train 20000 1
generate val 5000 2

echo "=== Training on CPU ==="
cd "$TRAINER_DIR"
"$PYTHON" train.py "$WORKDIR/train.bin" "$WORKDIR/val.bin" \
    --accelerator cpu --threads 2 --num-workers 1 \
    --batch-size 1024 --epoch-size 20000 --validation-size 5000 \
    --max_epochs 2 --enable_progress_bar false \
    --default_root_dir "$WORKDIR/logs"
CKPT=$(find "$WORKDIR/logs" -name last.ckpt | head -n 1)
test -n "$CKPT"

echo "=== Serializing ==="
# The engine only loads nets whose file name starts with the variant name.
NET="$WORKDIR/$VARIANT-smoke.nnue"
"$PYTHON" serialize.py "$CKPT" "$NET"
# Round trip .nnue -> .pt -> .nnue must be lossless. A .nnue file only contains
# the real (non-factorized) features, so read it with the plain feature set.
"$PYTHON" serialize.py "$NET" "$WORKDIR/roundtrip.pt" --features=HalfKAv2
"$PYTHON" serialize.py "$WORKDIR/roundtrip.pt" "$WORKDIR/$VARIANT-roundtrip.nnue" --features=HalfKAv2
cmp "$NET" "$WORKDIR/$VARIANT-roundtrip.nnue"

echo "=== Loading the net in the engine ==="
cd "$WORKDIR"
OUTPUT=$({ printf '%s\n' \
    "setoption name UCI_Variant value $VARIANT" \
    "setoption name EvalFile value $(basename "$NET")" \
    "isready" \
    "position startpos" \
    "go depth 6"
    sleep 5
    echo quit; } | "$ENGINE")
echo "$OUTPUT" | grep -E "NNUE|bestmove"
echo "$OUTPUT" | grep -q "NNUE evaluation using $(basename "$NET") enabled"
echo "$OUTPUT" | grep -q "^bestmove"

echo "=== Smoke test passed ==="
