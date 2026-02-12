#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_SRC="$ROOT_DIR/src/data"

source "$ROOT_DIR/config.env"

export DATA_DIR DATA_SAMPLES TRAIN_SPLIT SL HF_HOME HF_DATASETS_CACHE

echo "Fetching pseudo-camera-10k captions..."
python3 "$DATA_SRC/fetch.py" \
    --samples "${DATA_SAMPLES}" \
    --output "${DATA_DIR}/pseudo-camera-raw.jsonl"

# echo "Cleaning captions..."
# python3 "$DATA_SRC/clean.py" \
#     --input "${DATA_DIR}/pseudo-camera-raw.jsonl" \
#     --output "${DATA_DIR}/pseudo-camera.jsonl"

# echo "Encoding pseudo-camera-10k dataset..."
# python3 "$DATA_SRC/encode.py" \
#     --input "${DATA_DIR}/pseudo-camera.jsonl" \
#     --output-dir "${DATA_DIR}" \
#     --output-name "pc" \
#     --seq-length "${SL}" \
#     --max-samples "${DATA_SAMPLES}" \
#     --train-split "${TRAIN_SPLIT}"

# echo "Verifying encoded dataset..."
# python3 "$DATA_SRC/verify.py" \
#     --input-dir "${DATA_DIR}"
