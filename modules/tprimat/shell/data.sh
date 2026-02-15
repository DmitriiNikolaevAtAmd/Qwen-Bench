#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PREPARE_DIR="$ROOT_DIR/data"

source "$ROOT_DIR/config.env"

export DATA_DIR
export DATA_SAMPLES
export TRAIN_SPLIT
export SL

python3 "$PREPARE_DIR/fetch.py" \
    --samples "${DATA_SAMPLES}" \
    --output "${DATA_DIR}/allenai-c4-raw.jsonl" \
    --bookcorpus-output "${DATA_DIR}/bookcorpus-raw.jsonl"

python3 "$PREPARE_DIR/clean.py" \
    --input "${DATA_DIR}/allenai-c4-raw.jsonl" \
    --output "${DATA_DIR}/allenai-c4.jsonl"

python3 "$PREPARE_DIR/clean.py" \
    --input "${DATA_DIR}/bookcorpus-raw.jsonl" \
    --output "${DATA_DIR}/bookcorpus.jsonl"

echo "Encoding C4 dataset..."
python3 "$PREPARE_DIR/encode.py" \
    --input "${DATA_DIR}/allenai-c4.jsonl" \
    --output-dir "${DATA_DIR}" \
    --output-name "c4" \
    --seq-length "${SL}" \
    --max-samples "${DATA_SAMPLES}" \
    --train-split "${TRAIN_SPLIT}"

echo "Encoding BookCorpus dataset..."
python3 "$PREPARE_DIR/encode.py" \
    --input "${DATA_DIR}/bookcorpus.jsonl" \
    --output-dir "${DATA_DIR}" \
    --output-name "bc" \
    --seq-length "${SL}" \
    --max-samples "${DATA_SAMPLES}" \
    --train-split "${TRAIN_SPLIT}"

python3 "$PREPARE_DIR/verify.py" \
    --input-dir "${DATA_DIR}"
