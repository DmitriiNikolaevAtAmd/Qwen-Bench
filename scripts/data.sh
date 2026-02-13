#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_SRC="$ROOT_DIR/src/data"

source "$ROOT_DIR/config.env"

# If not inside a container, re-launch via Docker
if [ ! -f /.dockerenv ]; then
    source "$SCRIPT_DIR/platform.sh"

    if [ "$PLATFORM" = "rocm" ]; then
        DOCKER_ARGS+=(--network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
    fi

    run_docker \
        --network=host \
        --env-file config.env \
        --env-file secrets.env \
        "$IMAGE" bash scripts/data.sh "$@"
    exit $?
fi

# --- Running inside container ---
export DATA_DIR DATA_SAMPLES TRAIN_SPLIT SEED HF_HOME HF_DATASETS_CACHE

mkdir -p "$DATA_DIR" "$HF_HOME"

# Step 1: Load image-caption pairs
echo "=== Step 1/3: Loading pseudo-camera-10k images + captions ==="
python3 "$DATA_SRC/load.py" \
    --samples "${DATA_SAMPLES}" \
    --output "${DATA_DIR}/pseudo-camera-raw.jsonl"

# Step 2: Encode into WebDataset shards
echo "=== Step 2/3: Encoding into WebDataset shards ==="
python3 "$DATA_SRC/encode.py" \
    --input "${DATA_DIR}/pseudo-camera-raw.jsonl" \
    --output-dir "${DATA_DIR}/webdataset" \
    --max-samples "${DATA_SAMPLES}" \
    --train-split "${TRAIN_SPLIT}"

# Step 3: Verify shards
echo "=== Step 3/3: Verifying WebDataset shards ==="
python3 "$DATA_SRC/verify.py" \
    --input-dir "${DATA_DIR}/webdataset"

echo "=== Data pipeline complete ==="
