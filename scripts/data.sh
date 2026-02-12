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

# Fix numpy/pandas binary incompatibility in pre-built image
NUMPY_MAJOR=$(python3 -c "import numpy; print(numpy.__version__.split('.')[0])" 2>/dev/null || echo "0")
if [ "$NUMPY_MAJOR" -lt 2 ]; then
    echo "Fixing numpy (v1.x detected, need v2.x for pandas)..."
    python3 -m pip install --no-deps --force-reinstall --no-cache-dir "numpy>=2.2,<2.3"
fi

# Ensure runtime dependencies are available
python3 -c "import alive_progress" 2>/dev/null || \
    python3 -m pip install -q --no-cache-dir "alive-progress>=3.0"
python3 -c "import webdataset" 2>/dev/null || \
    python3 -m pip install -q --no-cache-dir "webdataset>=0.2.100"

mkdir -p "$DATA_DIR" "$HF_HOME"

# Step 1: Fetch image-caption pairs
echo "=== Step 1/4: Fetching pseudo-camera-10k images + captions ==="
python3 "$DATA_SRC/fetch.py" \
    --samples "${DATA_SAMPLES}" \
    --output "${DATA_DIR}/pseudo-camera-raw.jsonl"

# Step 2: Clean data
echo "=== Step 2/4: Cleaning data ==="
python3 "$DATA_SRC/clean.py" \
    --input "${DATA_DIR}/pseudo-camera-raw.jsonl" \
    --output "${DATA_DIR}/pseudo-camera.jsonl"

# Step 3: Encode into WebDataset shards
echo "=== Step 3/4: Encoding into WebDataset shards ==="
python3 "$DATA_SRC/encode.py" \
    --input "${DATA_DIR}/pseudo-camera.jsonl" \
    --output-dir "${DATA_DIR}/webdataset" \
    --max-samples "${DATA_SAMPLES}" \
    --train-split "${TRAIN_SPLIT}"

# Step 4: Verify shards
echo "=== Step 4/4: Verifying WebDataset shards ==="
python3 "$DATA_SRC/verify.py" \
    --input-dir "${DATA_DIR}/webdataset"

echo "=== Data pipeline complete ==="
