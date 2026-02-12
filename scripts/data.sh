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
export DATA_DIR DATA_SAMPLES TRAIN_SPLIT SL HF_HOME HF_DATASETS_CACHE

mkdir -p "$DATA_DIR" "$HF_HOME"

echo "Fetching pseudo-camera-10k captions..."
python3 "$DATA_SRC/fetch.py" \
    --samples "${DATA_SAMPLES}" \
    --output "${DATA_DIR}/pseudo-camera-raw.jsonl"
