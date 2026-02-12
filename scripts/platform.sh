#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if command -v nvidia-smi &>/dev/null; then
    PLATFORM="cuda"
    COMPOSE_FILE="$ROOT_DIR/docker/cuda-compose.yml"
    CONTAINER="qwen-bench-cuda"
    DOCKER_ARGS=(--gpus all)
elif [ -e /dev/kfd ]; then
    PLATFORM="rocm"
    COMPOSE_FILE="$ROOT_DIR/docker/rocm-compose.yml"
    CONTAINER="qwen-bench-rocm"
    DOCKER_ARGS=(--device=/dev/kfd --device=/dev/dri --group-add video)
else
    echo "ERROR: No GPU runtime detected (need nvidia-smi or /dev/kfd)"
    exit 1
fi

IMAGE="qwen-bench:${PLATFORM}"

cd "$ROOT_DIR"

run_docker() {
    docker run --rm \
        "${DOCKER_ARGS[@]}" \
        -v "$(pwd)":/workspace/code \
        -v /data:/data \
        -w /workspace/code \
        "$@"
}
