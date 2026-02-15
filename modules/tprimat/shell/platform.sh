#!/bin/bash
# Detect GPU platform and set common project paths.
# Source this script; do NOT execute it directly.
#
# Sets:
#   SCRIPT_DIR   – absolute path to shell/
#   ROOT_DIR     – project root (parent of shell/)
#   PLATFORM     – "nvd" or "amd"
#   IMAGE        – docker image tag  (tprimat:nvd / tprimat:amd)
#   DOCKER_ARGS  – base docker flags for GPU pass-through (array)
#
# Defines:
#   run_docker [args...] – run a container with project mounts and GPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if command -v nvidia-smi &>/dev/null; then
    PLATFORM="nvd"
    DOCKER_ARGS=(--gpus all)
elif [ -e /dev/kfd ]; then
    PLATFORM="amd"
    DOCKER_ARGS=(--device=/dev/kfd --device=/dev/dri --group-add video)
else
    echo "ERROR: No GPU runtime detected (need nvidia-smi or /dev/kfd)"
    exit 1
fi

IMAGE="tprimat:${PLATFORM}"

cd "$ROOT_DIR"

run_docker() {
    docker run --rm \
        "${DOCKER_ARGS[@]}" \
        -v "$(pwd)":/workspace/code \
        -v /data:/data \
        -w /workspace/code \
        "$@"
}
