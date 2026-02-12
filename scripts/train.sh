#!/bin/bash
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"

if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/train.sh <config.yaml>"
    echo "Example: ./scripts/train.sh configs/qwen2_5vl_lora_sft.yaml"
    exit 1
fi

CONFIG="$1"

if [ "$PLATFORM" = "rocm" ]; then
    DOCKER_ARGS+=(--network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
fi

run_docker \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "$IMAGE" llamafactory-cli train "$CONFIG"
