#!/bin/bash
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"

PLATFORM_ARGS=()
if [ "$PLATFORM" = "rocm" ]; then
    DOCKER_ARGS+=(--cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
    PLATFORM_ARGS+=(training=rocm)
fi

ENV_FILE="config.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Warning: $ENV_FILE not found. Create it from config.tpl:" >&2
    echo "  cp config.tpl config.env && vi config.env" >&2
    ENV_FILE="/dev/null"
fi

run_docker \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network=host \
    --env-file "$ENV_FILE" \
    "$IMAGE" python -m src "${PLATFORM_ARGS[@]}" "$@"
