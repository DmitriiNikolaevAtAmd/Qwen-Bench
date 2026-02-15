#!/bin/bash
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"

PLATFORM_ARGS=()
if [ "$PLATFORM" = "rocm" ]; then
    DOCKER_ARGS+=(--cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
    PLATFORM_ARGS+=(training=rocm)
fi

ENV_FILE="server.env"
if [ ! -f "$ENV_FILE" ]; then
    echo " warn  server.env not found — cp server.tpl server.env" >&2
    ENV_FILE="/dev/null"
elif ! grep -q '^HF_TOKEN=.' "$ENV_FILE" || grep -q 'your_token_here' "$ENV_FILE"; then
    echo " warn  HF_TOKEN not set in server.env" >&2
fi

run_docker \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network=host \
    --env-file "$ENV_FILE" \
    "$IMAGE" python -m src "${PLATFORM_ARGS[@]}" "$@"
