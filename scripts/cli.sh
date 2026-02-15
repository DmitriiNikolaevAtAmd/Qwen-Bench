#!/bin/bash
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"

PLATFORM_ARGS=()
if [ "$PLATFORM" = "rocm" ]; then
    DOCKER_ARGS+=(--cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
    PLATFORM_ARGS+=(training=rocm)
fi

run_docker \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network=host \
    --env-file secrets.env \
    "$IMAGE" python -m src "${PLATFORM_ARGS[@]}" "$@"
