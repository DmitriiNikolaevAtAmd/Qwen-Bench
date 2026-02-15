#!/bin/bash
# Launch interactive container. Usage: ./entry.sh [command...]

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -e

    if [ "$PLATFORM" = "nvd" ]; then
        DOCKER_ARGS+=(-e CUDA_LAUNCH_BLOCKING=1)
    elif [ "$PLATFORM" = "amd" ]; then
        DOCKER_ARGS+=(--network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
    fi

    if [ $# -eq 0 ]; then
        set -- fish
    fi

    run_docker \
        -it \
        --name tprimat \
        --shm-size=64g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --env-file config.env \
        --env-file secrets.env \
        "$IMAGE" "$@"
fi
