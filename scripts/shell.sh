#!/bin/bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -e

    if [ "$PLATFORM" = "rocm" ]; then
        DOCKER_ARGS+=(--network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
    fi

    if [ $# -eq 0 ]; then
        set -- fish
    fi

    run_docker \
        -it \
        --name "$CONTAINER" \
        --shm-size=64g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p 8000:8000 \
        "$IMAGE" "$@"
fi
