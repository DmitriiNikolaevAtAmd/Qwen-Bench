#!/bin/bash
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"

if [ "$PLATFORM" = "amd" ]; then
    DOCKER_ARGS+=(--network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
fi

source "$ROOT_DIR/config.env"

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "Data not found in $DATA_DIR, running data preparation..."
    "$SCRIPT_DIR/run_data.sh"
fi

"$SCRIPT_DIR/build.sh"

run_docker \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --env-file config.env \
    --env-file secrets.env \
    "$IMAGE" bash shell/train.sh
