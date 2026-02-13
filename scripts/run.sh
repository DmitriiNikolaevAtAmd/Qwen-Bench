#!/bin/bash
# Unified Docker dispatcher for Qwen-Bench Hydra CLI.
#
# Usage:
#   ./scripts/run.sh stage=data
#   ./scripts/run.sh stage=train
#   ./scripts/run.sh stage=all
#   ./scripts/run.sh stage=train training.learning_rate=1e-3
#   ./scripts/run.sh stage=train training=full
#   ./scripts/run.sh --multirun training.learning_rate=1e-4,3e-4,1e-3
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"

if [ "$PLATFORM" = "rocm" ]; then
    DOCKER_ARGS+=(--network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
fi

run_docker \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network=host \
    --env-file secrets.env \
    "$IMAGE" python -m src "$@"
