#!/bin/bash
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"

run_docker \
    --network=host \
    --env-file config.env \
    --env-file secrets.env \
    "$IMAGE" bash shell/data.sh "$@"
