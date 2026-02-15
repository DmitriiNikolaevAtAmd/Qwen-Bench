#!/bin/bash
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"

# Read HF_TOKEN from server.env if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
if [ -f "$ROOT_DIR/server.env" ]; then
    # shellcheck source=/dev/null
    source "$ROOT_DIR/server.env"
fi

BUILD_ARGS=()
if [ -n "$HF_TOKEN" ]; then
    BUILD_ARGS+=(--build-arg "HF_TOKEN=$HF_TOKEN")
fi

echo "Detected $PLATFORM GPU"
echo "Building ekviduel:${PLATFORM} ..."
docker build "${BUILD_ARGS[@]}" -f "docker/${PLATFORM}/Dockerfile" -t "$IMAGE" .
echo "Done: $IMAGE"
