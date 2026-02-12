#!/bin/bash
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"

echo "Detected $PLATFORM GPU"
echo "Building qwen-bench:${PLATFORM} ..."
docker build -f "docker/${PLATFORM}.Dockerfile" -t "$IMAGE" .
echo "Done: $IMAGE"
