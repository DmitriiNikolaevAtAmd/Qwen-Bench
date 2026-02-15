#!/bin/bash
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"
cd "$ROOT_DIR"

if [ "$PLATFORM" = "nvd" ]; then
    echo "Detected NVIDIA GPU"
elif [ "$PLATFORM" = "amd" ]; then
    echo "Detected AMD GPU"
fi

echo "Building tprimat:${PLATFORM} ..."
docker build -f "${PLATFORM}.Dockerfile" -t "tprimat:${PLATFORM}" .
echo "Done: tprimat:${PLATFORM}"
