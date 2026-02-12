#!/bin/bash
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"

run_docker "$IMAGE" bash scripts/purge.sh "$@"
