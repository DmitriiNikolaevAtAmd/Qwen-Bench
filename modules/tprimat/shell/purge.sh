#!/bin/bash
# Usage: ./shell/purge.sh [--with-data]
set -e

rm -rf cache torchelastic_* *.out *.err *.zip output
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

if [ "$1" == "--with-data" ]; then
    echo "Purging the data at /data/tprimat ..."
    rm -rf /data/tprimat
fi
