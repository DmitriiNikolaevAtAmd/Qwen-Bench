#!/bin/bash
set -e

if [ ! -d "output" ]; then
    echo "ERROR: output/ directory not found"
    exit 1
fi

if [ -z "$(find output -type f 2>/dev/null)" ]; then
    echo "ERROR: output/ directory is empty"
    exit 1
fi

zip -q -r output.zip output

exit 0
