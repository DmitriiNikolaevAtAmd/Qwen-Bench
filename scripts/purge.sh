#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Removing output files..."
rm -rf output/*
touch output/.keep

echo "Removing cache..."
rm -rf cache/

if [ "$1" = "--with-data" ]; then
    echo "Removing data..."
    rm -rf data/*
    touch data/.keep
fi

echo "Done"
