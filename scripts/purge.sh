#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/config.env"

echo "Removing output files..."
rm -rf "${OUTPUT_DIR:?}"/*
mkdir -p "$OUTPUT_DIR"

echo "Removing cache..."
rm -rf "$HF_HOME" "$HF_DATASETS_CACHE"

if [ "$1" = "--with-data" ]; then
    echo "Removing data..."
    rm -rf "${DATA_DIR:?}"/*
    mkdir -p "$DATA_DIR"
fi

echo "Done"
