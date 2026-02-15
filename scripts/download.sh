#!/usr/bin/env bash
# Fetch output.zip from CUDA and ROCM servers, unzip into output/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$ROOT_DIR/config.env"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: $CONFIG not found. Copy config.tpl and fill in your values."
    exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

fetch_and_unzip() {
    local server="$1"       # ssh host alias or user@host
    local remote_path="$2"  # full path to output.zip on server
    local label="$3"        # cuda or rocm

    local dest="$ROOT_DIR/output/${label}-output"

    if [[ -z "$server" || "$server" == "user@"*"-host" ]]; then
        echo "  skip  $label — server not configured"
        return
    fi

    local local_zip="$TMP_DIR/${label}-output.zip"

    echo "  fetch  $server:$remote_path"
    scp -q "$server:$remote_path" "$local_zip"

    # Unzip: archive contains output/*, strip that prefix
    local extract="$TMP_DIR/${label}-extract"
    mkdir -p "$extract"
    unzip -qo "$local_zip" -d "$extract"

    # Move output/* into dest
    mkdir -p "$dest"
    if [[ -d "$extract/output" ]]; then
        cp -a "$extract/output/"* "$dest/"
    else
        cp -a "$extract/"* "$dest/"
    fi

    local count
    count=$(find "$dest" -type f | wc -l | tr -d ' ')
    echo "  done   $label — $count files → $dest"
}

echo "Fetching outputs..."
echo
fetch_and_unzip "${CUDA_SERVER:-}" "${CUDA_REMOTE_PATH:-}" "cuda"
fetch_and_unzip "${ROCM_SERVER:-}" "${ROCM_REMOTE_PATH:-}" "rocm"
echo
echo "Run 'make eval' to generate compare.png"
