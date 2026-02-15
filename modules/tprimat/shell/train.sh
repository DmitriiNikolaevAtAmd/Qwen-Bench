#!/bin/bash
set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/platform.sh"
source "$ROOT_DIR/config.env"

export DATA_DIR
export OUTPUT_DIR
export HF_HOME

mkdir -p "$OUTPUT_DIR"

# FRAMEWORK selects which training backends to run: mega, nemo, prim, or all (default: all)
# Cross-platform mapping: prim (AMD-only) <-> nemo (NVIDIA-only)
FRAMEWORK="${FRAMEWORK:-all}"

if [ "$PLATFORM" = "nvd" ] && [ "$FRAMEWORK" = "prim" ]; then
    echo "[TPrimat] FRAMEWORK=prim is AMD-only; remapping to nemo on NVIDIA"
    FRAMEWORK="nemo"
elif [ "$PLATFORM" = "amd" ] && [ "$FRAMEWORK" = "nemo" ]; then
    echo "[TPrimat] FRAMEWORK=nemo is NVIDIA-only; remapping to prim on AMD"
    FRAMEWORK="prim"
fi

if [ "$PLATFORM" = "nvd" ]; then
    if [ "$FRAMEWORK" = "mega" ] || [ "$FRAMEWORK" = "all" ]; then
        "$ROOT_DIR/train/train_nvd_mega_llama.sh"
        "$ROOT_DIR/train/train_nvd_mega_qwen.sh"
    fi
    if [ "$FRAMEWORK" = "nemo" ] || [ "$FRAMEWORK" = "all" ]; then
        "$ROOT_DIR/train/train_nvd_nemo_llama.sh"
        "$ROOT_DIR/train/train_nvd_nemo_qwen.sh"
    fi
elif [ "$PLATFORM" = "amd" ]; then
    if [ "$FRAMEWORK" = "mega" ] || [ "$FRAMEWORK" = "all" ]; then
        "$ROOT_DIR/train/train_amd_mega_llama.sh"
        "$ROOT_DIR/train/train_amd_mega_qwen.sh"
    fi
    if [ "$FRAMEWORK" = "prim" ] || [ "$FRAMEWORK" = "all" ]; then
        "$ROOT_DIR/train/train_amd_prim_llama.sh"
        "$ROOT_DIR/train/train_amd_prim_qwen.sh"
    fi
else
    echo "ERROR: Unsupported platform '$PLATFORM' with framework '$FRAMEWORK'"
    exit 1
fi

"$ROOT_DIR/evaluate/compare.sh"
"$ROOT_DIR/shell/wrap.sh"
