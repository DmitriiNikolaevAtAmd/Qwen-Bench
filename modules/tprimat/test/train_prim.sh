#!/bin/bash
# Simple Primus training launcher with hardcoded params.
# Usage:
#   train/train_prim.sh          # trains llama (default)
#   train/train_prim.sh llama
#   train/train_prim.sh qwen
set -e

NUM_GPUS=8
MODEL="${1:-llama}"
MASTER_PORT="${MASTER_PORT:-29500}"

# AMD / ROCm performance tuning
export RCCL_MSCCL_ENABLE=1
export RCCL_MSCCLPP_ENABLE=1
export RCCL_MSCCLPP_FORCE_ENABLE=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIPBLASLT_FORCE_REDUCE_SCATTERING=1
export ROCM_FORCE_HIGH_PERF=1
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1
export HIP_FORCE_DEV_KERNARG=1

# Pick free port if default is busy
if ss -tlnp 2>/dev/null | grep -q ":${MASTER_PORT} " || lsof -i ":${MASTER_PORT}" &>/dev/null; then
    MASTER_PORT=$((29500 + RANDOM % 1000))
fi

echo "=== Launching torchrun: ${MODEL} on ${NUM_GPUS} GPUs (port ${MASTER_PORT}) ==="
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    train/train_prim.py "$MODEL"
