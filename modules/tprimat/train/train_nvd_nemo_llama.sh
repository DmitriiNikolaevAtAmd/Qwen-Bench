#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPRIMAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$TPRIMAT_PATH/config.env"

mkdir -p "$TPRIMAT_PATH/output"

export OUTPUT_DIR="$TPRIMAT_PATH/output"
export DATA_DIR
export SEED=${SEED:-42}

export TP=${TP:-1}
export PP=${PP:-1}
export DP=${DP:-8}
export GA=${GA:-8}

export MBS=${MBS:-1}
export GBS=$((MBS * DP * GA))

echo "Config: TP=${TP} PP=${PP} DP=${DP} GA=${GA}"
echo "Batch: MBS=${MBS} GBS=${GBS} SL=${SL}"
echo "Seed: ${SEED}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=ERROR
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_MODULE_LOADING=EAGER
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# NCCL_NVLS_ENABLE disabled for fair cross-platform comparison
export NCCL_NVLS_ENABLE=0

# Maximum performance: enable TransformerEngine fused/flash attention kernels
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=1
unset NVTE_UNFUSED_ATTN 2>/dev/null || true

# FP8 disabled for fair cross-platform comparison (no AMD equivalent)
export FP8_HYBRID=${FP8_HYBRID:-false}
export FP8_PARAM=${FP8_PARAM:-false}

export PROFILING=${PROFILING:-false}
export PROFILE_WAIT=${PROFILE_WAIT:-5}
export PROFILE_WARMUP=${PROFILE_WARMUP:-1}
export PROFILE_ACTIVE=${PROFILE_ACTIVE:-2}
export PROFILE_REPEAT=${PROFILE_REPEAT:-1}

DATASET="${DATASET:-bc}"
export DATASET
DATA_PREFIX="${DATA_DIR}/${DATASET}-train"

if [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
    echo "ERROR: Data files not found at ${DATA_PREFIX}.bin/.idx"
    echo "       Run shell/data.sh first to generate the data"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training llama (nemo) on dataset: ${DATASET}"
echo "=========================================="
echo "Dataset: ${DATA_PREFIX} (${DATASET})"

LOG_FILE="$TPRIMAT_PATH/output/training_nvd_nemo_llama_${DATASET}.log"
python3 -u "$SCRIPT_DIR/train_nemo.py" llama 2>&1 | tee "$LOG_FILE"
