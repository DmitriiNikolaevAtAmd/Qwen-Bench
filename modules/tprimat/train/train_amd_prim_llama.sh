#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPRIMAT_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"
export PRIMUS_PATH

source "$TPRIMAT_PATH/config.env"

# Avoid EADDRINUSE: use configurable port (Primus default 1234 often in use after previous run)
export MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p "$TPRIMAT_PATH/output"
export OUTPUT_DIR="$TPRIMAT_PATH/output"

TOKENIZER_PATH="${DATA_DIR}/tokenizers/llama"
TOKENIZER_HF="meta-llama/Llama-3.1-8B"

if [ -d "${TOKENIZER_PATH}" ]; then
    TOKENIZER_MODEL="${TOKENIZER_PATH}"
    echo "Using local tokenizer: ${TOKENIZER_MODEL}"
else
    TOKENIZER_MODEL="${TOKENIZER_HF}"
    echo "Using HuggingFace tokenizer: ${TOKENIZER_MODEL}"
fi

NUM_GPUS=$((TP * PP * DP))
GBS=$((MBS * DP * GA))
LR_DECAY_ITERS=$TRAIN_ITERS

echo "Config: TP=${TP} PP=${PP} DP=${DP} GA=${GA}"
echo "Batch: MBS=${MBS} GBS=${GBS} SL=${SL}"
echo "Seed: ${SEED}"

export RCCL_DEBUG=ERROR
export NCCL_DEBUG=ERROR
export GLOO_LOG_LEVEL=ERROR
export RCCL_MSCCL_ENABLE=0  # Disabled for fair cross-platform comparison (matches NVD NCCL_NVLS_ENABLE=0)
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_SDMA=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Maximum performance: ROCm-optimized GEMM and HIP kernels
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIPBLASLT_FORCE_REDUCE_SCATTERING=1
export GPU_MAX_HW_QUEUES=2
export ROCM_FORCE_HIGH_PERF=1

# TunableOp: use pre-cached GEMM kernels but do NOT tune during benchmark
# Run a warmup pass with TUNING=1 beforehand to populate the cache
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_VERBOSE=0

# Device-side kernel arguments — lower dispatch latency
export HIP_FORCE_DEV_KERNARG=1

export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_SHOW_CPP_STACKTRACES=0

export PROFILING=${PROFILING:-false}
export PROFILE_WAIT=${PROFILE_WAIT:-5}
export PROFILE_WARMUP=${PROFILE_WARMUP:-1}
export PROFILE_ACTIVE=${PROFILE_ACTIVE:-2}
export PROFILE_REPEAT=${PROFILE_REPEAT:-1}

# Disable GC for fair comparison (NVD side uses GCCallback in Python)
python3 -c "import site; open(site.getsitepackages()[0] + '/disable_gc.pth', 'w').write('import gc; gc.disable()')" 2>/dev/null || true
python3 -c "import site; open(site.getsitepackages()[0] + '/primus.pth', 'w').write('$PRIMUS_PATH')" 2>/dev/null || true
export PYTHONPATH="$PRIMUS_PATH:${PYTHONPATH:-}"

CONFIG_FILE="$TPRIMAT_PATH/config/llama3.1_8B-BF16-pretrain.yaml"

TRAIN_SCRIPT="./examples/run_pretrain.sh"
if [ ! -f "$PRIMUS_PATH/$TRAIN_SCRIPT" ]; then
    TRAIN_SCRIPT="./examples/train.sh"
fi

if [ ! -f "$PRIMUS_PATH/$TRAIN_SCRIPT" ]; then
    echo "ERROR: Neither run_pretrain.sh nor train.sh found in $PRIMUS_PATH/examples/"
    exit 1
fi

DATASET="${DATASET:-bc}"
DATA_PREFIX="${DATA_DIR}/${DATASET}-train"
DATA_CACHE_PATH="${DATA_DIR}/index_cache"
mkdir -p "$DATA_CACHE_PATH"

if [ ! -f "${DATA_PREFIX}.bin" ] || [ ! -f "${DATA_PREFIX}.idx" ]; then
    echo "ERROR: Data files not found at ${DATA_PREFIX}.bin/.idx"
    echo "       Run shell/data.sh first to generate the data"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training llama (prim) on dataset: ${DATASET}"
echo "=========================================="
echo "Dataset: ${DATA_PREFIX} (${DATASET})"

cd "$PRIMUS_PATH"

bash "$TPRIMAT_PATH/shell/apply_primus_patches.sh"

PATCHED_CONFIG="$TPRIMAT_PATH/output/llama3.1_8B-BF16-pretrain.yaml"
export PATCHED_CONFIG TP PP GBS MBS SL GA TRAIN_ITERS WARMUP_STEPS LR WEIGHT_DECAY BETA1 BETA2 DATA_PREFIX TOKENIZER_MODEL SEED OUTPUT_DIR
if python3 -c "import yaml" 2>/dev/null; then
    python3 "$TPRIMAT_PATH/utils/configure.py" "$CONFIG_FILE" "$PATCHED_CONFIG" --profiling
else
    echo "WARNING: pyyaml not available, copying unpatched config"
    cp "$CONFIG_FILE" "$PATCHED_CONFIG"
fi

export EXP="$PATCHED_CONFIG"

MEMORY_LOG="$TPRIMAT_PATH/output/memory_llama_${DATASET}.log"
(
    while true; do
        if command -v rocm-smi &>/dev/null; then
            rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU|Used" >> "$MEMORY_LOG"
        elif command -v nvidia-smi &>/dev/null; then
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$MEMORY_LOG"
        fi
        sleep 2
    done
) &
MEMORY_PID=$!

LOG_FILE="$TPRIMAT_PATH/output/training_main_llama_${DATASET}.log"
: > "$LOG_FILE"
tail -f "$LOG_FILE" &
TAIL_PID=$!

PROFILE_ARGS=""
if [ "$PROFILING" = "true" ]; then
    PROFILE_DIR="$TPRIMAT_PATH/output/profiles"
    mkdir -p "$PROFILE_DIR"
    PROFILE_ARGS="--profile --use_pytorch_profiler"
    echo "[TPrimat] Profiling enabled for Primus training"
    echo "[TPrimat] Profile traces will be saved to: $PROFILE_DIR"
fi

bash "$TRAIN_SCRIPT" \
    --train_iters "$TRAIN_ITERS" \
    --global_batch_size "$GBS" \
    --micro_batch_size "$MBS" \
    --seq_length "$SL" \
    --tensor_model_parallel_size "$TP" \
    --pipeline_model_parallel_size "$PP" \
    --lr "$LR" \
    --min_lr 0.0 \
    --lr_warmup_iters "$WARMUP_STEPS" \
    --lr_decay_style cosine \
    --lr_decay_iters "$TRAIN_ITERS" \
    --weight_decay "$WEIGHT_DECAY" \
    --data_path "$DATA_PREFIX" \
    --tokenizer_type HuggingFaceTokenizer \
    --tokenizer_model "$TOKENIZER_MODEL" \
    --split 100,0,0 \
    --data-cache-path "$DATA_CACHE_PATH" \
    --seed "$SEED" \
    --use_flash_attn \
    $PROFILE_ARGS \
    >> "$LOG_FILE" 2>&1 || true

# Check if training actually completed (profiler may crash during shutdown)
if grep -q "after training is done" "$LOG_FILE"; then
    echo "[TPrimat] Training completed successfully (post-training exit ignored)"
else
    echo "[TPrimat] ERROR: Training failed. Last 80 lines of log:"
    tail -n 80 "$LOG_FILE"
    exit 1
fi

kill $TAIL_PID 2>/dev/null || true
wait $TAIL_PID 2>/dev/null || true

kill $MEMORY_PID 2>/dev/null || true

cd "$TPRIMAT_PATH"

MEMORY_ARG=""
if [ -f "$MEMORY_LOG" ]; then
    MEMORY_ARG="--memory-log $MEMORY_LOG"
fi

python3 evaluate/extract.py \
    --log-file "$TPRIMAT_PATH/output/training_main_llama_${DATASET}.log" \
    --model-name "llama" \
    --dataset "$DATASET" \
    --output "$TPRIMAT_PATH/output/train_amd_prim_llama_${DATASET}.json" \
    --num-gpus "$NUM_GPUS" \
    --global-batch-size "$GBS" \
    --micro-batch-size "$MBS" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --sequence-length "$SL" \
    --parallel-strategy "TP${TP}_PP${PP}_DP${DP}" \
    $MEMORY_ARG

rm -f "$MEMORY_LOG"
