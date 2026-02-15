# Primus-Rival

LLM training benchmark suite for comparing AMD MI300X and NVIDIA H100 GPUs.

Runs identical training workloads across platforms and frameworks, producing unified JSON metrics for fair comparison.

## Supported Configurations

| Platform | GPU | Frameworks | Models |
|----------|-----|------------|--------|
| AMD | MI300X 192GB | Megatron, Primus | Llama 3.1 8B, Qwen 2.5 7B |
| NVIDIA | H100 80GB | Megatron, NeMo | Llama 3.1 8B, Qwen 2.5 7B |

## Quick Start

The entire workflow -- build, prepare data, train, compare -- runs inside Docker. The platform (AMD or NVIDIA) is auto-detected.

```bash
# 1. Copy secrets template and set your HuggingFace token
cp secrets.env.example secrets.env
vim secrets.env

# 2. Build the Docker image
./shell/build.sh

# 3. Prepare datasets (downloads and encodes C4 + BookCorpus)
./shell/run_data.sh

# 4. Run training (all frameworks and models)
./shell/run_train.sh

# 5. Download results
./shell/run_wrap.sh      # creates output.zip
```

Results are written to `output/` as JSON files and comparison plots.

## Configuration

All training parameters live in `config.env`:

```bash
# Dataset: bc (BookCorpus) or c4
DATASET=bc

# Framework: mega, prim, nemo, or all
FRAMEWORK=prim

# Hyperparameters
MBS=1           # Micro batch size per GPU
SL=2048         # Sequence length
LR=3.0e-4       # Peak learning rate
TRAIN_ITERS=500 # Total training iterations

# Parallelism
TP=1            # Tensor Parallel
PP=1            # Pipeline Parallel
DP=8            # Data Parallel (8 GPUs)
GA=8            # Gradient Accumulation
# Global Batch Size = MBS * DP * GA = 1 * 8 * 8 = 64

# Precision
PRECISION=bf16
```

The `FRAMEWORK` variable controls which training backends to run. Cross-platform mapping applies automatically: `prim` on NVIDIA remaps to `nemo`, and `nemo` on AMD remaps to `prim`.

## Fair Comparison Settings

All parameters are kept identical across platforms to ensure a fair benchmark. A dash (`-`) indicates the parameter is not applicable on that platform.

### Training Algorithm

| Parameter | NVIDIA | AMD |
|---|---|---|
| Seed | 42 | 42 |
| Micro Batch Size (MBS) | 2 | 2 |
| Sequence Length (SL) | 2048 | 2048 |
| Gradient Accumulation (GA) | 4 | 4 |
| Global Batch Size (GBS) | 64 | 64 |
| Peak Learning Rate | 3e-4 | 3e-4 |
| LR Decay | cosine, min_lr=0 | cosine, min_lr=0 |
| LR Warmup Steps | 50 | 50 |
| Train Iters | 500 | 500 |
| Weight Decay | 0.1 | 0.1 |
| Adam Beta1 / Beta2 | 0.9 / 0.95 | 0.9 / 0.95 |
| init_method_std | 0.02 | 0.02 |
| Gradient Checkpointing | Disabled | Disabled |
| random_data_seed | 42 | 42 |
| Data Split | 100,0,0 | 100,0,0 |
| Dataset | BookCorpus (bc) | BookCorpus (bc) |
| tokenizer_type | HuggingFaceTokenizer | HuggingFaceTokenizer |
| Checkpointing/Logging | Disabled | Disabled |
| Number of Samples | 50000 | 50000 |

### Parallelism and Precision

| Parameter | NVIDIA | AMD |
|---|---|---|
| tensor_model_parallel_size | 1 | 1 |
| pipeline_model_parallel_size | 1 | 1 |
| data_parallel_size | 8 | 8 |
| gradient_accumulation | 8 | 8 |
| sequence_parallel | False | False |
| context_parallel_size | 1 | 1 |
| precision | bf16 | bf16 |
| fp8 | False | - |
| fp32_residual_connection | False | False |

### Distributed Parallelism

| Parameter | NVIDIA | AMD |
|---|---|---|
| Data Parallel (DP) | 8 | 8 |
| Tensor Parallel (TP) | 1 | 1 |
| Pipeline Parallel (PP) | 1 | 1 |
| Distributed Optimizer | Yes | Yes |
| use_distributed_optimizer | True | True |
| grad_reduce_in_fp32 | False | False |
| overlap_grad_reduce | False | - |
| overlap_param_gather | False | - |

### Fused Operations

| Parameter | NVIDIA | AMD |
|---|---|---|
| bias_activation_fusion | True | True |
| bias_dropout_fusion | True | True |
| masked_softmax_fusion | True | True |
| persist_layer_norm | True | True |
| apply_rope_fusion | True | True |
| cross_entropy_loss_fusion | True | True |

### Attention Backend

| Parameter | NVIDIA | AMD |
|---|---|---|
| use_flash_attn | True | True |
| NVTE_FUSED_ATTN | 1 | - |
| NVTE_FLASH_ATTN | 1 | - |
| enable_primus_turbo | - | True |
| use_turbo_attention | - | True |

## Project Structure

```
tprimat/
├── config.env                      # Training configuration
├── secrets.env.example             # HuggingFace token template
├── amd.Dockerfile                  # AMD image (rocm/primus:v25.11)
├── nvd.Dockerfile                  # NVIDIA image (nvcr.io/nvidia/nemo:25.04)
├── amd-requirements.txt            # AMD Python dependencies
├── nvd-requirements.txt            # NVIDIA Python dependencies
│
├── utils/                           # Shared utilities
│   ├── hardware.py                 # GPU/device detection
│   ├── logging.py                  # Log parsing, extraction, summary
│   ├── monitor.py                  # BenchmarkCallback (Lightning/HF)
│   ├── utils.py                    # Re-exports (backward compatible)
│   ├── configure.py                # Env-based config overrides
│   └── dataset.py                  # Megatron indexed dataset loader
│
├── data/                            # Data preparation pipeline
│   ├── fetch.py                    # Download datasets and tokenizers
│   ├── clean.py                    # Clean and filter raw data
│   ├── encode.py                   # Tokenize to Megatron format (.bin/.idx)
│   └── verify.py                   # Validate prepared datasets
│
├── train/                          # Training scripts
│   ├── train_mega.py               # Megatron training (Python)
│   ├── train_nemo.py               # NeMo training (Python)
│   ├── train_amd_mega_llama.sh     # AMD + Megatron + Llama
│   ├── train_amd_mega_qwen.sh      # AMD + Megatron + Qwen
│   ├── train_amd_prim_llama.sh     # AMD + Primus + Llama
│   ├── train_amd_prim_qwen.sh      # AMD + Primus + Qwen
│   ├── train_nvd_mega_llama.sh     # NVIDIA + Megatron + Llama
│   ├── train_nvd_mega_qwen.sh      # NVIDIA + Megatron + Qwen
│   ├── train_nvd_nemo_llama.sh     # NVIDIA + NeMo + Llama
│   └── train_nvd_nemo_qwen.sh      # NVIDIA + NeMo + Qwen
│
├── evaluate/                        # Results analysis
│   ├── compare.py                  # Generate comparison plots
│   ├── compare.sh                  # Wrapper for compare.py
│   ├── extract.py                  # Extract metrics from training logs
│   ├── fingerprint.py              # Dataset/model fingerprinting
│   ├── probe.py                    # System probing
│   └── validate.py                 # Result validation
│
├── shell/                        # Orchestration
│   ├── platform.sh                 # Auto-detect AMD vs NVIDIA
│   ├── train.sh                    # Run all training jobs
│   ├── data.sh                     # Run data preparation pipeline
│   ├── build.sh                    # Build Docker image (platform auto-detected)
│   ├── purge.sh                    # Remove caches and outputs
│   ├── wrap.sh                     # Zip output directory
│   ├── entry.sh                     # Launch interactive container
│   ├── run_data.sh                 # Prepare data in Docker
│   ├── run_train.sh                # Train in Docker
│   ├── run_purge.sh                # Purge in Docker
│   └── run_wrap.sh                 # Wrap results in Docker
│
└── output/                         # Results (generated)
    ├── train_*.json                # Benchmark results per run
    └── compare.png                 # Comparison plot
```

## Training Scripts

Shell scripts in `train/` follow the naming pattern `train_{platform}_{framework}_{model}.sh`:

| Script | Platform | Framework | Model |
|--------|----------|-----------|-------|
| `train_amd_mega_llama.sh` | AMD | Megatron | Llama 3.1 8B |
| `train_amd_mega_qwen.sh` | AMD | Megatron | Qwen 2.5 7B |
| `train_amd_prim_llama.sh` | AMD | Primus | Llama 3.1 8B |
| `train_amd_prim_qwen.sh` | AMD | Primus | Qwen 2.5 7B |
| `train_nvd_mega_llama.sh` | NVIDIA | Megatron | Llama 3.1 8B |
| `train_nvd_mega_qwen.sh` | NVIDIA | Megatron | Qwen 2.5 7B |
| `train_nvd_nemo_llama.sh` | NVIDIA | NeMo | Llama 3.1 8B |
| `train_nvd_nemo_qwen.sh` | NVIDIA | NeMo | Qwen 2.5 7B |

The orchestrator `shell/train.sh` selects which scripts to run based on the detected platform and the `FRAMEWORK` variable.

## Running Individual Scripts

To run a specific training job instead of the full suite, launch an interactive Docker container and execute the script directly:

```bash
# Start interactive container
./shell/entry.sh

# Inside the container
bash train/train_amd_prim_llama.sh
```

Or run a single framework for both models:

```bash
FRAMEWORK=mega ./shell/run_train.sh
```

## Data Preparation

The preparation pipeline downloads, filters, tokenizes, and validates datasets. Data pipeline deps (`transformers`, `datasets`, `huggingface_hub`) are installed in each container via `amd-requirements.txt` / `nvd-requirements.txt`.

```bash
./shell/run_data.sh
```

Steps:

1. **fetch.py** -- downloads C4 and BookCorpus from HuggingFace
2. **clean.py** -- cleans and filters raw JSONL data
3. **encode.py** -- tokenizes into Megatron indexed format (`.bin` / `.idx`)
4. **verify.py** -- validates dataset integrity

Data is written to `/data/tprimat` by default (configurable via `DATA_DIR` in `config.env`).

## Docker

Images are built from platform-specific Dockerfiles:

| Platform | Dockerfile | Base Image |
|----------|------------|------------|
| AMD | `amd.Dockerfile` | `rocm/primus:v25.11` |
| NVIDIA | `nvd.Dockerfile` | `nvcr.io/nvidia/nemo:25.04` |

Build manually:

```bash
docker build -f amd.Dockerfile -t tprimat:amd .
docker build -f nvd.Dockerfile -t tprimat:nvd .
```

Or use the auto-detecting builder:

```bash
./shell/build.sh
```

### Profiling with Megatron Bridge (NVIDIA / NeMo Run)

Megatron training (`train_mega.py`) uses **Megatron Bridge**’s standard profiling, not a custom Kineto callback.

**Enable PyTorch profiler:**

1. Set environment variables (or in `config.env`):
   - `PROFILING=true`
   - `PROFILE_STEP_START=1` — first step to profile (default 1)
   - `PROFILE_STEP_END=5` — last step to profile (default 5)

2. Run training as usual. If the recipe exposes Bridge config, profiling runs for the given step range on rank 0 and writes under `output/profiles/`.

**If your run uses NeMo Run with an executor (e.g. Slurm):** use the Bridge **PyTorchProfilerPlugin** so the framework sets up the profiler and output path:

```python
from megatron.bridge.recipes.run_plugins import PyTorchProfilerPlugin

plugins = [
    PyTorchProfilerPlugin(
        profile_step_start=1,
        profile_step_end=5,
        profile_ranks=[0],
        record_shapes=True,
        record_memory_history=True,
        memory_snapshot_path="output/profiles/memory_snapshot.pickle",
    )
]
# Pass plugins to run.Experiment when adding the recipe
```

**Config API (when using Bridge ConfigContainer):** set `cfg.profiling = ProfilingConfig(use_pytorch_profiler=True, profile_step_start=..., profile_step_end=..., profile_ranks=[0], record_shapes=True)` on your config before training. Traces can be viewed in TensorBoard.

**Nsys (system-wide):** Bridge also supports `use_nsys_profiler=True` and the `NsysPlugin`; launch the script with `nsys profile ...` as in the Bridge docs.

### Profiling with rocprofv3 (AMD)

The AMD image adds `/opt/rocm/bin` to `PATH` and tries to install `rocprofiler-sdk` so you can use `rocprofv3` inside the container.

**Why the output directory can be empty:** If you run `rocprofv3 ... -- bash shell/train.sh`, the profiler traces **bash**, not the **Python** process that runs the GPU kernels. The HIP/kernel activity happens in child processes, so you get a `tw053` (or `%hostname%`) folder but no trace files. You must profile the process that actually runs on the GPU.

**Recommended: attach to the training process**

1. Start a single training job in the background:
   ```bash
   mkdir -p /workspace/code/output/prof
   FRAMEWORK=prim bash shell/train.sh &
   TRAIN_PID=$!
   ```
2. After the run has started (e.g. 30–60 s), find the **Python** PID (the one using the GPU), e.g. `pgrep -P $TRAIN_PID -f python` or inspect `ps -ef` / `pstree -p $TRAIN_PID`.
3. Attach rocprofv3 (no command after `--`):
   ```bash
   rocprofv3 --attach <PYTHON_PID> --kernel-trace -d /workspace/code/output/prof --output-format csv
   ```
   Run for the steps you care about, then Ctrl+C. Output files appear under `/workspace/code/output/prof/` (often in a `%hostname%/%pid%` subdir). Use `--attach-duration-msec 60000` to auto-detach after 60 s.
4. Stop training when done: `kill $TRAIN_PID 2>/dev/null; wait $TRAIN_PID 2>/dev/null`.

**If you launch by command:** Use `-d` for the output directory (not `-o`). Without `-d`, outputs go to `%hostname%/%pid%` under the current directory. Find with `find . -name '*kernel_trace*' -o -name '*.rocpd'`.

If `rocprofiler-sdk` is not available in the base image’s repos, install it on the host or use an image that includes the full ROCm stack.

## Results

Each training run produces a JSON file in `output/` with unified metrics:

```json
{
  "platform": "amd",
  "gpu_info": {
    "device_name": "AMD Instinct MI300X",
    "device_count": 8,
    "total_memory_gb": 192.0
  },
  "training_config": {
    "max_steps": 500,
    "global_batch_size": 64,
    "micro_batch_size": 1,
    "sequence_length": 2048,
    "num_gpus": 8
  },
  "performance_metrics": {
    "avg_step_time_seconds": 0.45,
    "tokens_per_second": 580000,
    "tokens_per_second_per_gpu": 72500
  },
  "step_times": [],
  "loss_values": []
}
```

After all training jobs complete, `evaluate/compare.py` generates comparison plots automatically.

## Purge

```bash
# Remove caches, outputs, and temp files
./shell/purge.sh

# Also remove downloaded datasets
./shell/purge.sh --with-data
```
