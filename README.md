# ekvi:rival

GPU training benchmark comparing AMD and NVIDIA platforms using NeMo/Megatron.

## Quick Start

```bash
# on training servers (for make build):
cp server.tpl server.env   # set HF_TOKEN

# on local machine (for make fetch):
cp local.tpl local.env     # set CUDA_SERVER, ROCM_SERVER, paths

./scripts/build.sh
./scripts/cli.sh stage=data
./scripts/cli.sh stage=train
```

Platform (AMD/NVIDIA) is auto-detected.

## CLI

All pipeline stages are driven by a single Hydra-based Python CLI (`python -m src`).
From the host, use `scripts/cli.sh` which launches Docker and forwards all arguments:

```bash
./scripts/cli.sh stage=<stage> [overrides...]
```

### Stages

| Stage   | Description |
|---------|-------------|
| `data`  | Download dataset, split into WebDataset shards, create Energon metadata |
| `train` | Build NeMo/Megatron recipe and run training |
| `wrap`  | Package output directory into `output.zip` |
| `purge` | Remove outputs and caches |
| `all`   | Run data, train, and wrap in sequence |

### Examples

```bash
# Prepare data
./scripts/cli.sh stage=data

# Train with CUDA config (default)
./scripts/cli.sh stage=train

# Train with ROCm config
./scripts/cli.sh stage=train training=rocm

# Switch model
./scripts/cli.sh stage=train model=llama31_8b

# Override a single hyperparameter
./scripts/cli.sh stage=train training.learning_rate=1e-3

# Sweep across learning rates (Hydra multirun)
./scripts/cli.sh --multirun training.learning_rate=1e-4,3e-4,1e-3

# Full pipeline: data -> train -> wrap
./scripts/cli.sh stage=all

# Purge outputs and caches
./scripts/cli.sh stage=purge
```

### Inside the container

```bash
./scripts/shell.sh
# then:
python -m src stage=train
python -m src stage=train training=rocm
```

## Configuration

Configuration uses [Hydra](https://hydra.cc/) with hierarchical YAML files under `config/`:

```
config/
├── config.yaml              # defaults, paths, stage selector
├── model/
│   ├── qwen2_5vl_7b.yaml   # Qwen 2.5 7B
│   └── llama31_8b.yaml     # Llama 3.1 8B
├── data/
│   └── pseudo_camera.yaml   # dataset, samples, train_split
├── training/
│   ├── cuda.yaml            # CUDA platform config (default)
│   └── rocm.yaml            # ROCm platform config
└── theme/
    ├── nord.yaml
    ├── rainbow.yaml
    └── ...
```

Override any value from the command line:

```bash
./scripts/cli.sh stage=train \
    training.train_iters=1000 \
    training.parallel.tensor=2 \
    training.warmup_steps=100
```

Switch config groups:

```bash
./scripts/cli.sh stage=train training=rocm model=llama31_8b
```

## Shell Scripts

| Script | Description |
|--------|-------------|
| `scripts/platform.sh` | Auto-detect GPU platform, set Docker vars |
| `scripts/build.sh` | Build Docker image |
| `scripts/shell.sh` | Launch interactive container |
| `scripts/cli.sh` | Run any pipeline stage via Docker |

### Build

```bash
./scripts/build.sh
```

### Interactive shell

```bash
./scripts/shell.sh
```

## Project Structure

```
ekviduel/
├── config/                      # Hydra config hierarchy
│   ├── config.yaml
│   ├── model/
│   │   ├── qwen2_5vl_7b.yaml
│   │   └── llama31_8b.yaml
│   ├── data/
│   │   └── pseudo_camera.yaml
│   └── training/
│       ├── cuda.yaml
│       └── rocm.yaml
├── src/
│   ├── __init__.py
│   ├── __main__.py              # Hydra CLI entrypoint
│   ├── themes.py                # Rich theme system
│   ├── stages/                  # Pipeline stage orchestrators
│   │   ├── data.py
│   │   ├── train.py
│   │   ├── wrap.py
│   │   └── purge.py
│   ├── train/                   # Training logic
│   │   ├── args.py              # Megatron CLI args builder
│   │   ├── dataset.py           # Energon/WebDataset GPT dataset
│   │   ├── tokenizer.py         # Tokenizer resolution & HF auth
│   │   └── worker.py            # Megatron pretrain worker
│   └── data/                    # Data processing scripts
│       ├── load.py
│       ├── split.py
│       └── store.py
├── utils/                       # Shared utilities
│   ├── hardware.py
│   ├── dataset.py
│   ├── logging.py
│   └── monitor.py
├── docker/
│   ├── rocm/Dockerfile
│   └── cuda/Dockerfile
├── requirements/
│   ├── rocm/requirements.txt
│   └── cuda/requirements.txt
├── scripts/
│   ├── platform.sh
│   ├── build.sh
│   ├── shell.sh
│   └── cli.sh
├── output/
├── server.tpl
├── local.tpl
├── .gitignore
└── .dockerignore
```

## Docker Images

| Platform | Dockerfile | Base Image |
|----------|------------|------------|
| AMD | `docker/rocm/Dockerfile` | `rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0` |
| NVIDIA | `docker/cuda/Dockerfile` | `hiyouga/pytorch:th2.6.0-cu124-flashattn2.7.4-cxx11abi0-devel` |

Both images install NeMo/Megatron dependencies. The HuggingFace cache is mounted from the host at `/data`.
