# Qwen-Bench

Qwen2.5-VL fine-tuning benchmark comparing AMD and NVIDIA GPUs using LLaMA Factory.

## Quick Start

```bash
cp secrets.env.example secrets.env
# set your HF_TOKEN

./scripts/build.sh
./scripts/run.sh stage=data
./scripts/run.sh stage=train
```

Platform (AMD/NVIDIA) is auto-detected.

## CLI

All pipeline stages are driven by a single Hydra-based Python CLI (`python -m src`).
From the host, use `scripts/run.sh` which launches Docker and forwards all arguments:

```bash
./scripts/run.sh stage=<stage> [overrides...]
```

### Stages

| Stage   | Description |
|---------|-------------|
| `data`  | Download dataset, split into WebDataset shards, create Energon metadata |
| `train` | Generate LLaMA Factory config and run training |
| `wrap`  | Package output directory into `output.zip` |
| `purge` | Remove outputs and caches |
| `all`   | Run data, train, and wrap in sequence |

### Examples

```bash
# Prepare data
./scripts/run.sh stage=data

# Train with default LoRA config
./scripts/run.sh stage=train

# Train with full fine-tuning instead of LoRA
./scripts/run.sh stage=train training=full

# Override a single hyperparameter
./scripts/run.sh stage=train training.learning_rate=1e-3

# Sweep across learning rates (Hydra multirun)
./scripts/run.sh --multirun training.learning_rate=1e-4,3e-4,1e-3

# Full pipeline: data -> train -> wrap
./scripts/run.sh stage=all

# Purge outputs and caches
./scripts/run.sh stage=purge
```

### Inside the container

```bash
./scripts/entry.sh
# then:
python -m src stage=train
python -m src stage=train training.learning_rate=1e-3
```

## Configuration

Configuration uses [Hydra](https://hydra.cc/) with hierarchical YAML files under `config/`:

```
config/
├── config.yaml              # defaults, paths, stage selector
├── model/
│   └── qwen2_5vl_7b.yaml   # model_name_or_path, template
├── data/
│   └── pseudo_camera.yaml   # dataset, samples, train_split
└── training/
    ├── lora.yaml            # LoRA fine-tuning (default)
    └── full.yaml            # full fine-tuning
```

Override any value from the command line:

```bash
./scripts/run.sh stage=train \
    model.model_name_or_path=Qwen/Qwen2.5-VL-3B-Instruct \
    training.per_device_train_batch_size=2 \
    data.samples=10000
```

Switch config groups:

```bash
./scripts/run.sh stage=train training=full
```

## Shell Scripts

| Script | Description |
|--------|-------------|
| `scripts/platform.sh` | Auto-detect GPU platform, set Docker vars |
| `scripts/build.sh` | Build Docker image |
| `scripts/entry.sh` | Launch interactive container |
| `scripts/run.sh` | Run any pipeline stage via Docker |

### Build

```bash
./scripts/build.sh
```

### Interactive shell

```bash
./scripts/entry.sh
```

## Project Structure

```
Qwen-Bench/
├── config/                      # Hydra config hierarchy
│   ├── config.yaml
│   ├── model/
│   │   └── qwen2_5vl_7b.yaml
│   ├── data/
│   │   └── pseudo_camera.yaml
│   └── training/
│       ├── lora.yaml
│       └── full.yaml
├── src/
│   ├── __init__.py
│   ├── __main__.py              # Hydra CLI entrypoint
│   ├── stages/                  # Pipeline stage implementations
│   │   ├── data.py
│   │   ├── train.py
│   │   ├── wrap.py
│   │   └── purge.py
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
│   ├── entry.sh
│   └── run.sh
├── output/
├── secrets.env.example
├── .gitignore
└── .dockerignore
```

## Docker Images

| Platform | Dockerfile | Base Image |
|----------|------------|------------|
| AMD | `docker/rocm/Dockerfile` | `rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0` |
| NVIDIA | `docker/cuda/Dockerfile` | `hiyouga/pytorch:th2.6.0-cu124-flashattn2.7.4-cxx11abi0-devel` |

Both images install LLaMA Factory from PyPI with DeepSpeed and metrics support. The HuggingFace cache is mounted from the host at `/data`.
