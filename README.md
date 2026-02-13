# Qwen-Bench

Qwen2.5-VL fine-tuning benchmark comparing AMD and NVIDIA GPUs using LLaMA Factory.

## Quick Start

```bash
cp secrets.env.example secrets.env
# set your HF_TOKEN

./scripts/build.sh
./scripts/entry.sh
```

Platform (AMD/NVIDIA) is auto-detected.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/platform.sh` | Auto-detect GPU platform, set vars |
| `scripts/build.sh` | Build Docker image |
| `scripts/entry.sh` | Launch interactive container |
| `scripts/train.sh` | Run training with a config |
| `scripts/purge.sh` | Remove outputs and caches |

### Build

```bash
./scripts/build.sh
```

### Interactive shell

```bash
./scripts/entry.sh
```

### Train

```bash
./scripts/train.sh configs/qwen2_5vl_lora_sft.yaml
```

Or inside the container:

```bash
llamafactory-cli train configs/qwen2_5vl_lora_sft.yaml
```

### Purge

```bash
./scripts/purge.sh             # remove outputs and cache
./scripts/purge.sh --with-data # also remove data
```

## Project Structure

```
Qwen-Bench/
├── docker/
│   ├── rocm/
│   │   └── Dockerfile
│   └── cuda/
│       └── Dockerfile
├── requirements/
│   ├── rocm/
│   │   └── requirements.txt
│   └── cuda/
│       └── requirements.txt
├── scripts/
│   ├── platform.sh
│   ├── build.sh
│   ├── entry.sh
│   ├── train.sh
│   └── purge.sh
├── configs/
├── data/
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

Both images install LLaMA Factory from PyPI with DeepSpeed and metrics support. The HuggingFace cache is mounted from the host at `~/.cache/huggingface`.
