# Qwen-Bench

Qwen2.5-VL fine-tuning benchmark comparing AMD and NVIDIA GPUs using LLaMA Factory.

## Quick Start

### 1. Set up secrets

```bash
cp secrets.env.example secrets.env
# edit secrets.env and set your HF_TOKEN
```

### 2. Build and run

**AMD (ROCm)**

```bash
docker compose -f docker/rocm-compose.yml build
docker compose -f docker/rocm-compose.yml up -d
docker exec -it qwen-bench-rocm fish
```

**NVIDIA (CUDA)**

```bash
docker compose -f docker/cuda-compose.yml build
docker compose -f docker/cuda-compose.yml up -d
docker exec -it qwen-bench-cuda fish
```

### 3. Inside the container

```bash
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft.yaml

llamafactory-cli chat --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct

llamafactory-cli webui
```

## Project Structure

```
Qwen-Bench/
├── docker/
│   ├── rocm.Dockerfile
│   ├── rocm-compose.yml
│   ├── cuda.Dockerfile
│   └── cuda-compose.yml
├── reqs/
│   ├── rocm-requirements.txt
│   └── cuda-requirements.txt
├── secrets.env.example
├── .gitignore
└── .dockerignore
```

## Docker Images

| Platform | Dockerfile | Base Image |
|----------|------------|------------|
| AMD | `docker/rocm.Dockerfile` | `rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0` |
| NVIDIA | `docker/cuda.Dockerfile` | `hiyouga/pytorch:th2.6.0-cu124-flashattn2.7.4-cxx11abi0-devel` |

Both images install LLaMA Factory from PyPI with DeepSpeed and metrics support. The HuggingFace cache is mounted from the host at `~/.cache/huggingface` so downloaded models persist across container restarts.
