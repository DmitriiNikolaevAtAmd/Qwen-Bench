#!/usr/bin/env python3
"""
Simple Primus training script — uses standard MI300X configs as-is.

Usage:
    torchrun --nproc_per_node=8 train/train_prim.py llama
    torchrun --nproc_per_node=8 train/train_prim.py qwen
"""

import argparse
import os
import sys
from pathlib import Path

# =====================================================================
# Hardcoded parameters
# =====================================================================
PRIMUS_PATH = Path("/workspace/Primus")
DATA_DIR = Path("/data/tprimat")
NUM_GPUS = 8

DATASET = "bc"
DATA_PREFIX = str(DATA_DIR / f"{DATASET}-train")

MODELS = {
    "llama": {
        "configs": [
            "examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml",
            "examples/megatron/configs/MI300X/llama3.1_8B-pretrain.yaml",
        ],
        "tokenizer": str(DATA_DIR / "tokenizers/llama"),
        "tokenizer_hf": "meta-llama/Llama-3.1-8B",
    },
    "qwen": {
        "configs": [
            "examples/megatron/configs/MI300X/qwen2.5_7B-BF16-pretrain.yaml",
            "examples/megatron/configs/MI300X/qwen2.5_7B-pretrain.yaml",
        ],
        "tokenizer": str(DATA_DIR / "tokenizers/qwen"),
        "tokenizer_hf": "Qwen/Qwen2.5-7B",
    },
}

# =====================================================================
# AMD / ROCm env
# =====================================================================
for k, v in {
    "RCCL_DEBUG": "ERROR", "NCCL_DEBUG": "ERROR", "GLOO_LOG_LEVEL": "ERROR",
    "RCCL_MSCCL_ENABLE": "0", "RCCL_MSCCLPP_ENABLE": "0",
    "HSA_NO_SCRATCH_RECLAIM": "1", "HSA_ENABLE_SDMA": "1",
    "HSA_FORCE_FINE_GRAIN_PCIE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1", "GPU_MAX_HW_QUEUES": "2",
    "TOKENIZERS_PARALLELISM": "false", "TRANSFORMERS_VERBOSITY": "error",
    "TORCH_CPP_LOG_LEVEL": "ERROR",
}.items():
    os.environ.setdefault(k, v)

os.environ.setdefault("NNODES", "1")
os.environ.setdefault("NODE_RANK", "0")
os.environ.setdefault("NGPUS_PER_NODE", str(NUM_GPUS))


# =====================================================================
# Launch Primus training
# =====================================================================

def launch_training(config_path, model_name):
    primus_str = str(PRIMUS_PATH)
    if primus_str not in sys.path:
        sys.path.insert(0, primus_str)
    os.environ["PYTHONPATH"] = f"{primus_str}:{os.environ.get('PYTHONPATH', '')}"

    from primus.pretrain import launch_pretrain_trainer, setup_backend_path, setup_env
    from primus.core.launcher.parser import load_primus_config

    # Resolve tokenizer: prefer local, fall back to HuggingFace
    info = MODELS[model_name]
    tokenizer = info["tokenizer"] if os.path.isdir(info["tokenizer"]) else info["tokenizer_hf"]

    args = argparse.Namespace(
        config=str(config_path),
        data_path=str(DATA_DIR),
        backend_path=None,
        export_config=None,
    )
    setup_env(data_path=args.data_path)

    overrides = [
        # Training
        "modules.pre_trainer.overrides.train_iters=50",
        "modules.pre_trainer.overrides.log_interval=1",
        "modules.pre_trainer.overrides.log_throughput=true",
        # Real data instead of mock
        "modules.pre_trainer.overrides.mock_data=false",
        f"modules.pre_trainer.overrides.train_data_path={DATA_PREFIX}",
        "modules.pre_trainer.overrides.tokenizer_type=HuggingFaceTokenizer",
        f"modules.pre_trainer.overrides.tokenizer_model={tokenizer}",
    ]
    primus_cfg, _ = load_primus_config(args, overrides=overrides)

    framework = primus_cfg.get_module_config("pre_trainer").framework
    setup_backend_path(framework=framework, backend_path=None, verbose=True)

    # Build Megatron C++ dataset helpers if missing (rank 0 only)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        import importlib
        try:
            importlib.import_module("megatron.core.datasets.helpers_cpp")
        except (ImportError, ModuleNotFoundError):
            import subprocess
            megatron_ds = Path(sys.modules["megatron"].__path__[0]) / "core" / "datasets"
            makefile = megatron_ds / "Makefile"
            if makefile.exists():
                subprocess.check_call(["make", "-C", str(megatron_ds)])
            else:
                subprocess.check_call(
                    [sys.executable, "setup.py", "build_ext", "--inplace"],
                    cwd=str(megatron_ds),
                )

    # Barrier so other ranks wait for the build
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()

    launch_pretrain_trainer(primus_cfg=primus_cfg, extra_args=[])


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    model_name = sys.argv[1].lower() if len(sys.argv) > 1 else "llama"

    if model_name not in MODELS:
        print(f"Unknown model: {model_name}. Choose from: {list(MODELS.keys())}")
        sys.exit(1)

    config_path = None
    for candidate in MODELS[model_name]["configs"]:
        p = PRIMUS_PATH / candidate
        if p.exists():
            config_path = p
            break

    if config_path is None:
        print(f"Config not found. Tried: {[str(PRIMUS_PATH / c) for c in MODELS[model_name]['configs']]}")
        sys.exit(1)

    launch_training(config_path, model_name)
