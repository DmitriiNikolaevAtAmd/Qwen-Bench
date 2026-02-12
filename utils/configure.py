#!/usr/bin/env python3
"""
Generate or patch a LLaMA Factory training config YAML from environment variables.

Environment variables are read from config.env (sourced by shell scripts) or set directly.
Static defaults live in the base YAML; this module only overrides values that are set in env.

Usage:
    python3 -m utils.configure <output_yaml>
    python3 -m utils.configure <input_yaml> <output_yaml>
    python3 utils/configure.py <output_yaml> --base configs/base.yaml
"""
import argparse
import os
import sys
from pathlib import Path

import yaml


# Mapping from env var names to LLaMA Factory config keys.
# Values are (yaml_key, type_converter, default_if_missing_or_None).
ENV_MAP = {
    # Model
    "MODEL_NAME": ("model_name_or_path", str, None),
    "TEMPLATE": ("template", str, None),
    # Dataset
    "DATASET": ("dataset", str, None),
    "CUTOFF_LEN": ("cutoff_len", int, 2048),
    # Fine-tuning
    "FINETUNING_TYPE": ("finetuning_type", str, "lora"),
    "LORA_RANK": ("lora_rank", int, 8),
    "LORA_ALPHA": ("lora_alpha", int, 16),
    "LORA_TARGET": ("lora_target", str, "all"),
    "LORA_DROPOUT": ("lora_dropout", float, 0.05),
    # Training hyperparameters
    "SEED": ("seed", int, 42),
    "NUM_TRAIN_EPOCHS": ("num_train_epochs", float, 1.0),
    "MAX_STEPS": ("max_steps", int, -1),
    "PER_DEVICE_BATCH_SIZE": ("per_device_train_batch_size", int, 1),
    "GA": ("gradient_accumulation_steps", int, 8),
    "LR": ("learning_rate", float, 3e-4),
    "LR_SCHEDULER": ("lr_scheduler_type", str, "cosine"),
    "WEIGHT_DECAY": ("weight_decay", float, 0.1),
    "WARMUP_RATIO": ("warmup_ratio", float, 0.1),
    # Precision
    "PRECISION": ("bf16", lambda v: v.lower() in ("bf16", "true"), True),
    # Logging & saving
    "LOGGING_STEPS": ("logging_steps", int, 10),
    "SAVE_STEPS": ("save_steps", int, 500),
    "EVAL_STEPS": ("eval_steps", int, 500),
    # Output
    "OUTPUT_DIR": ("output_dir", str, "./output"),
}


def _parse_deepspeed_stage(stage_str: str) -> str | None:
    """Return the path to a built-in DeepSpeed config, or None."""
    stage = str(stage_str).strip()
    if stage in ("0", "none", ""):
        return None
    # LLaMA Factory ships built-in ds configs as examples/deepspeed/ds_z{N}_config.json
    # Users can also provide a custom path.
    if os.path.isfile(stage):
        return stage
    return f"examples/deepspeed/ds_z{stage}_config.json"


def configure(config: dict | None = None) -> dict:
    """Build or patch a LLaMA Factory config dict from environment variables.

    Args:
        config: Optional base config dict to patch. If None, starts from scratch.

    Returns:
        The patched (or newly created) config dict.
    """
    if config is None:
        config = {}

    # Always set stage to SFT with training enabled
    config.setdefault("stage", "sft")
    config.setdefault("do_train", True)

    # Apply environment overrides
    for env_key, (yaml_key, converter, default) in ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is not None:
            try:
                config[yaml_key] = converter(raw)
            except (ValueError, TypeError):
                if default is not None:
                    config[yaml_key] = default
        elif default is not None and yaml_key not in config:
            config[yaml_key] = default

    # DeepSpeed (handled separately because it's conditional)
    ds_stage = os.environ.get("DEEPSPEED_STAGE", "")
    ds_path = _parse_deepspeed_stage(ds_stage)
    if ds_path:
        config["deepspeed"] = ds_path

    # bf16 precision: if PRECISION is "fp16", switch flags
    precision = os.environ.get("PRECISION", "bf16").lower()
    if precision == "fp16":
        config["bf16"] = False
        config["fp16"] = True
    elif precision == "bf16":
        config["bf16"] = True
        config.pop("fp16", None)

    # Strip LoRA keys when doing full fine-tuning
    if config.get("finetuning_type") == "full":
        for key in ("lora_rank", "lora_alpha", "lora_target", "lora_dropout"):
            config.pop(key, None)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLaMA Factory config YAML from environment variables"
    )
    parser.add_argument(
        "output_yaml",
        help="Path to write the generated/patched config",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Optional base YAML to patch (overrides are applied on top)",
    )
    args = parser.parse_args()

    base_config = None
    if args.base and Path(args.base).exists():
        with open(args.base) as f:
            base_config = yaml.safe_load(f) or {}

    config = configure(base_config)

    output_path = Path(args.output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[Qwen-Bench] Config written to: {output_path}")


if __name__ == "__main__":
    main()
    sys.exit(0)
