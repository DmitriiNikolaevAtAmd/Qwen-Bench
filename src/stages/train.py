"""Training stage: build LLaMA Factory config and invoke training.

Flattens the Hydra config groups (model, training, data) into a single
LLaMA Factory-compatible YAML, writes it to a temp file, and calls
``llamafactory-cli train``.
"""
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# Keys that map directly from the Hydra training group to LLaMA Factory YAML.
TRAINING_KEYS = {
    "finetuning_type",
    "lora_rank",
    "lora_alpha",
    "lora_target",
    "lora_dropout",
    "num_train_epochs",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "lr_scheduler_type",
    "weight_decay",
    "warmup_ratio",
    "logging_steps",
    "save_steps",
    "eval_steps",
}


def _resolve_deepspeed(stage: int | str) -> str | None:
    """Return the path to a built-in DeepSpeed config, or None."""
    stage_str = str(stage).strip()
    if stage_str in ("0", "none", ""):
        return None
    if os.path.isfile(stage_str):
        return stage_str
    return f"examples/deepspeed/ds_z{stage_str}_config.json"


def _build_llama_factory_config(cfg: DictConfig) -> dict:
    """Build a plain dict that LLaMA Factory accepts as training YAML."""
    lf: dict = {}

    # Always SFT with training enabled
    lf["stage"] = "sft"
    lf["do_train"] = True

    # Model group
    lf["model_name_or_path"] = cfg.model.model_name_or_path
    lf["template"] = cfg.model.template

    # Dataset group
    lf["dataset"] = cfg.data.dataset
    lf["cutoff_len"] = int(cfg.data.cutoff_len)

    # Training group -- copy direct keys
    training = OmegaConf.to_container(cfg.training, resolve=True)
    for key in TRAINING_KEYS:
        if key in training:
            lf[key] = training[key]

    # Precision handling
    precision = str(training.get("precision", "bf16")).lower()
    if precision == "fp16":
        lf["bf16"] = False
        lf["fp16"] = True
    else:
        lf["bf16"] = True

    # DeepSpeed
    ds_stage = training.get("deepspeed_stage", 0)
    ds_path = _resolve_deepspeed(ds_stage)
    if ds_path:
        lf["deepspeed"] = ds_path

    # Output
    lf["output_dir"] = str(cfg.paths.output_dir)

    # Seed
    lf["seed"] = int(cfg.seed)

    # Strip LoRA keys when doing full fine-tuning
    if lf.get("finetuning_type") == "full":
        for key in ("lora_rank", "lora_alpha", "lora_target", "lora_dropout"):
            lf.pop(key, None)

    return lf


def run(cfg: DictConfig) -> None:
    """Generate LLaMA Factory YAML and launch training."""
    lf_config = _build_llama_factory_config(cfg)

    log.info("LLaMA Factory config:\n%s", yaml.dump(lf_config, default_flow_style=False))

    # Write the resolved config to a temp YAML file
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(lf_config, f, default_flow_style=False, sort_keys=False)
    log.info("Training config written to %s", config_path)

    # Invoke LLaMA Factory
    cmd = ["llamafactory-cli", "train", str(config_path)]
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
