import os
import subprocess
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from rich.syntax import Syntax

from src import console

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
    stage_str = str(stage).strip()
    if stage_str in ("0", "none", ""):
        return None
    if os.path.isfile(stage_str):
        return stage_str
    return f"examples/deepspeed/ds_z{stage_str}_config.json"


def _build_llama_factory_config(cfg: DictConfig) -> dict:
    lf: dict = {}

    lf["stage"] = "sft"
    lf["do_train"] = True
    lf["model_name_or_path"] = cfg.model.model_name_or_path
    lf["template"] = cfg.model.template
    lf["dataset"] = cfg.data.dataset
    lf["cutoff_len"] = int(cfg.data.cutoff_len)

    training = OmegaConf.to_container(cfg.training, resolve=True)
    for key in TRAINING_KEYS:
        if key in training:
            lf[key] = training[key]

    precision = str(training.get("precision", "bf16")).lower()
    if precision == "fp16":
        lf["bf16"] = False
        lf["fp16"] = True
    else:
        lf["bf16"] = True

    ds_stage = training.get("deepspeed_stage", 0)
    ds_path = _resolve_deepspeed(ds_stage)
    if ds_path:
        lf["deepspeed"] = ds_path

    lf["output_dir"] = str(cfg.paths.output_dir)
    lf["seed"] = int(cfg.seed)

    if lf.get("finetuning_type") == "full":
        for key in ("lora_rank", "lora_alpha", "lora_target", "lora_dropout"):
            lf.pop(key, None)

    return lf


def run(cfg: DictConfig) -> None:
    lf_config = _build_llama_factory_config(cfg)

    yaml_str = yaml.dump(lf_config, default_flow_style=False, sort_keys=False)
    console.print(Panel(
        Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False),
        title="[dim]LLaMA Factory config[/dim]",
        border_style="dim",
        expand=False,
        padding=(0, 1),
    ))

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(lf_config, f, default_flow_style=False, sort_keys=False)
    console.print(f"Config written to [cyan]{config_path}[/cyan]")

    cmd = ["llamafactory-cli", "train", str(config_path)]
    console.print(f"Running [bold]{' '.join(cmd)}[/bold]")
    console.print()
    subprocess.run(cmd, check=True)
