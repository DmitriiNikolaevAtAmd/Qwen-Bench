import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.syntax import Syntax

console = Console()

VALID_STAGES = ("data", "train", "wrap", "purge", "all")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    stage = cfg.stage
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Unknown stage: '{stage}'. Must be one of {VALID_STAGES}"
        )

    console.rule(f"[bold cyan]Qwen-Bench[/bold cyan]  stage=[bold]{stage}[/bold]")
    console.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="monokai", line_numbers=False))

    os.environ["HF_HOME"] = str(cfg.paths.hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(cfg.paths.hf_datasets_cache)

    from src.stages import data as data_stage
    from src.stages import purge as purge_stage
    from src.stages import train as train_stage
    from src.stages import wrap as wrap_stage

    stages = {
        "data": data_stage.run,
        "train": train_stage.run,
        "wrap": wrap_stage.run,
        "purge": purge_stage.run,
    }

    if stage == "all":
        for name in ("data", "train", "wrap"):
            console.rule(f"[bold yellow]{name}[/bold yellow]")
            t0 = time.time()
            stages[name](cfg)
            console.print(f"[dim]{name} completed in {time.time() - t0:.1f}s[/dim]\n")
    else:
        t0 = time.time()
        stages[stage](cfg)
        console.print()
        console.rule(f"[bold green]done[/bold green] [dim]{time.time() - t0:.1f}s[/dim]")


if __name__ == "__main__":
    main()
