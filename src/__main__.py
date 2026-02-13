import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console()

VALID_STAGES = ("data", "train", "wrap", "purge", "all")

STAGE_COLORS = {
    "data": "blue",
    "train": "magenta",
    "wrap": "yellow",
    "purge": "red",
}


def _banner(stage: str) -> None:
    title = Text()
    title.append("QWEN", style="bold cyan")
    title.append("-", style="dim")
    title.append("BENCH", style="bold white")
    console.print()
    console.rule(title)
    console.print(f"  stage = [bold]{stage}[/bold]", highlight=False)
    console.print()


def _show_config(cfg: DictConfig) -> None:
    yaml_str = OmegaConf.to_yaml(cfg)
    console.print(Panel(
        Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False),
        title="[dim]Configuration[/dim]",
        border_style="dim",
        expand=False,
        padding=(0, 1),
    ))
    console.print()


def _stage_open(name: str) -> None:
    color = STAGE_COLORS.get(name, "white")
    console.print(f"[bold {color}]<{name}>[/bold {color}]")
    console.print()


def _stage_close(name: str, elapsed: float) -> None:
    color = STAGE_COLORS.get(name, "white")
    console.print()
    console.print(f"[bold {color}]</{name}>[/bold {color}]  [dim]{elapsed:.1f}s[/dim]")
    console.print()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    stage = cfg.stage
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Unknown stage: '{stage}'. Must be one of {VALID_STAGES}"
        )

    _banner(stage)
    _show_config(cfg)

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

    t_total = time.time()

    if stage == "all":
        for name in ("data", "train", "wrap"):
            _stage_open(name)
            t0 = time.time()
            stages[name](cfg)
            _stage_close(name, time.time() - t0)
    else:
        _stage_open(stage)
        t0 = time.time()
        stages[stage](cfg)
        _stage_close(stage, time.time() - t0)

    elapsed = time.time() - t_total
    console.rule(f"[bold green]done[/bold green] [dim]{elapsed:.1f}s[/dim]")
    console.print()


if __name__ == "__main__":
    main()
