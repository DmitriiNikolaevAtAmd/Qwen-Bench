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

STAGE_ICONS = {
    "data": "[bold blue]1[/bold blue]",
    "train": "[bold magenta]2[/bold magenta]",
    "wrap": "[bold yellow]3[/bold yellow]",
    "purge": "[bold red]x[/bold red]",
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
        title="[dim]resolved config[/dim]",
        border_style="dim",
        expand=False,
        padding=(0, 1),
    ))
    console.print()


def _stage_header(name: str) -> None:
    icon = STAGE_ICONS.get(name, "[bold]>[/bold]")
    console.rule(f" {icon}  [bold]{name}[/bold] ")


def _stage_footer(name: str, elapsed: float) -> None:
    console.print(f"  [dim]{name} done in {elapsed:.1f}s[/dim]")
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
            _stage_header(name)
            t0 = time.time()
            stages[name](cfg)
            _stage_footer(name, time.time() - t0)
    else:
        _stage_header(stage)
        stages[stage](cfg)

    elapsed = time.time() - t_total
    console.print()
    console.rule(f"[bold green]done[/bold green] [dim]{elapsed:.1f}s[/dim]")


if __name__ == "__main__":
    main()
