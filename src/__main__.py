import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from src import console

VALID_STAGES = ("data", "train", "wrap", "purge", "all")

STAGE_STYLE = {
    "data": ("bright_blue", "blue"),
    "train": ("bright_magenta", "magenta"),
    "wrap": ("bright_yellow", "yellow"),
    "purge": ("bright_red", "red"),
}


def _banner(stage: str) -> None:
    color, _ = STAGE_STYLE.get(stage, ("white", "white"))
    title = Text.assemble(
        ("Q", "bold bright_cyan"),
        ("W", "bold bright_blue"),
        ("E", "bold bright_magenta"),
        ("N", "bold bright_yellow"),
        ("-", "dim"),
        ("B", "bold bright_green"),
        ("E", "bold bright_cyan"),
        ("N", "bold bright_blue"),
        ("C", "bold bright_magenta"),
        ("H", "bold bright_yellow"),
    )
    console.print()
    console.print(Panel(
        Text.assemble(
            "\n", title, "\n\n",
            ("  stage ", "dim"),
            (stage, f"bold {color}"),
            ("\n", ""),
        ),
        border_style="bright_cyan",
        expand=False,
        padding=(0, 8),
    ))
    console.print()


def _show_config(cfg: DictConfig) -> None:
    yaml_str = OmegaConf.to_yaml(cfg)
    console.print(Panel(
        Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False),
        title="[bold bright_white]Configuration[/bold bright_white]",
        border_style="bright_cyan",
        expand=False,
        padding=(0, 1),
    ))
    console.print()


def _stage_open(name: str) -> None:
    color, _ = STAGE_STYLE.get(name, ("white", "white"))
    console.rule(f"[bold {color}]{name}[/bold {color}]", style=color)
    console.print()


def _stage_close(name: str, elapsed: float) -> None:
    color, _ = STAGE_STYLE.get(name, ("white", "white"))
    console.print()
    console.rule(
        f"[bold {color}]{name}[/bold {color}]  [dim]{elapsed:.1f}s[/dim]",
        style=color,
    )
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

    result = Table.grid(padding=(0, 1))
    result.add_row(
        "[bold bright_green]done[/bold bright_green]",
        f"[dim]{elapsed:.1f}s[/dim]",
    )
    console.print(Panel(
        result,
        border_style="bright_green",
        expand=False,
        padding=(0, 2),
    ))
    console.print()


if __name__ == "__main__":
    main()
