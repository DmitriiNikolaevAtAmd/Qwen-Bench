import json
import os
import time
import urllib.request

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

VALID_STAGES = ("data", "train", "wrap", "purge", "all")

QUOTE_URL = "https://zenquotes.io/api/today"
QUOTE_TIMEOUT = 2


def _get_console():
    import src
    return src.console


def _stage_color(cfg: DictConfig, name: str) -> str:
    return str(getattr(cfg.theme.colors, name, cfg.theme.colors.primary))


def _fetch_quote() -> tuple[str, str] | None:
    try:
        req = urllib.request.Request(QUOTE_URL, headers={"User-Agent": "qwen-bench"})
        with urllib.request.urlopen(req, timeout=QUOTE_TIMEOUT) as resp:
            data = json.loads(resp.read())
        if data and isinstance(data, list):
            return data[0].get("q", ""), data[0].get("a", "")
    except Exception:
        pass
    return None


def _banner(cfg: DictConfig, stage: str) -> None:
    console = _get_console()
    c = cfg.theme.colors
    color = _stage_color(cfg, stage)

    parts = []
    parts.append(("", ""))
    parts.append(("\n", ""))
    parts.append(("QWEN-BENCH", f"bold {c.primary}"))
    parts.append(("\n\n", ""))
    parts.append(("  stage ", "dim"))
    parts.append((stage, f"bold {color}"))

    quote = _fetch_quote()
    if quote:
        text, author = quote
        parts.append(("\n\n", ""))
        parts.append((f'  "{text}"', f"italic {c.accent}"))
        parts.append(("\n", ""))
        parts.append((f"  -- {author}", f"dim"))

    parts.append(("\n", ""))

    console.print()
    console.print(Panel(
        Text.assemble(*parts),
        border_style=str(c.primary),
        expand=False,
        padding=(0, 6),
    ))
    console.print()


def _show_config(cfg: DictConfig) -> None:
    console = _get_console()
    yaml_str = OmegaConf.to_yaml(cfg)
    console.print(Panel(
        Syntax(yaml_str, "yaml", theme=cfg.theme.syntax, line_numbers=False, background_color="default"),
        title=f"[bold {cfg.theme.colors.accent}]Configuration[/bold {cfg.theme.colors.accent}]",
        border_style=str(cfg.theme.colors.primary),
        expand=False,
        padding=(0, 1),
    ))
    console.print()


def _stage_open(cfg: DictConfig, name: str) -> None:
    console = _get_console()
    color = _stage_color(cfg, name)
    console.rule(f"[bold {color}]{name}[/bold {color}]", style=color)
    console.print()


def _stage_close(cfg: DictConfig, name: str, elapsed: float) -> None:
    console = _get_console()
    color = _stage_color(cfg, name)
    console.print()
    console.rule(
        f"[bold {color}]{name}[/bold {color}]  [dim]{elapsed:.1f}s[/dim]",
        style=color,
    )
    console.print()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.themes import apply_theme
    apply_theme(cfg.theme)

    console = _get_console()

    stage = cfg.stage
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Unknown stage: '{stage}'. Must be one of {VALID_STAGES}"
        )

    _banner(cfg, stage)
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
            _stage_open(cfg, name)
            t0 = time.time()
            stages[name](cfg)
            _stage_close(cfg, name, time.time() - t0)
    else:
        _stage_open(cfg, stage)
        t0 = time.time()
        stages[stage](cfg)
        _stage_close(cfg, stage, time.time() - t0)

    elapsed = time.time() - t_total

    c = cfg.theme.colors
    result = Table.grid(padding=(0, 1))
    result.add_row(
        f"[bold {c.success}]done[/bold {c.success}]",
        f"[dim]{elapsed:.1f}s[/dim]",
    )
    console.print(Panel(
        result,
        border_style=str(c.success),
        expand=False,
        padding=(0, 2),
    ))
    console.print()


if __name__ == "__main__":
    main()
