import json
import os
import time
import urllib.request
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel

VALID_STAGES = ("data", "train", "eval", "wrap", "purge", "all")

QUOTE_URL = "https://zenquotes.io/api/random"
QUOTE_TIMEOUT = 2


def _get_console():
    import src
    return src.console


def _stage_color(cfg: DictConfig, name: str) -> str:
    return str(getattr(cfg.theme.colors, name, cfg.theme.colors.primary))


def _fetch_quote() -> tuple[str, str] | None:
    try:
        url = f"{QUOTE_URL}?nocache={time.time()}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "ekviduel",
            "Cache-Control": "no-cache",
        })
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

    lines = []
    lines.append(f"[bold {c.primary}]EKVI:DUEL[/bold {c.primary}]")

    quote = _fetch_quote()
    if quote:
        text, author = quote
        lines.append("")
        lines.append(f'[italic {c.accent}]"{text}"[/italic {c.accent}]')
        lines.append(f"[dim]-- {author}[/dim]")

    console.print()
    console.print(Panel(
        "\n".join(lines),
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()


def _save_config(cfg: DictConfig) -> None:
    """Save the resolved configuration (without theme) to output_dir/config.yaml."""
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered = {k: v for k, v in cfg.items() if k != "theme"}
    yaml_str = OmegaConf.to_yaml(filtered)
    (output_dir / "config.yaml").write_text(yaml_str)


def _stage_open(cfg: DictConfig, name: str) -> None:
    console = _get_console()
    color = _stage_color(cfg, name)
    console.rule(f"[bold {color}]<{name.capitalize()}>[/bold {color}]", style=color)
    console.print()


def _stage_close(cfg: DictConfig, name: str) -> None:
    console = _get_console()
    color = _stage_color(cfg, name)
    console.print()
    console.rule(
        f"[bold {color}]</{name.capitalize()}>[/bold {color}]",
        style=color,
    )
    console.print()


def run(cfg: DictConfig) -> None:
    from src.core.themes import apply_theme
    apply_theme(cfg.theme)

    console = _get_console()

    stage = cfg.stage
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Unknown stage: '{stage}'. Must be one of {VALID_STAGES}"
        )

    _banner(cfg, stage)
    _save_config(cfg)

    os.environ["HF_HOME"] = str(cfg.paths.hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(cfg.paths.hf_datasets_cache)

    from src.stage import data as data_stage
    from src.stage import eval as eval_stage
    from src.stage import purge as purge_stage
    from src.stage import train as train_stage
    from src.stage import wrap as wrap_stage

    stages = {
        "data": data_stage.run,
        "train": train_stage.run,
        "eval": eval_stage.run,
        "wrap": wrap_stage.run,
        "purge": purge_stage.run,
    }

    t_total = time.time()

    # When running 'train', automatically chain: train -> eval -> wrap
    # (like tprimat: extract metrics, build charts, package output)
    if stage == "train":
        pipeline = ("train", "eval", "wrap")
    elif stage == "all":
        pipeline = ("data", "train", "eval", "wrap")
    else:
        pipeline = (stage,)

    for name in pipeline:
        _stage_open(cfg, name)
        stages[name](cfg)
        _stage_close(cfg, name)

    elapsed = time.time() - t_total
    mins, secs = divmod(int(elapsed), 60)
    duration = f"{mins}m {secs}s" if mins else f"{secs}s"
    c = cfg.theme.colors
    console.print(Panel(
        f"[bold {c.success}]Done[/bold {c.success}]  [dim]{duration}[/dim]",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()
