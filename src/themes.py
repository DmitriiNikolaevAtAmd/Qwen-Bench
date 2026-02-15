import src
from omegaconf import DictConfig
from rich.console import Console
from rich.theme import Theme


def apply_theme(theme_cfg: DictConfig) -> None:
    colors = theme_cfg.colors

    rich_theme = Theme({
        "banner": f"bold {colors.banner}",
        "stage.data": f"bold {colors.data}",
        "stage.train": f"bold {colors.train}",
        "stage.eval": f"bold {colors.eval}",
        "stage.wrap": f"bold {colors.wrap}",
        "stage.purge": f"bold {colors.purge}",
        "step": f"bold {colors.primary}",
        "success": f"bold {colors.success}",
        "warn": f"bold {colors.warn}",
        "err": f"bold {colors.error}",
        "path": str(colors.primary),
        "num": f"bold {colors.accent}",
        "key": str(colors.accent),
        "val": str(colors.primary),
        "dim": "dim",
        "ok": f"bold {colors.success}",
        "skip": f"bold {colors.success}",
    })

    src.console = Console(
        theme=rich_theme,
        highlight=False,
        force_terminal=True,
        color_system="truecolor",
    )
