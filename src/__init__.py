from rich.console import Console
from rich.theme import Theme

theme = Theme({
    "banner": "bold bright_cyan",
    "stage.data": "bold bright_blue",
    "stage.train": "bold bright_magenta",
    "stage.wrap": "bold bright_yellow",
    "stage.purge": "bold bright_red",
    "step": "bold cyan",
    "success": "bold bright_green",
    "warn": "bold yellow",
    "err": "bold bright_red",
    "path": "cyan",
    "num": "bold bright_white",
    "key": "bright_white",
    "val": "bright_cyan",
    "dim": "dim",
    "ok": "bold bright_green",
    "skip": "bold green",
})

console = Console(theme=theme, highlight=False, force_terminal=True, color_system="truecolor")
