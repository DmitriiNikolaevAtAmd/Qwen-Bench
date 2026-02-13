import shutil
from pathlib import Path

from omegaconf import DictConfig
from rich.table import Table

from src import console


def _remove(cfg: DictConfig, path: Path, label: str) -> tuple[str, str, str]:
    c = cfg.theme.colors
    if path.exists():
        shutil.rmtree(path)
        return (
            f"[bold {c.error}]removed[/bold {c.error}]",
            f"[{c.accent}]{label}[/{c.accent}]",
            f"[{c.primary}]{path}[/{c.primary}]",
        )
    return (
        "[dim]skipped[/dim]",
        f"[dim]{label}[/dim]",
        "[dim](not found)[/dim]",
    )


def run(cfg: DictConfig) -> None:
    c = cfg.theme.colors
    output_dir = Path(cfg.paths.output_dir)
    hf_home = Path(cfg.paths.hf_home)
    hf_datasets_cache = Path(cfg.paths.hf_datasets_cache)

    rows = []
    rows.append(_remove(cfg, output_dir, "output"))
    rows.append(_remove(cfg, hf_home, "hf cache"))
    rows.append(_remove(cfg, hf_datasets_cache, "datasets cache"))

    if cfg.get("with_data", False):
        data_dir = Path(cfg.paths.data_dir)
        rows.append(_remove(cfg, data_dir, "data"))

    table = Table(
        border_style=str(c.purge),
        header_style=f"bold {c.purge}",
        padding=(0, 1),
        show_edge=True,
    )
    table.add_column("Status", justify="center")
    table.add_column("Target")
    table.add_column("Path")

    for status, label, path in rows:
        table.add_row(status, label, path)

    console.print(table)
    console.print(
        f"  [bold {c.success}]Purge complete[/bold {c.success}]"
    )
