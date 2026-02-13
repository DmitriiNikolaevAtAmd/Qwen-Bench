import shutil
from pathlib import Path

from omegaconf import DictConfig
from rich.table import Table

from src import console


def _remove(path: Path, label: str) -> tuple[str, str, str]:
    if path.exists():
        shutil.rmtree(path)
        return (
            f"[bold bright_red]removed[/bold bright_red]",
            f"[bright_white]{label}[/bright_white]",
            f"[cyan]{path}[/cyan]",
        )
    return (
        f"[dim]skipped[/dim]",
        f"[dim]{label}[/dim]",
        f"[dim](not found)[/dim]",
    )


def run(cfg: DictConfig) -> None:
    output_dir = Path(cfg.paths.output_dir)
    hf_home = Path(cfg.paths.hf_home)
    hf_datasets_cache = Path(cfg.paths.hf_datasets_cache)

    rows = []
    rows.append(_remove(output_dir, "output"))
    rows.append(_remove(hf_home, "hf cache"))
    rows.append(_remove(hf_datasets_cache, "datasets cache"))

    if cfg.get("with_data", False):
        data_dir = Path(cfg.paths.data_dir)
        rows.append(_remove(data_dir, "data"))

    table = Table(
        border_style="bright_red",
        header_style="bold bright_red",
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
        "  [bold bright_green]Purge complete[/bold bright_green]"
    )
