import zipfile
from pathlib import Path

from omegaconf import DictConfig
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn,
)
from rich.table import Table

from src import console


def _fmt_size(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.1f} KB"
    return f"{nbytes} B"


def run(cfg: DictConfig) -> None:
    c = cfg.theme.colors
    output_dir = Path(cfg.paths.output_dir)

    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    files = [f for f in output_dir.rglob("*") if f.is_file()]
    if not files:
        raise FileNotFoundError(f"Output directory is empty: {output_dir}")

    archive_path = Path("output.zip")

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Packaging", total=len(files))
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fpath in files:
                zf.write(fpath, fpath.relative_to(output_dir.parent))
                progress.advance(task)

    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim")
    info.add_column()
    info.add_row("archive", str(archive_path))
    info.add_row("files", f"{len(files):,}")
    info.add_row("size", _fmt_size(archive_path.stat().st_size))

    console.print(Panel(
        info,
        title=f"[{c.success}]Packaged[/{c.success}]",
        border_style="dim",
        padding=(1, 2),
    ))
