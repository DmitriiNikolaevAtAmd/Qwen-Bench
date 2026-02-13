import zipfile
from pathlib import Path

from omegaconf import DictConfig

from src import console


def run(cfg: DictConfig) -> None:
    output_dir = Path(cfg.paths.output_dir)

    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    files = list(output_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    if file_count == 0:
        raise FileNotFoundError(f"Output directory is empty: {output_dir}")

    archive_path = Path("output.zip")
    console.print(f"Packaging [bold]{file_count}[/bold] files from [cyan]{output_dir}[/cyan]")

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in files:
            if fpath.is_file():
                zf.write(fpath, fpath.relative_to(output_dir.parent))

    size = archive_path.stat().st_size / 1e6
    console.print(f"Archive created: [green]{archive_path}[/green] ([bold]{size:.1f} MB[/bold])")
