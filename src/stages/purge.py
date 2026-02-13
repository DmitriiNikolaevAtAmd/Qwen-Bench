import shutil
from pathlib import Path

from omegaconf import DictConfig

from src import console


def _remove(path: Path, label: str) -> None:
    if path.exists():
        shutil.rmtree(path)
        console.print(f"  [red]\u2716[/red]  {label}  [dim]{path}[/dim]")
    else:
        console.print(f"  [dim]\u2500[/dim]  {label}  [dim](not found)[/dim]")


def run(cfg: DictConfig) -> None:
    output_dir = Path(cfg.paths.output_dir)
    hf_home = Path(cfg.paths.hf_home)
    hf_datasets_cache = Path(cfg.paths.hf_datasets_cache)

    _remove(output_dir, "output")
    _remove(hf_home, "hf cache")
    _remove(hf_datasets_cache, "datasets cache")

    if cfg.get("with_data", False):
        data_dir = Path(cfg.paths.data_dir)
        _remove(data_dir, "data")

    console.print()
    console.print("  [bold green]\u2714[/bold green]  Purge complete")
