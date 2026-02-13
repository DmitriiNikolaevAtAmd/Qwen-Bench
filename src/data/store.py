import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src import console

DATA_DIR = os.environ.get("DATA_DIR", "/data")

SAMPLE_TYPE = "CaptioningSample"
FIELD_MAP = {"image": "png", "caption": "txt"}


def run_energon_prepare(input_dir: Path) -> None:
    cmd = [
        sys.executable, "-m", "megatron.energon.cli.main", "prepare",
        "--non-interactive",
        "--force-overwrite",
        "--split-parts", "train:train/.*",
        "--split-parts", "val:val/.*",
        "--split-parts", "test:test/.*",
        "--sample-type", SAMPLE_TYPE,
        "--field-map", json.dumps(FIELD_MAP),
        str(input_dir),
    ]

    with console.status(
        "[bold bright_cyan]Running energon prepare...[/bold bright_cyan]",
        spinner="dots",
    ):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        output = (result.stdout.strip() + "\n" + result.stderr.strip()).strip()
        console.print(Panel(
            output,
            title="[bold bright_red]energon prepare failed[/bold bright_red]",
            border_style="bright_red",
            expand=False,
            padding=(1, 2),
        ))
        sys.exit(result.returncode)

    console.print("  [bright_green]ok[/bright_green]  energon prepare")


def verify_nv_meta(input_dir: Path, syntax_theme: str = "one-dark") -> None:
    meta_dir = input_dir / ".nv-meta"
    expected_files = ["dataset.yaml", "split.yaml", ".info.json"]

    table = Table(
        title="[bold bright_white]Metadata[/bold bright_white]",
        border_style="bright_blue",
        header_style="bold bright_cyan",
        padding=(0, 1),
    )
    table.add_column("File", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Size", justify="right", style="dim")

    all_ok = True
    for fname in expected_files:
        fpath = meta_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            table.add_row(
                f".nv-meta/{fname}",
                "[bold bright_green]OK[/bold bright_green]",
                f"{size:,} B",
            )
        else:
            table.add_row(
                f".nv-meta/{fname}",
                "[bold bright_red]MISSING[/bold bright_red]",
                "-",
            )
            all_ok = False

    for fname in ["index.sqlite", "index.uuid"]:
        fpath = meta_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            table.add_row(
                f".nv-meta/{fname}",
                "[bold bright_green]OK[/bold bright_green]",
                f"{size:,} B",
            )

    console.print(table)

    if not all_ok:
        console.print("[bold bright_red]Some metadata files are missing[/bold bright_red]")
        sys.exit(1)

    dataset_yaml = meta_dir / "dataset.yaml"
    if dataset_yaml.exists():
        content = dataset_yaml.read_text().rstrip()
        console.print(Panel(
            Syntax(content, "yaml", theme=syntax_theme, line_numbers=False, background_color="default"),
            title="[dim]dataset.yaml[/dim]",
            border_style="bright_blue",
            expand=False,
            padding=(0, 1),
        ))


def store_metadata(input_dir: str, syntax_theme: str = "one-dark") -> None:
    input_path = Path(input_dir)
    if not input_path.exists():
        console.print(f"[bold bright_red]Directory not found:[/bold bright_red] {input_path}")
        sys.exit(1)

    for split in ["train", "val", "test"]:
        split_dir = input_path / split
        if not split_dir.exists():
            console.print(f"[bold bright_red]Split directory not found:[/bold bright_red] {split_dir}")
            sys.exit(1)
        tar_count = len(list(split_dir.glob("*.tar")))
        if tar_count == 0:
            console.print(f"[bold bright_red]No tar files in[/bold bright_red] {split_dir}")
            sys.exit(1)
        console.print(
            f"  [bright_white]{split:<6}[/bright_white]"
            f"[bright_cyan]{tar_count}[/bright_cyan] shard(s)"
        )

    meta_dir = input_path / ".nv-meta"
    required = ["dataset.yaml", "split.yaml", ".info.json"]
    if all((meta_dir / f).exists() for f in required):
        info = Table.grid(padding=(0, 2))
        info.add_column(style="bright_white")
        info.add_column(style="bright_cyan")
        for f in required:
            info.add_row(f, f"{(meta_dir / f).stat().st_size:,} B")

        console.print(Panel(
            info,
            title="[bold bright_green]Skipped -- .nv-meta/ complete[/bold bright_green]",
            border_style="green",
            expand=False,
            padding=(0, 2),
        ))
        return

    console.print()
    run_energon_prepare(input_path)
    console.print()
    verify_nv_meta(input_path, syntax_theme=syntax_theme)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=f"{DATA_DIR}/webdataset")
    args = parser.parse_args()
    store_metadata(args.input_dir)


if __name__ == "__main__":
    main()
