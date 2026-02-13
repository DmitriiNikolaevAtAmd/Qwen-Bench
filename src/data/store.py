import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

DATA_DIR = os.environ.get("DATA_DIR", "/data")

console = Console()

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

    with console.status("[bold]Running energon prepare...[/bold]"):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(Panel(
            (result.stdout.strip() + "\n" + result.stderr.strip()).strip(),
            title="[bold red]energon prepare failed[/bold red]",
            border_style="red",
            expand=False,
        ))
        sys.exit(result.returncode)


def verify_nv_meta(input_dir: Path) -> None:
    meta_dir = input_dir / ".nv-meta"
    expected_files = ["dataset.yaml", "split.yaml", ".info.json"]

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("File", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Size", justify="right", style="dim")

    all_ok = True
    for fname in expected_files:
        fpath = meta_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            table.add_row(f".nv-meta/{fname}", "[bold green]OK[/bold green]", f"{size:,} B")
        else:
            table.add_row(f".nv-meta/{fname}", "[bold red]MISSING[/bold red]", "-")
            all_ok = False

    for fname in ["index.sqlite", "index.uuid"]:
        fpath = meta_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            table.add_row(f".nv-meta/{fname}", "[bold green]OK[/bold green]", f"{size:,} B")

    console.print(table)

    if not all_ok:
        console.print("[bold red]Some metadata files are missing[/bold red]")
        sys.exit(1)

    dataset_yaml = meta_dir / "dataset.yaml"
    if dataset_yaml.exists():
        content = dataset_yaml.read_text().rstrip()
        console.print()
        console.print(Syntax(content, "yaml", theme="monokai", line_numbers=False))


def store_metadata(input_dir: str) -> None:
    input_path = Path(input_dir)
    if not input_path.exists():
        console.print(f"[bold red]Directory not found:[/bold red] {input_path}")
        sys.exit(1)

    splits_table = Table(show_header=False, show_edge=False, pad_edge=False)
    splits_table.add_column("Split", style="cyan")
    splits_table.add_column("Shards", justify="right")

    for split in ["train", "val", "test"]:
        split_dir = input_path / split
        if not split_dir.exists():
            console.print(f"[bold red]Split directory not found:[/bold red] {split_dir}")
            sys.exit(1)
        tar_count = len(list(split_dir.glob("*.tar")))
        if tar_count == 0:
            console.print(f"[bold red]No tar files in[/bold red] {split_dir}")
            sys.exit(1)
        splits_table.add_row(split, f"{tar_count} shard(s)")

    console.print(splits_table)

    meta_dir = input_path / ".nv-meta"
    required = ["dataset.yaml", "split.yaml", ".info.json"]
    if all((meta_dir / f).exists() for f in required):
        console.print(Panel(
            f"[bold green]Skipped[/bold green] -- .nv-meta/ already complete\n\n"
            + "\n".join(f"  {f}: {(meta_dir / f).stat().st_size:,} B" for f in required),
            title="store", expand=False,
        ))
        return

    console.print()
    run_energon_prepare(input_path)
    console.print()
    verify_nv_meta(input_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=f"{DATA_DIR}/webdataset")
    args = parser.parse_args()
    store_metadata(args.input_dir)


if __name__ == "__main__":
    main()
