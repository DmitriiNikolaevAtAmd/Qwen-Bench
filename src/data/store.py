#!/usr/bin/env python3
"""Store Megatron-Energon metadata (.nv-meta/) for WebDataset shards."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

DATA_DIR = os.environ.get("DATA_DIR", "/data")

console = Console()

SAMPLE_TYPE = "CaptioningSample"
FIELD_MAP = {"image": "png", "caption": "txt"}


def run_energon_prepare(input_dir: Path) -> None:
    """Run energon prepare in non-interactive mode with pre-split folders."""
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

    console.print(f"Running: [dim]{' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        for line in result.stdout.strip().splitlines():
            console.print(f"  {line}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            console.print(f"  [yellow]{line}[/yellow]")

    if result.returncode != 0:
        console.print(f"[bold red]energon prepare failed (exit {result.returncode})[/bold red]")
        sys.exit(result.returncode)


def verify_nv_meta(input_dir: Path) -> None:
    """Check that .nv-meta/ was created with expected files."""
    meta_dir = input_dir / ".nv-meta"
    expected_files = ["dataset.yaml", "split.yaml", ".info.json"]

    table = Table(title="Energon Metadata")
    table.add_column("File", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Size", justify="right")

    all_ok = True
    for fname in expected_files:
        fpath = meta_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            table.add_row(f".nv-meta/{fname}", "[bold green]OK[/bold green]", f"{size:,} B")
        else:
            table.add_row(f".nv-meta/{fname}", "[bold red]MISSING[/bold red]", "-")
            all_ok = False

    # Check for index files (sqlite + uuid)
    for fname in ["index.sqlite", "index.uuid"]:
        fpath = meta_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            table.add_row(f".nv-meta/{fname}", "[bold green]OK[/bold green]", f"{size:,} B")
        else:
            table.add_row(f".nv-meta/{fname}", "[bold yellow]ABSENT[/bold yellow]", "-")

    console.print(table)

    if not all_ok:
        console.print("[bold red]FAIL: Some metadata files are missing[/bold red]")
        sys.exit(1)

    # Show dataset.yaml contents
    dataset_yaml = meta_dir / "dataset.yaml"
    if dataset_yaml.exists():
        console.print(f"\n[dim]{dataset_yaml}:[/dim]")
        console.print(dataset_yaml.read_text().rstrip())


def store_metadata(input_dir: str) -> None:
    """Create Megatron-Energon metadata for WebDataset shards.

    This is the importable entry point used by the Hydra stage.

    Args:
        input_dir: Directory containing WebDataset splits (train/val/test).
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        console.print(f"[bold red]FAIL:[/bold red] directory not found: {input_path}")
        sys.exit(1)

    # Verify split folders exist
    for split in ["train", "val", "test"]:
        split_dir = input_path / split
        if not split_dir.exists():
            console.print(f"[bold red]FAIL:[/bold red] split directory not found: {split_dir}")
            sys.exit(1)
        tar_count = len(list(split_dir.glob("*.tar")))
        if tar_count == 0:
            console.print(f"[bold red]FAIL:[/bold red] no tar files in {split_dir}")
            sys.exit(1)
        console.print(f"  {split}: {tar_count} shard(s)")

    # --- Idempotency check: skip if .nv-meta/ already has all required files ---
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
    parser = argparse.ArgumentParser(
        description="Store Megatron-Energon metadata for WebDataset shards"
    )
    parser.add_argument(
        "--input-dir", type=str,
        default=f"{DATA_DIR}/webdataset",
        help="Directory containing WebDataset splits (train/val/test)",
    )
    args = parser.parse_args()
    store_metadata(args.input_dir)


if __name__ == "__main__":
    main()
