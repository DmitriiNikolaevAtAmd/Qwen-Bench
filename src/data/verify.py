#!/usr/bin/env python3
"""Verify WebDataset tar shards: completeness, integrity, image-caption pairing."""
import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

DATA_DIR = os.environ.get("DATA_DIR", "/data")

console = Console()


def verify_shard(tar_path: str) -> dict:
    """Verify a single tar shard. Returns dict with counts, sizes, and errors."""
    samples = 0
    total_img_bytes = 0
    total_txt_bytes = 0
    errors = []

    try:
        with tarfile.open(tar_path, "r") as tar:
            members = {m.name: m for m in tar.getmembers()}

            keys = set()
            for name in members:
                key = name.rsplit(".", 1)[0] if "." in name else name
                keys.add(key)

            for key in sorted(keys):
                png_name = f"{key}.png"
                txt_name = f"{key}.txt"

                if png_name not in members:
                    errors.append(f"{key}: missing .png")
                    continue
                if txt_name not in members:
                    errors.append(f"{key}: missing .txt")
                    continue

                png_member = members[png_name]
                txt_member = members[txt_name]

                if png_member.size == 0:
                    errors.append(f"{key}: empty .png")
                    continue

                f = tar.extractfile(png_member)
                if f is None:
                    errors.append(f"{key}: cannot extract .png")
                    continue
                header = f.read(8)
                if not header.startswith(b"\x89PNG"):
                    errors.append(f"{key}: invalid PNG header")
                    continue

                if txt_member.size == 0:
                    errors.append(f"{key}: empty .txt")
                    continue

                total_img_bytes += png_member.size
                total_txt_bytes += txt_member.size
                samples += 1

    except Exception as e:
        errors.append(f"Failed to open tar: {e}")

    return {
        "samples": samples,
        "img_bytes": total_img_bytes,
        "txt_bytes": total_txt_bytes,
        "errors": errors,
        "path": tar_path,
    }


def fmt_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.1f} KB"
    return f"{nbytes} B"


def verify_split(split_dir: str, expected_count: int = None) -> dict:
    """Verify all shards in a split directory. Returns result dict."""
    split_path = Path(split_dir)
    tar_files = sorted(split_path.glob("shard-*.tar"))

    if not tar_files:
        return {
            "shards": 0, "samples": 0, "expected": expected_count,
            "img_bytes": 0, "txt_bytes": 0, "disk_bytes": 0,
            "errors": ["No shards found"],
        }

    total_samples = 0
    total_img_bytes = 0
    total_txt_bytes = 0
    all_errors = []

    for tar_file in tar_files:
        result = verify_shard(str(tar_file))
        total_samples += result["samples"]
        total_img_bytes += result["img_bytes"]
        total_txt_bytes += result["txt_bytes"]
        if result["errors"]:
            all_errors.extend(f"{tar_file.name}: {e}" for e in result["errors"])

    disk_bytes = sum(f.stat().st_size for f in tar_files)

    if expected_count is not None and total_samples != expected_count:
        all_errors.append(f"Count mismatch: {total_samples} != expected {expected_count}")

    return {
        "shards": len(tar_files),
        "samples": total_samples,
        "expected": expected_count,
        "img_bytes": total_img_bytes,
        "txt_bytes": total_txt_bytes,
        "disk_bytes": disk_bytes,
        "errors": all_errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify WebDataset shards")
    parser.add_argument(
        "--input-dir", type=str,
        default=f"{DATA_DIR}/webdataset",
        help="Directory containing WebDataset splits",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        console.print(f"[bold red]FAIL:[/bold red] directory not found: {input_dir}")
        sys.exit(1)

    # Load expected counts from split_info.json
    info_path = input_dir / "split_info.json"
    expected = {}
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        expected = {k: info[k] for k in ["train", "val", "test"] if k in info}
    else:
        console.print("[yellow]Warning:[/yellow] split_info.json not found (cannot check completeness)")

    # Verify each split
    split_results = {}
    for split_name in ["train", "val", "test"]:
        split_dir = input_dir / split_name
        if not split_dir.exists():
            continue
        split_results[split_name] = verify_split(str(split_dir), expected.get(split_name))

    # Build results table
    table = Table(title="Verification Results")
    table.add_column("Split", style="cyan")
    table.add_column("Shards", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Images", justify="right")
    table.add_column("Captions", justify="right")
    table.add_column("Disk", justify="right")
    table.add_column("Status", justify="center")

    all_ok = True
    grand_samples = 0
    grand_disk = 0

    for split_name, res in split_results.items():
        ok = len(res["errors"]) == 0
        if not ok:
            all_ok = False
        grand_samples += res["samples"]
        grand_disk += res["disk_bytes"]

        status = "[bold green]PASS[/bold green]" if ok else "[bold red]FAIL[/bold red]"
        exp_str = str(res["expected"]) if res["expected"] is not None else "-"

        table.add_row(
            split_name,
            str(res["shards"]),
            str(res["samples"]),
            exp_str,
            fmt_size(res["img_bytes"]),
            fmt_size(res["txt_bytes"]),
            fmt_size(res["disk_bytes"]),
            status,
        )

    total_expected = info.get("total", None) if info_path.exists() else None
    exp_total_str = str(total_expected) if total_expected is not None else "-"

    table.add_row(
        "[bold]total[/bold]", "",
        f"[bold]{grand_samples}[/bold]",
        f"[bold]{exp_total_str}[/bold]",
        "", "",
        f"[bold]{fmt_size(grand_disk)}[/bold]",
        "",
    )

    console.print(table)

    # Print errors if any
    for split_name, res in split_results.items():
        if res["errors"]:
            console.print(f"\n[bold red]Errors in {split_name}:[/bold red]")
            for e in res["errors"][:10]:
                console.print(f"  - {e}")
            if len(res["errors"]) > 10:
                console.print(f"  ... and {len(res['errors']) - 10} more")

    # Final verdict
    console.print()
    if all_ok:
        console.print(Rule("[bold green]PASS: All data downloaded and stored correctly[/bold green]"))
    else:
        console.print(Rule("[bold red]FAIL: Some data is missing or corrupted[/bold red]"))

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
