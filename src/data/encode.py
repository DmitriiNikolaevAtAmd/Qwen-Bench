#!/usr/bin/env python3
"""Pack image-caption pairs into WebDataset tar shards for Megatron-Energon."""
import argparse
import json
import math
import os
import random
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
from rich.table import Table

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DATA_SAMPLES = int(os.environ.get("DATA_SAMPLES", 50000))
TRAIN_SPLIT = float(os.environ.get("TRAIN_SPLIT", 0.9))
SEED = int(os.environ.get("SEED", 42))

console = Console()


def load_records(input_file: str, max_samples: int = None):
    """Load image-caption records from JSONL."""
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            records.append(doc)
            if max_samples and len(records) >= max_samples:
                break
    return records


def write_shards(records, output_dir: str, split_name: str, max_per_shard: int, progress, task_id):
    """Write records into WebDataset tar shards.

    Each sample in the tar contains:
      <key>.png   -- the raw image bytes
      <key>.txt   -- the caption text
    """
    import webdataset as wds

    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    shard_pattern = str(split_dir / "shard-%05d.tar")

    written = 0
    with wds.ShardWriter(shard_pattern, maxcount=max_per_shard) as sink:
        for idx, rec in enumerate(records):
            image_path = rec["image"]
            caption = rec["caption"]

            with open(image_path, "rb") as img_f:
                image_bytes = img_f.read()

            sample = {
                "__key__": f"sample_{idx:06d}",
                "png": image_bytes,
                "txt": caption.encode("utf-8"),
            }
            sink.write(sample)
            written += 1
            progress.advance(task_id)

    actual_shards = len(list(split_dir.glob("shard-*.tar")))
    return written, actual_shards


def main():
    parser = argparse.ArgumentParser(description="Encode image-caption pairs into WebDataset shards")
    parser.add_argument(
        "--input", type=str,
        default=f"{DATA_DIR}/pseudo-camera-raw.jsonl",
        help="Input JSONL file with image-caption pairs",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=f"{DATA_DIR}/webdataset",
        help="Output directory for WebDataset shards",
    )
    parser.add_argument(
        "--max-samples", type=int, default=DATA_SAMPLES,
        help=f"Maximum number of samples to process (default: {DATA_SAMPLES})",
    )
    parser.add_argument(
        "--train-split", type=float, default=TRAIN_SPLIT,
        help=f"Fraction for training data (default: {TRAIN_SPLIT})",
    )
    parser.add_argument(
        "--max-per-shard", type=int, default=1000,
        help="Maximum samples per tar shard (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed for shuffle (default: {SEED})",
    )
    args = parser.parse_args()

    console.print(f"Loading records from [cyan]{args.input}[/cyan]...")
    records = load_records(args.input, args.max_samples)
    console.print(f"Loaded [bold]{len(records)}[/bold] records")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(records)

    n_train = int(len(records) * args.train_split)
    n_val = (len(records) - n_train) // 2
    n_test = len(records) - n_train - n_val

    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]

    splits = [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    total_samples = sum(len(r) for _, r in splits)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Writing shards", total=total_samples)
        for split_name, split_records in splits:
            written, num_shards = write_shards(
                split_records, str(output_dir), split_name, args.max_per_shard,
                progress, task,
            )
            results[split_name] = (written, num_shards)

    # Summary table
    table = Table(title="Encoding Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Shards", justify="right")

    for split_name, (written, num_shards) in results.items():
        table.add_row(split_name, str(written), str(num_shards))

    table.add_row("[bold]total[/bold]", f"[bold]{sum(w for w, _ in results.values())}[/bold]", "")
    console.print(table)

    # Write split info for downstream tools
    split_info = {
        "train": len(train_records),
        "val": len(val_records),
        "test": len(test_records),
        "total": len(records),
        "seed": args.seed,
        "max_per_shard": args.max_per_shard,
    }
    info_path = output_dir / "split_info.json"
    with open(info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    console.print(f"Split info written to [cyan]{info_path}[/cyan]")


if __name__ == "__main__":
    main()
