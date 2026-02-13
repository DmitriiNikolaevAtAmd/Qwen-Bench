import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
from rich.table import Table

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DATA_SAMPLES = int(os.environ.get("DATA_SAMPLES", 50000))
TRAIN_SPLIT = float(os.environ.get("TRAIN_SPLIT", 0.9))
SEED = int(os.environ.get("SEED", 42))

console = Console()


def load_records(input_file: str, max_samples: int = None):
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
    import webdataset as wds

    split_dir = Path(output_dir) / split_name
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    shard_pattern = str(split_dir / "shard-%05d.tar")

    written = 0
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    sys.stdout = devnull

    try:
        with wds.ShardWriter(shard_pattern, maxcount=max_per_shard) as sink:
            for idx, rec in enumerate(records):
                image_path = rec["image"]
                caption = rec["caption"]

                with open(image_path, "rb") as img_f:
                    image_bytes = img_f.read()

                sample = {
                    "__key__": f"{split_name}_{idx:06d}",
                    "png": image_bytes,
                    "txt": caption.encode("utf-8"),
                }
                sink.write(sample)
                written += 1
                progress.advance(task_id)
    finally:
        sys.stdout = old_stdout
        devnull.close()

    actual_shards = len(list(split_dir.glob("shard-*.tar")))
    return written, actual_shards


def split_shards(
    input_file: str,
    output_dir: str,
    max_samples: int = 50000,
    train_split: float = 0.9,
    max_per_shard: int = 1000,
    seed: int = 42,
) -> None:
    info_path = Path(output_dir) / "split_info.json"
    if info_path.exists():
        with open(info_path) as f:
            prev = json.load(f)
        if (prev.get("seed") == seed
                and prev.get("train_split") == train_split
                and prev.get("max_samples") == max_samples
                and prev.get("max_per_shard") == max_per_shard):
            console.print(Panel(
                f"[bold green]Skipped[/bold green] -- shards match current params\n\n"
                f"  Dir:     {output_dir}\n"
                f"  Total:   {prev['total']:,} samples\n"
                f"  Splits:  train={prev['train']}, val={prev['val']}, test={prev['test']}\n"
                f"  Seed:    {prev['seed']}",
                title="split", expand=False,
            ))
            return

    nv_meta = Path(output_dir) / ".nv-meta"
    if nv_meta.exists():
        shutil.rmtree(nv_meta)

    with console.status("[bold]Loading records...[/bold]"):
        records = load_records(input_file, max_samples)
    console.print(f"Loaded [bold]{len(records):,}[/bold] records from [cyan]{input_file}[/cyan]")

    random.seed(seed)
    random.shuffle(records)

    n_train = int(len(records) * train_split)
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

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = {}
    total_samples = sum(len(r) for _, r in splits)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Writing shards", total=total_samples)
        for split_name, split_records in splits:
            written, num_shards = write_shards(
                split_records, output_dir, split_name, max_per_shard,
                progress, task,
            )
            results[split_name] = (written, num_shards)

    table = Table(title="Shards", show_edge=False, pad_edge=False)
    table.add_column("Split", style="cyan")
    table.add_column("Samples", justify="right", style="bold")
    table.add_column("Shards", justify="right")

    for split_name, (written, num_shards) in results.items():
        table.add_row(split_name, f"{written:,}", str(num_shards))

    table.add_row(
        "[bold]total[/bold]",
        f"[bold green]{sum(w for w, _ in results.values()):,}[/bold green]",
        "",
    )
    console.print(table)

    split_info = {
        "train": len(train_records),
        "val": len(val_records),
        "test": len(test_records),
        "total": len(records),
        "seed": seed,
        "train_split": train_split,
        "max_samples": max_samples,
        "max_per_shard": max_per_shard,
    }
    info_path = out_path / "split_info.json"
    with open(info_path, "w") as f:
        json.dump(split_info, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=f"{DATA_DIR}/pseudo-camera-raw.jsonl")
    parser.add_argument("--output-dir", type=str, default=f"{DATA_DIR}/webdataset")
    parser.add_argument("--max-samples", type=int, default=DATA_SAMPLES)
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--max-per-shard", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    split_shards(
        input_file=args.input,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        train_split=args.train_split,
        max_per_shard=args.max_per_shard,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
