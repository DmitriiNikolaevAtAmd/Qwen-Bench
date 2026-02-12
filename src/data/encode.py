#!/usr/bin/env python3
"""Pack cleaned image-caption pairs into WebDataset tar shards for Megatron-Energon."""
import argparse
import json
import math
import os
import random
from pathlib import Path

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DATA_SAMPLES = int(os.environ.get("DATA_SAMPLES", 50000))
TRAIN_SPLIT = float(os.environ.get("TRAIN_SPLIT", 0.9))
SEED = int(os.environ.get("SEED", 42))


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


def write_shards(records, output_dir: str, split_name: str, max_per_shard: int = 1000):
    """Write records into WebDataset tar shards.

    Each sample in the tar contains:
      <key>.png   -- the raw image bytes
      <key>.txt   -- the caption text
    """
    import webdataset as wds

    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    num_shards = max(1, math.ceil(len(records) / max_per_shard))
    shard_pattern = str(split_dir / f"shard-%05d.tar")

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

    actual_shards = len(list(split_dir.glob("shard-*.tar")))
    print(f"  {split_name}: {written} samples -> {actual_shards} shards in {split_dir}")
    return actual_shards


def main():
    parser = argparse.ArgumentParser(description="Encode image-caption pairs into WebDataset shards")
    parser.add_argument(
        "--input", type=str,
        default=f"{DATA_DIR}/pseudo-camera.jsonl",
        help="Input clean JSONL file",
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

    print(f"Loading records from {args.input}...")
    records = load_records(args.input, args.max_samples)
    print(f"Loaded {len(records)} records")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(records)

    n_train = int(len(records) * args.train_split)
    n_val = (len(records) - n_train) // 2
    n_test = len(records) - n_train - n_val

    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]

    print(f"Split: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_shards(train_records, args.output_dir, "train", args.max_per_shard)
    write_shards(val_records, args.output_dir, "val", args.max_per_shard)
    write_shards(test_records, args.output_dir, "test", args.max_per_shard)

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
    print(f"Split info written to {info_path}")


if __name__ == "__main__":
    main()
