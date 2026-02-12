#!/usr/bin/env python3
"""Verify WebDataset tar shards: integrity, image-caption pairing, sample counts."""
import argparse
import io
import json
import os
import sys
import tarfile
from pathlib import Path

DATA_DIR = os.environ.get("DATA_DIR", "/data")


def verify_shard(tar_path: str) -> dict:
    """Verify a single tar shard. Returns dict with counts and errors."""
    samples = 0
    errors = []

    try:
        with tarfile.open(tar_path, "r") as tar:
            members = {m.name: m for m in tar.getmembers()}

            # Group by key (stem before the extension)
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

                # Check image is non-empty and starts with PNG header
                png_member = members[png_name]
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

                # Check caption is non-empty
                txt_member = members[txt_name]
                if txt_member.size == 0:
                    errors.append(f"{key}: empty .txt")
                    continue

                samples += 1

    except Exception as e:
        errors.append(f"Failed to open tar: {e}")

    return {"samples": samples, "errors": errors, "path": tar_path}


def verify_split(split_dir: str) -> bool:
    """Verify all shards in a split directory."""
    split_path = Path(split_dir)
    tar_files = sorted(split_path.glob("shard-*.tar"))

    if not tar_files:
        print(f"  No shards found in {split_dir}")
        return False

    total_samples = 0
    total_errors = []
    print(f"  Checking {len(tar_files)} shards...")

    for tar_file in tar_files:
        result = verify_shard(str(tar_file))
        total_samples += result["samples"]
        if result["errors"]:
            total_errors.extend(
                f"{tar_file.name}: {e}" for e in result["errors"]
            )

    print(f"  Samples: {total_samples}")

    if total_errors:
        print(f"  Errors ({len(total_errors)}):")
        for e in total_errors[:10]:
            print(f"    - {e}")
        if len(total_errors) > 10:
            print(f"    ... and {len(total_errors) - 10} more")
        return False

    print(f"  OK")
    return True


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
        print(f"FAIL: directory not found: {input_dir}")
        sys.exit(1)

    # Check split info
    info_path = input_dir / "split_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        print(f"Split info: {json.dumps(info)}")
    else:
        print("Warning: split_info.json not found")

    all_ok = True
    for split_name in ["train", "val", "test"]:
        split_dir = input_dir / split_name
        if not split_dir.exists():
            print(f"\n[{split_name}] Skipped (not found)")
            continue

        print(f"\n[{split_name}]")
        ok = verify_split(str(split_dir))
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("PASS: All shards verified OK")
    else:
        print("FAIL: Some shards have errors")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
