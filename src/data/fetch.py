#!/usr/bin/env python3
"""Fetch bghira/pseudo-camera-10k images and captions to paired JSONL."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
REPO_ID = "bghira/pseudo-camera-10k"


def fetch_pseudo_camera(num_samples: int, output_file: str):
    from huggingface_hub import snapshot_download
    from alive_progress import alive_bar

    # Download both images and captions in one snapshot (parallel, cached)
    print(f"Downloading images and captions from {REPO_ID}...")
    local_dir = snapshot_download(
        REPO_ID,
        repo_type="dataset",
        allow_patterns=["train/*.png", "caption/*.txt"],
    )

    image_dir = Path(local_dir) / "train"
    caption_dir = Path(local_dir) / "caption"

    image_files = {f.stem: f for f in image_dir.glob("*.png")}
    caption_files = {f.stem: f for f in caption_dir.glob("*.txt")}

    # Match by filename stem (images and captions share the same stem)
    common_stems = sorted(set(image_files) & set(caption_files))
    print(f"Found {len(image_files)} images, {len(caption_files)} captions, {len(common_stems)} matched pairs")

    if not common_stems:
        raise ValueError("No matched image-caption pairs found")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = min(len(common_stems), num_samples) if num_samples else len(common_stems)
    saved = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as fout, \
         alive_bar(total, title="Pairing images+captions") as bar:
        for stem in common_stems:
            if num_samples and saved >= num_samples:
                break

            caption = caption_files[stem].read_text(encoding="utf-8").strip()
            if not caption:
                skipped += 1
                bar()
                continue

            record = {
                "image": str(image_files[stem]),
                "caption": caption,
            }
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")
            saved += 1
            bar()

    print(f"Saved {saved} pairs to {output_file} (skipped {skipped} empty)")


def main():
    parser = argparse.ArgumentParser(description="Fetch pseudo-camera-10k image-caption pairs")
    parser.add_argument(
        "--samples", type=int,
        default=int(os.environ.get("DATA_SAMPLES", 10000)),
        help="Max number of pairs to fetch (default: DATA_SAMPLES env or 10K)",
    )
    parser.add_argument(
        "--output", type=str,
        default=f"{DATA_DIR}/pseudo-camera-raw.jsonl",
        help="Output JSONL file path",
    )
    args = parser.parse_args()
    fetch_pseudo_camera(args.samples, args.output)


if __name__ == "__main__":
    main()
