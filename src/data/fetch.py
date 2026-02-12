#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

DATA_DIR = os.environ.get("DATA_DIR", "/data")


def fetch_pseudo_camera(num_samples: int, output_file: str):
    """Fetch bghira/pseudo-camera-10k captions to JSONL via bulk snapshot download."""
    from huggingface_hub import snapshot_download

    repo_id = "bghira/pseudo-camera-10k"

    # Bulk-download only caption/*.txt files (parallel, cached)
    print(f"Downloading captions from {repo_id}...")
    local_dir = snapshot_download(
        repo_id,
        repo_type="dataset",
        allow_patterns=["caption/*.txt"],
    )

    caption_dir = Path(local_dir) / "caption"
    txt_files = sorted(caption_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} caption files")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from alive_progress import alive_bar

    total = min(len(txt_files), num_samples) if num_samples else len(txt_files)
    saved = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as fout, \
         alive_bar(total, title="Processing captions") as bar:
        for txt_file in txt_files:
            if num_samples and saved >= num_samples:
                break

            text = txt_file.read_text(encoding="utf-8").strip()
            if not text:
                skipped += 1
                bar()
                continue

            json.dump({"text": text}, fout, ensure_ascii=False)
            fout.write("\n")
            saved += 1
            bar()

    print(f"Saved {saved} captions to {output_file} (skipped {skipped} empty)")


def main():
    parser = argparse.ArgumentParser(description="Fetch pseudo-camera-10k dataset and tokenizers")
    parser.add_argument(
        "--samples",
        type=int,
        default=int(os.environ.get("DATA_SAMPLES", 10000)),
        help="Number of samples to fetch (default: DATA_SAMPLES env or 10K)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"{DATA_DIR}/pseudo-camera-raw.jsonl",
        help="Output JSONL file path",
    )
    
    args = parser.parse_args()
    
    fetch_pseudo_camera(args.samples, args.output)


if __name__ == "__main__":
    main()
