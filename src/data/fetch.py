#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

DATA_DIR = os.environ.get("DATA_DIR", "/data")


def fetch_pseudo_camera(num_samples: int, output_file: str, max_retries: int = 3):
    """Fetch bghira/pseudo-camera-10k captions from the caption/ directory to JSONL."""
    from huggingface_hub import HfApi, hf_hub_download
    import time

    repo_id = "bghira/pseudo-camera-10k"
    api = HfApi()

    # List caption .txt files from the caption/ directory
    for attempt in range(max_retries):
        try:
            tree = list(api.list_repo_tree(
                repo_id, repo_type="dataset", path_in_repo="caption"
            ))
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s before retry [Retry {attempt+1}/{max_retries}].")
                time.sleep(wait)
            else:
                raise

    caption_files = [f for f in tree if f.rfilename.endswith(".txt")]
    print(f"Found {len(caption_files)} caption files in {repo_id}/caption/")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for file_info in caption_files:
            if num_samples and saved >= num_samples:
                break

            for attempt in range(max_retries):
                try:
                    local_path = hf_hub_download(
                        repo_id, file_info.rfilename, repo_type="dataset"
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(5 * (attempt + 1))
                    else:
                        print(f"Warning: skipping {file_info.rfilename}: {e}")
                        local_path = None

            if local_path is None:
                skipped += 1
                continue

            with open(local_path, "r", encoding="utf-8") as fin:
                text = fin.read().strip()

            if not text:
                skipped += 1
                continue

            json.dump({"text": text}, fout, ensure_ascii=False)
            fout.write("\n")
            saved += 1

            if saved % 1000 == 0:
                print(f"  {saved} captions saved...")

    print(f"Saved {saved} captions from pseudo-camera-10k to {output_file} (skipped {skipped} empty)")


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
