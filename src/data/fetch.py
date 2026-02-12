#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

DATA_DIR = os.environ.get("DATA_DIR", "/data")

MODELS = {
    "meta-llama/Llama-3.1-8B": "meta-llama-llama-31-8b",
    "Qwen/Qwen2.5-7B": "qwen-qwen25-7b",
}


def fetch_tokenizers(output_dir: str):
    from transformers import AutoTokenizer
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for hf_name, local_name in MODELS.items():
        local_path = output_path / local_name
        tokenizer_marker = local_path / "tokenizer_config.json"
        
        if tokenizer_marker.exists():
            continue
        
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        tokenizer.save_pretrained(local_path)
    
    return output_path


def fetch_pseudo_camera(num_samples: int, output_file: str, max_retries: int = 3):
    """Fetch bghira/pseudo-camera-10k dataset and extract CogVLM captions to JSONL."""
    from datasets import DownloadConfig
    import time
    
    download_config = DownloadConfig(
        num_proc=1,
        max_retries=5,
    )
    
    for attempt in range(max_retries):
        try:
            dataset = load_dataset(
                "bghira/pseudo-camera-10k",
                split="train",
                trust_remote_code=True,
                download_config=download_config,
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                time.sleep(wait)
            else:
                raise
    
    # Detect the text/caption column
    text_col = None
    for col in ["text", "caption", "description", "prompt"]:
        if col in dataset.column_names:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError(
            f"No text column found in pseudo-camera-10k. "
            f"Available columns: {dataset.column_names}"
        )
    
    print(f"Using column '{text_col}' from pseudo-camera-10k (columns: {dataset.column_names})")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    skipped = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            if num_samples and saved >= num_samples:
                break
            
            text = example.get(text_col, "")
            if not text or not str(text).strip():
                skipped += 1
                continue
            
            text = str(text).strip()
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")
            saved += 1
    
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
    
    output_dir = str(Path(args.output).parent)
    
    fetch_tokenizers(output_dir)
    fetch_pseudo_camera(args.samples, args.output)


if __name__ == "__main__":
    main()
