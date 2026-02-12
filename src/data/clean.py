#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

DATA_DIR = os.environ.get("DATA_DIR", "/data")


def clean_data(input_file: str, output_file: str, min_chars: int, min_words: int):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = 0
    kept = 0
    
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            line = line.strip()
            
            if not line:
                continue
            
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
            except json.JSONDecodeError:
                continue
            
            total += 1
            
            if not text or not text.strip():
                continue
            
            text = text.strip()
            
            if len(text) < min_chars:
                continue
            
            word_count = len(text.split())
            if word_count < min_words:
                continue
            
            json.dump({"text": text}, fout, ensure_ascii=False)
            fout.write("\n")
            kept += 1
    
    print(f"Cleaned {input_file}: {kept}/{total} records kept")


def main():
    parser = argparse.ArgumentParser(description="Clean raw JSONL data")
    parser.add_argument(
        "--input",
        type=str,
        default=f"{DATA_DIR}/pseudo-camera-raw.jsonl",
        help="Input raw JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"{DATA_DIR}/pseudo-camera.jsonl",
        help="Output clean JSONL file",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Minimum text length in characters (default: 20)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=5,
        help="Minimum word count (default: 5)",
    )
    
    args = parser.parse_args()
    clean_data(args.input, args.output, args.min_chars, args.min_words)


if __name__ == "__main__":
    main()
