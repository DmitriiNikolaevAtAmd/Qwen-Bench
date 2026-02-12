#!/usr/bin/env python3
"""Clean raw image-caption JSONL: validate images exist, filter short captions."""
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
    missing_img = 0
    short_caption = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1

            image_path = doc.get("image", "")
            caption = doc.get("caption", "").strip()

            # Validate image file exists
            if not image_path or not Path(image_path).is_file():
                missing_img += 1
                continue

            # Filter empty / short captions
            if not caption:
                short_caption += 1
                continue
            if len(caption) < min_chars:
                short_caption += 1
                continue
            if len(caption.split()) < min_words:
                short_caption += 1
                continue

            json.dump({"image": image_path, "caption": caption}, fout, ensure_ascii=False)
            fout.write("\n")
            kept += 1

    print(f"Cleaned {input_file}: {kept}/{total} kept "
          f"(missing_img={missing_img}, short_caption={short_caption})")


def main():
    parser = argparse.ArgumentParser(description="Clean raw image-caption JSONL")
    parser.add_argument(
        "--input", type=str,
        default=f"{DATA_DIR}/pseudo-camera-raw.jsonl",
        help="Input raw JSONL file",
    )
    parser.add_argument(
        "--output", type=str,
        default=f"{DATA_DIR}/pseudo-camera.jsonl",
        help="Output clean JSONL file",
    )
    parser.add_argument(
        "--min-chars", type=int, default=20,
        help="Minimum caption length in characters (default: 20)",
    )
    parser.add_argument(
        "--min-words", type=int, default=5,
        help="Minimum caption word count (default: 5)",
    )
    args = parser.parse_args()
    clean_data(args.input, args.output, args.min_chars, args.min_words)


if __name__ == "__main__":
    main()
