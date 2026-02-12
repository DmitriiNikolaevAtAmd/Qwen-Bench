#!/usr/bin/env python3
"""
LLaMA Factory dataset utilities.

Converts cleaned JSONL (from src/data/clean.py) into formats consumed by LLaMA Factory
and generates the required dataset_info.json registry.

LLaMA Factory dataset formats:
  - sharegpt: list of {"conversations": [{"from": "human", "value": "..."}, ...]}
  - alpaca:   list of {"instruction": "...", "input": "...", "output": "..."}
"""
import json
import os
from pathlib import Path
from typing import Optional


def jsonl_to_sharegpt(
    input_path: str,
    output_path: str,
    *,
    system_prompt: str = "You are a helpful assistant.",
    max_samples: Optional[int] = None,
) -> int:
    """Convert cleaned JSONL ({"text": "..."}) to LLaMA Factory sharegpt format.

    Each text entry becomes a single-turn conversation where the human provides
    an image description prompt and the assistant responds with the caption text.

    Args:
        input_path: Path to cleaned JSONL file.
        output_path: Path to write the sharegpt JSON file.
        system_prompt: System message for each conversation.
        max_samples: Maximum number of samples to convert (None = all).

    Returns:
        Number of conversations written.
    """
    conversations = []
    count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get("text", "").strip()
            if not text:
                continue

            entry = {
                "conversations": [
                    {"from": "human", "value": "Describe this image in detail."},
                    {"from": "gpt", "value": text},
                ],
            }
            if system_prompt:
                entry["system"] = system_prompt

            conversations.append(entry)
            count += 1

            if max_samples and count >= max_samples:
                break

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"Converted {count} samples to sharegpt format: {output_path}")
    return count


def jsonl_to_alpaca(
    input_path: str,
    output_path: str,
    *,
    max_samples: Optional[int] = None,
) -> int:
    """Convert cleaned JSONL to LLaMA Factory alpaca format.

    Args:
        input_path: Path to cleaned JSONL file.
        output_path: Path to write the alpaca JSON file.
        max_samples: Maximum number of samples to convert (None = all).

    Returns:
        Number of records written.
    """
    records = []
    count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get("text", "").strip()
            if not text:
                continue

            records.append({
                "instruction": "Describe this image in detail.",
                "input": "",
                "output": text,
            })
            count += 1

            if max_samples and count >= max_samples:
                break

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Converted {count} samples to alpaca format: {output_path}")
    return count


def generate_dataset_info(
    data_dir: str,
    datasets: dict,
    output_path: Optional[str] = None,
) -> str:
    """Generate a LLaMA Factory dataset_info.json.

    Args:
        data_dir: Directory containing dataset files.
        datasets: Dict mapping dataset names to their config, e.g.:
            {
                "pseudo_camera": {
                    "file_name": "pseudo_camera_sharegpt.json",
                    "formatting": "sharegpt",
                }
            }
        output_path: Where to write dataset_info.json.
            Defaults to <data_dir>/dataset_info.json.

    Returns:
        Path to the written dataset_info.json.
    """
    if output_path is None:
        output_path = os.path.join(data_dir, "dataset_info.json")

    # Build the registry with LLaMA Factory schema
    registry = {}
    for name, cfg in datasets.items():
        entry = {"file_name": cfg["file_name"]}
        formatting = cfg.get("formatting", "sharegpt")
        entry["formatting"] = formatting

        if formatting == "sharegpt":
            entry["columns"] = cfg.get("columns", {"messages": "conversations"})
        elif formatting == "alpaca":
            entry["columns"] = cfg.get("columns", {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            })

        # Optional fields
        for key in ("subset", "split", "ranking", "folder"):
            if key in cfg:
                entry[key] = cfg[key]

        registry[name] = entry

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"Dataset info written to: {output_path}")
    return output_path


def check_dataset_exists(data_dir: str, dataset_name: str) -> bool:
    """Check if a LLaMA Factory dataset is registered and its file exists."""
    info_path = Path(data_dir) / "dataset_info.json"
    if not info_path.exists():
        return False

    with open(info_path) as f:
        registry = json.load(f)

    if dataset_name not in registry:
        return False

    file_name = registry[dataset_name].get("file_name", "")
    data_file = Path(data_dir) / file_name
    return data_file.exists()


def get_dataset_stats(data_path: str) -> dict:
    """Get basic statistics for a LLaMA Factory JSON dataset.

    Args:
        data_path: Path to the dataset JSON file.

    Returns:
        Dict with num_samples, avg_turns, and avg_response_length.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return {"num_samples": 0}

    num_samples = len(data)
    total_turns = 0
    total_response_len = 0

    for entry in data:
        convs = entry.get("conversations", [])
        total_turns += len(convs)
        for msg in convs:
            if msg.get("from") == "gpt":
                total_response_len += len(msg.get("value", ""))

    gpt_count = sum(
        1 for entry in data
        for msg in entry.get("conversations", [])
        if msg.get("from") == "gpt"
    )

    return {
        "num_samples": num_samples,
        "avg_turns": round(total_turns / num_samples, 1) if num_samples else 0,
        "avg_response_length": round(total_response_len / gpt_count) if gpt_count else 0,
    }
