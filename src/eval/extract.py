"""Extract training metrics from Megatron-Core log output.

Megatron logs lines like:
    [2024-01-15 10:30:00] iteration       10/      50 | ...
    elapsed time per iteration (ms): 1234.5 | ...
    lm loss: 8.12345 | learning rate: 3.000E-04 | grad norm: 1.234 | ...
    [Rank 0] (after 10 iterations) memory (MB) | allocated: 42212.64 | ... | reserved: 56926.00 | ...

This module parses those lines into metric series and writes a
minimal JSON file containing only arrays (time series).
"""
import json
import re
from pathlib import Path
from typing import Optional


def _safe_float(s: str, lo: float = 0, hi: float = 1e12) -> Optional[float]:
    """Parse a float, returning None if outside [lo, hi]."""
    try:
        v = float(s)
        return v if lo <= v <= hi else None
    except (ValueError, TypeError):
        return None


# -- Dataset name abbreviations ------------------------------------------------

DATASET_ABBREV = {
    "pseudo_camera": "pc",
    "benchmark": "bc",
    "fineweb": "fw",
    "openwebtext": "owt",
    "redpajama": "rp",
    "slimpajama": "sp",
    "the_pile": "tp",
    "c4": "c4",
}

# -- Framework name abbreviations ----------------------------------------------

FRAMEWORK_ABBREV = {
    "megatron": "mega",
    "nemo": "nemo",
    "primus": "prim",
    "deepspeed": "deep",
    "transformers": "tran",
    "fsdp": "fsdp",
}


def abbrev_dataset(name: str) -> str:
    """Shorten dataset name to 2-3 letter abbreviation."""
    return DATASET_ABBREV.get(name, name[:3] if len(name) > 4 else name)


def abbrev_framework(name: str) -> str:
    """Shorten framework name."""
    return FRAMEWORK_ABBREV.get(name, name[:4] if len(name) > 4 else name)


def extract_from_log(log_file: str) -> dict:
    """Parse a Megatron-Core training log and return time-series dict.

    Returns dict with keys: step_times, loss_values, learning_rates,
    grad_norms, memory_allocated, memory_reserved.
    """
    step_times: list[float] = []
    loss_values: list[float] = []
    learning_rates: list[float] = []
    grad_norms: list[float] = []
    memory_allocated: list[float] = []
    memory_reserved: list[float] = []

    path = Path(log_file)
    if not path.exists():
        return {}

    with open(path, "r") as f:
        for line in f:
            # Step time: "elapsed time per iteration (ms): 1234.5"
            m = re.search(r"elapsed time per iteration \(ms\):\s*([0-9.]+)", line)
            if m:
                step_ms = _safe_float(m.group(1), 0.001, 1e6)
                if step_ms is not None:
                    step_times.append(round(step_ms / 1000.0, 5))

            # Loss: "lm loss: 8.12345E+00"
            m = re.search(r"lm loss:\s*([0-9.Ee+-]+)", line)
            if m:
                loss = _safe_float(m.group(1), 0, 10000)
                if loss is not None:
                    loss_values.append(round(loss, 5))

            # Learning rate: "learning rate: 3.000E-04"
            m = re.search(r"learning rate:\s*([0-9.Ee+-]+)", line)
            if m:
                lr = _safe_float(m.group(1), 0, 1)
                if lr is not None:
                    learning_rates.append(lr)

            # Grad norm: "grad norm: 1.234"
            m = re.search(r"grad norm[:\s]+([0-9.Ee+-]+)", line)
            if m:
                gn = _safe_float(m.group(1), 0, 1e6)
                if gn is not None:
                    grad_norms.append(round(gn, 5))

            # Memory (Megatron format, MB):
            # [Rank 0] (after N iterations) memory (MB) | allocated: 42212.64 | ... | reserved: 56926.00 |
            if "memory (MB)" in line:
                ma = re.search(r"\ballocated:\s*([0-9.]+)", line)
                mr = re.search(r"\breserved:\s*([0-9.]+)", line)
                if ma:
                    mem = _safe_float(ma.group(1), 0, 1e6)
                    if mem is not None:
                        memory_allocated.append(round(mem / 1024.0, 2))  # MB -> GB
                if mr:
                    mem = _safe_float(mr.group(1), 0, 1e6)
                    if mem is not None:
                        memory_reserved.append(round(mem / 1024.0, 2))  # MB -> GB

            # Memory (GB format): "mem-alloc-GB: 72.5"
            if "mem-alloc-GB" in line:
                m = re.search(r"mem-alloc-GB[:\s]+([0-9.]+)", line)
                if m:
                    mem = _safe_float(m.group(1), 0, 500)
                    if mem is not None:
                        memory_allocated.append(round(mem, 2))

            if "mem-reserved-GB" in line:
                m = re.search(r"mem-reserved-GB[:\s]+([0-9.]+)", line)
                if m:
                    mem = _safe_float(m.group(1), 0, 500)
                    if mem is not None:
                        memory_reserved.append(round(mem, 2))

    result = {}
    if step_times:
        result["step_times"] = step_times
    if loss_values:
        result["loss_values"] = loss_values
    if learning_rates:
        result["learning_rates"] = learning_rates
    if grad_norms:
        result["grad_norms"] = grad_norms
    if memory_allocated:
        result["memory_allocated"] = memory_allocated
    if memory_reserved:
        result["memory_reserved"] = memory_reserved

    return result


def save_benchmark(
    data: dict,
    output_dir: str,
    platform: str,
    framework: str = "megatron",
    model: str = "qwen",
    dataset: str = "benchmark",
) -> Path:
    """Save time-series data to JSON.

    Filename: train_{platform}_{framework_abbrev}_{model}_{dataset_abbrev}.json
    """
    fw = abbrev_framework(framework)
    ds = abbrev_dataset(dataset)
    filename = f"train_{platform}_{fw}_{model}_{ds}.json"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / filename

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath
