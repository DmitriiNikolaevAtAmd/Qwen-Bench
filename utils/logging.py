"""
Log parsing and result formatting for LLaMA Factory / HF Trainer training runs.

Parses:
  - HF Trainer text logs: {'loss': 2.345, 'grad_norm': 1.23, 'learning_rate': 3e-4, ...}
  - trainer_state.json: full log_history with per-step metrics
  - GPU memory from nvidia-smi / rocm-smi logs
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


def round_floats(obj: Any, precision: int = 5) -> Any:
    """Recursively round floats in nested dicts/lists."""
    if isinstance(obj, float):
        if abs(obj) < 0.001 and obj != 0:
            return round(obj, 10)
        return round(obj, precision)
    elif isinstance(obj, dict):
        return {key: round_floats(value, precision) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, precision) for item in obj]
    return obj


def parse_trainer_state(state_path: str) -> Optional[Dict[str, Any]]:
    """Parse a HF Trainer trainer_state.json file.

    Returns a dict with log_history and summary metrics extracted from
    the trainer state, or None if the file doesn't exist.
    """
    if not os.path.exists(state_path):
        return None

    with open(state_path, "r") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    if not log_history:
        return {"log_history": [], "total_steps": 0}

    return {
        "log_history": log_history,
        "total_steps": state.get("global_step", 0),
        "best_metric": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
    }


def extract_metrics_from_state(
    state_path: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Extract training metrics from trainer_state.json.

    Returns (loss_values, learning_rates, grad_norms, epochs).
    """
    loss_values: List[float] = []
    learning_rates: List[float] = []
    grad_norms: List[float] = []
    epochs: List[float] = []

    parsed = parse_trainer_state(state_path)
    if parsed is None:
        return loss_values, learning_rates, grad_norms, epochs

    for entry in parsed["log_history"]:
        if "loss" in entry:
            loss_values.append(float(entry["loss"]))
        if "learning_rate" in entry:
            learning_rates.append(float(entry["learning_rate"]))
        if "grad_norm" in entry:
            grad_norms.append(float(entry["grad_norm"]))
        if "epoch" in entry:
            epochs.append(float(entry["epoch"]))

    return loss_values, learning_rates, grad_norms, epochs


def extract_metrics_from_log(
    log_file: str,
) -> Tuple[List[float], List[float], List[float]]:
    """Extract training metrics from HF Trainer text log output.

    Parses lines like:
        {'loss': 2.345, 'grad_norm': 1.23, 'learning_rate': 0.0003, 'epoch': 0.5}

    Returns (loss_values, learning_rates, grad_norms).
    """
    loss_values: List[float] = []
    learning_rates: List[float] = []
    grad_norms: List[float] = []

    # Pattern to match HF Trainer log dicts (single-quoted keys)
    dict_pattern = re.compile(r"\{[^}]*'loss'[^}]*\}")
    float_val = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")

    with open(log_file, "r") as f:
        for line in f:
            match = dict_pattern.search(line)
            if not match:
                continue

            log_str = match.group(0)

            # Extract individual metrics
            loss_m = re.search(r"'loss'\s*:\s*(" + float_val.pattern + r")", log_str)
            if loss_m:
                loss_values.append(float(loss_m.group(1)))

            lr_m = re.search(r"'learning_rate'\s*:\s*(" + float_val.pattern + r")", log_str)
            if lr_m:
                learning_rates.append(float(lr_m.group(1)))

            gn_m = re.search(r"'grad_norm'\s*:\s*(" + float_val.pattern + r")", log_str)
            if gn_m:
                grad_norms.append(float(gn_m.group(1)))

    return loss_values, learning_rates, grad_norms


def extract_step_times_from_log(log_file: str) -> List[float]:
    """Extract per-step wall-clock times from HF Trainer logs.

    Looks for patterns like:
        'train_steps_per_second': 1.23
        Step X/Y ... [MM:SS<MM:SS, X.XXit/s]

    Returns list of step times in seconds.
    """
    step_times: List[float] = []

    # HF Trainer progress bar: [00:05<00:10, 2.50it/s]
    iter_pattern = re.compile(r"(\d+\.?\d*)\s*it/s")
    # Also: X.XX s/it
    sec_per_it_pattern = re.compile(r"(\d+\.?\d*)\s*s/it")

    with open(log_file, "r") as f:
        for line in f:
            m = iter_pattern.search(line)
            if m:
                its = float(m.group(1))
                if its > 0:
                    step_times.append(1.0 / its)
                continue

            m = sec_per_it_pattern.search(line)
            if m:
                step_times.append(float(m.group(1)))

    return step_times


def parse_memory_log(log_file: str, num_steps: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Parse nvidia-smi or rocm-smi memory log to extract VRAM usage values.

    Converts all values to decimal GB for consistency.

    Args:
        log_file: Path to memory log file.
        num_steps: If provided, interpolate memory values to match step count.

    Returns:
        Dict with memory_values and summary stats, or None if no data found.
    """
    if not os.path.exists(log_file):
        return None

    raw_values: List[float] = []

    with open(log_file, "r") as f:
        for line in f:
            # rocm-smi: "VRAM Total Used Memory (B): 73014444032"
            match = re.search(r"Used.*\(B\)[:\s]+(\d+)", line)
            if match:
                raw_values.append(int(match.group(1)) / 1e9)
                continue

            # rocm-smi: "Used: 69632 MB"
            match = re.search(r"Used[:\s]+(\d+)\s*MB", line, re.IGNORECASE)
            if match:
                raw_values.append(int(match.group(1)) * 1048576 / 1e9)
                continue

            # nvidia-smi CSV: "0, 65432" (index, memory_mib)
            match = re.search(r"^\d+,\s*(\d+)", line)
            if match:
                raw_values.append(int(match.group(1)) * 1048576 / 1e9)
                continue

    if not raw_values:
        return None

    # Interpolate to match step count if requested
    if num_steps and num_steps > 0 and len(raw_values) != num_steps:
        import numpy as np

        raw_indices = np.linspace(0, len(raw_values) - 1, len(raw_values))
        step_indices = np.linspace(0, len(raw_values) - 1, num_steps)
        memory_values = list(np.interp(step_indices, raw_indices, raw_values))
    else:
        memory_values = raw_values

    memory_values = [round(v, 2) for v in memory_values]

    return {
        "memory_values": memory_values,
        "peak_memory_gb": round(max(memory_values), 2),
        "avg_memory_gb": round(sum(memory_values) / len(memory_values), 2),
        "min_memory_gb": round(min(memory_values), 2),
        "raw_samples": len(raw_values),
    }


def print_summary(results: Dict[str, Any]) -> None:
    """Print a formatted benchmark summary to stdout."""
    platform = results.get("platform", "unknown").upper()
    gpu_name = results.get("gpu_info", {}).get("device_name", "Unknown")
    metrics = results.get("performance_metrics", {})

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK SUMMARY - Platform: {platform}")
    print(f"{'=' * 60}")
    print(f"Device: {gpu_name}")
    print(f"GPUs: {results.get('training_config', {}).get('num_gpus', 'N/A')}")
    print(f"Total Steps: {metrics.get('total_steps', 'N/A')}")

    avg_time = metrics.get("avg_step_time_seconds")
    if avg_time:
        print(f"Avg Step Time: {avg_time:.3f}s")

    tps = metrics.get("tokens_per_second")
    if tps:
        tps_gpu = metrics.get("tokens_per_second_per_gpu")
        print(f"\nThroughput:")
        print(f"  Total: {tps:,.0f} tokens/sec")
        if tps_gpu:
            print(f"  Per-GPU: {tps_gpu:,.0f} tokens/sec/GPU")

    mem = results.get("memory_metrics", {})
    if mem:
        peak = mem.get("peak_memory_allocated_gb")
        avg = mem.get("avg_memory_allocated_gb")
        print(f"\nMemory (per GPU):")
        if peak:
            print(f"  Peak: {peak:.2f} GB")
        if avg:
            print(f"  Avg:  {avg:.2f} GB")

    print(f"{'=' * 60}\n")
