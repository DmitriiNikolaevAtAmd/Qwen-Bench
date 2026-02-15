"""Extract training metrics from Megatron-Core log output.

Megatron logs lines like:
    [2024-01-15 10:30:00] iteration       10/      50 | ...
    elapsed time per iteration (ms): 1234.5 | ...
    lm loss: 8.12345 | learning rate: 3.000E-04 | grad norm: 1.234 | ...

This module parses those lines into metric series suitable for the
BenchmarkCallback / compare dashboard.
"""
import re
from pathlib import Path
from typing import Optional

from src.train.monitor import BenchmarkCallback


def _safe_float(s: str, lo: float = 0, hi: float = 1e12) -> Optional[float]:
    """Parse a float, returning None if outside [lo, hi]."""
    try:
        v = float(s)
        return v if lo <= v <= hi else None
    except (ValueError, TypeError):
        return None


def extract_from_log(
    log_file: str,
    output_dir: str = "./output",
    model_name: Optional[str] = None,
    platform: str = "auto",
    framework: str = "megatron",
    dataset: Optional[str] = None,
    num_gpus: int = 1,
    global_batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
) -> BenchmarkCallback:
    """Parse a Megatron-Core training log and return a populated BenchmarkCallback.

    The returned callback can be saved to JSON via ``cb.save()``.
    """
    cb = BenchmarkCallback(
        output_dir=output_dir,
        platform=platform,
        model_name=model_name,
        framework=framework,
        dataset=dataset,
    )
    cb.num_gpus = num_gpus
    cb.global_batch_size = global_batch_size
    cb.sequence_length = sequence_length

    path = Path(log_file)
    if not path.exists():
        return cb

    with open(path, "r") as f:
        for line in f:
            # Step time: "elapsed time per iteration (ms): 1234.5" or "1234.5/5678.9"
            m = re.search(r"elapsed time per iteration \(ms\):\s*([0-9.]+)", line)
            if m:
                step_ms = _safe_float(m.group(1), 0.001, 1e6)
                if step_ms is not None:
                    cb.step_times.append(step_ms / 1000.0)

            # Loss: "lm loss: 8.12345"
            m = re.search(r"lm loss:\s*([0-9.Ee+-]+)", line)
            if m:
                loss = _safe_float(m.group(1), 0, 10000)
                if loss is not None:
                    cb.loss_values.append(loss)

            # Learning rate: "learning rate: 3.000E-04"
            m = re.search(r"learning rate:\s*([0-9.Ee+-]+)", line)
            if m:
                lr = _safe_float(m.group(1), 0, 1)
                if lr is not None:
                    cb.learning_rates.append(lr)

            # Grad norm: "grad norm: 1.234"
            m = re.search(r"grad norm[:\s]+([0-9.Ee+-]+)", line)
            if m:
                gn = _safe_float(m.group(1), 0, 1e6)
                if gn is not None:
                    cb.grad_norms.append(gn)

            # Memory allocated: "mem-alloc-GB: 72.5"
            m = re.search(r"mem-alloc-GB[:\s]+([0-9.]+)", line)
            if m:
                mem = _safe_float(m.group(1), 0, 500)
                if mem is not None:
                    cb.memory_allocated.append(mem)

            # Memory reserved: "mem-reserved-GB: 78.0"
            m = re.search(r"mem-reserved-GB[:\s]+([0-9.]+)", line)
            if m:
                mem = _safe_float(m.group(1), 0, 500)
                if mem is not None:
                    cb.memory_reserved.append(mem)

            # Parameter count: "number of parameters on ... rank (0, 0): 8030261248"
            m = re.search(r"number of parameters.*?:\s*(\d+)", line)
            if m and cb.total_params is None:
                cb.total_params = int(m.group(1))
                cb.trainable_params = cb.total_params

            # Parallelism: "tensor_model_parallel_size: 2"
            m = re.search(r"tensor_model_parallel_size[:\s]+(\d+)", line)
            if m:
                cb.tensor_model_parallel_size = int(m.group(1))

            m = re.search(r"pipeline_model_parallel_size[:\s]+(\d+)", line)
            if m:
                cb.pipeline_model_parallel_size = int(m.group(1))

            m = re.search(r"data_parallel_size[:\s]+(\d+)", line)
            if m:
                cb.data_parallel_size = int(m.group(1))

            # Micro batch size: "micro_batch_size: 1"
            m = re.search(r"micro_batch_size[:\s]+(\d+)", line)
            if m and cb.micro_batch_size is None:
                cb.micro_batch_size = int(m.group(1))

            # Global batch size from log: "global_batch_size: 64"
            m = re.search(r"global_batch_size[:\s]+(\d+)", line)
            if m and cb.global_batch_size is None:
                cb.global_batch_size = int(m.group(1))

    # Compute total time from step times
    if cb.step_times:
        cb.total_time = sum(cb.step_times)

    # Infer gradient accumulation steps
    if cb.global_batch_size and cb.micro_batch_size and cb.data_parallel_size:
        cb.gradient_accumulation_steps = (
            cb.global_batch_size // (cb.micro_batch_size * cb.data_parallel_size)
        ) or 1

    return cb
