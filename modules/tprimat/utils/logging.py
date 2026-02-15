"""Log parsing, extraction, and representation (summary/formatting)."""
import os
import re
from typing import Dict, List, Optional, Any, Tuple


def round_floats(obj: Any, precision: int = 5) -> Any:
    if isinstance(obj, float):
        if abs(obj) < 0.001 and obj != 0:
            return round(obj, 10)
        return round(obj, precision)
    elif isinstance(obj, dict):
        return {key: round_floats(value, precision) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, precision) for item in obj]
    else:
        return obj


def extract_param_count_from_log(log_file: str) -> Optional[int]:
    """Extract model parameter count from Megatron/Primus training log.

    Megatron logs lines like:
        'number of parameters on (tensor, pipeline) model parallel rank (0, 0): 8030261248'
        'number of parameters: 8030261248'
    """
    patterns = [
        r'number of parameters on.*?rank\s*\(0,\s*0\)[:\s]+(\d+)',
        r'number of parameters[:\s]+(\d+)',
        r'total parameters[:\s]+([0-9,]+)',
        r'\[DIAG\]\s*Total parameters[:\s]+([0-9,]+)',
    ]
    with open(log_file, 'r') as f:
        for line in f:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        return int(match.group(1).replace(',', ''))
                    except (ValueError, IndexError):
                        continue
    return None


def extract_step_times_from_log(log_file: str) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    step_times = []
    tokens_per_gpu_values = []
    loss_values = []
    learning_rates = []
    grad_norms = []

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'elapsed time per iteration \(ms\):\s*([0-9.]+)/([0-9.]+)', line)
            if match:
                try:
                    step_time_ms = float(match.group(1))
                    step_time_s = step_time_ms / 1000.0
                    if 0.001 < step_time_s < 1000:
                        step_times.append(step_time_s)
                except (ValueError, IndexError):
                    continue

            tokens_match = re.search(r'tokens per GPU \(tokens/s/GPU\):\s*([0-9.]+)/([0-9.]+)', line)
            if tokens_match:
                try:
                    tokens_per_gpu = float(tokens_match.group(1))
                    if 0 < tokens_per_gpu < 1000000:
                        tokens_per_gpu_values.append(tokens_per_gpu)
                except (ValueError, IndexError):
                    continue

            loss_match = re.search(r'lm loss:\s*([0-9.Ee+-]+)', line)
            if loss_match:
                try:
                    loss = float(loss_match.group(1))
                    if 0 < loss < 10000:
                        loss_values.append(loss)
                except (ValueError, IndexError):
                    continue

            lr_match = re.search(r'learning rate:\s*([0-9.Ee+-]+)', line)
            if lr_match:
                try:
                    lr = float(lr_match.group(1))
                    if 0 < lr < 1:
                        learning_rates.append(lr)
                except (ValueError, IndexError):
                    continue

            # Megatron/Primus: "grad norm: 1.234" or "| grad norm: 1.234 |"
            gn_match = re.search(r'grad norm[:\s]+([0-9.Ee+-]+)', line)
            if gn_match:
                try:
                    gn = float(gn_match.group(1))
                    if 0 <= gn < 1e6:
                        grad_norms.append(gn)
                except (ValueError, IndexError):
                    continue

    return step_times, tokens_per_gpu_values, loss_values, learning_rates, grad_norms


def _extract_memory_values(log_file: str, patterns: List[tuple]) -> List[float]:
    """Extract per-iteration memory values (in GB) from training log using given patterns.

    Args:
        log_file: Path to training log file.
        patterns: List of (regex, unit) tuples. Unit is 'gb', 'gib', or 'mb'.

    Returns:
        List of memory values in decimal GB.
    """
    memory_values = []
    with open(log_file, 'r') as f:
        for line in f:
            for pattern, unit in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        raw_val = float(match.group(1))
                        if unit == 'gib':
                            memory_gb = raw_val * 1.073741824  # GiB -> GB
                        elif unit == 'mb':
                            memory_gb = raw_val / 1000.0
                        else:
                            memory_gb = raw_val
                        if 0 < memory_gb < 500:  # Reasonable GPU memory range
                            memory_values.append(round(memory_gb, 2))
                            break  # Found a match, move to next line
                    except (ValueError, IndexError):
                        pass
    return memory_values


def extract_memory_from_log(log_file: str) -> List[float]:
    """Extract per-iteration PyTorch tensor allocation memory (in GB) from training log.

    This captures torch.cuda.memory_allocated() — actual memory held by tensors.

    Supports multiple formats:
    - Megatron: "mem-alloc-GB: 72.5", "memory (GB) | allocated: 72.5"
    - Primus debug: "memory (MB) | allocated: 54638.4"
    - Generic: "allocated: 72.5 GB"

    Note: "hip mem usage" is NOT included here — it reports total VRAM (system level),
    not PyTorch tensor allocations.  Use extract_system_memory_from_log() for that.
    """
    # Patterns for PyTorch allocated memory only (order matters - more specific first)
    patterns = [
        # Megatron format: "mem-alloc-GB: 72.5"
        (r'mem-alloc-GB[:\s]+([0-9.]+)', 'gb'),
        # Megatron format: "memory (GB) | allocated: 72.5"
        (r'memory\s*\(GB\)\s*\|\s*allocated[:\s]+([0-9.]+)', 'gb'),
        # Megatron format: "gpu_memory_allocated: 72.5"
        (r'gpu_memory_allocated[:\s]+([0-9.]+)', 'gb'),
        # HIP/ROCm format: "hip mem allocated: 72.5 GB" (NOT hip mem usage)
        (r'hip mem allocated[^:]*:\s*([0-9.]+)\s*(?:GB|GiB)', 'gb'),
        # Primus debug: "memory (MB) | allocated: 54638.4"
        (r'memory\s*\(MB\)\s*\|\s*allocated[:\s]+([0-9.]+)', 'mb'),
        # Generic: "allocated: 72.5 GB" or "max allocated: 72.5 GB"
        (r'(?:max\s+)?allocated[:\s]+([0-9.]+)\s*GB', 'gb'),
        # Megatron throughput line with memory: "| mem-alloc-GB: 72.5 |"
        (r'\|\s*mem-alloc-GB[:\s]+([0-9.]+)\s*\|', 'gb'),
    ]
    return _extract_memory_values(log_file, patterns)


def extract_system_memory_from_log(log_file: str) -> List[float]:
    """Extract per-iteration system VRAM usage (in GB) from training log.

    This captures total GPU memory in use — equivalent to nvidia-smi/rocm-smi,
    sampled per iteration from within the training process.

    Sources (in priority order):
    - Megatron: "mem-reserved-GB: 78.0" (torch.cuda.memory_reserved / caching allocator)
    - Primus/ROCm: "hip mem usage/free/total/usage_ratio: 72.50GiB/..." (total VRAM)
    - Megatron: "memory (GB) | reserved: 78.0"
    - Generic: "memory: 72.5 GB", "memory usage: 72.5 GB"
    """
    # Patterns for system / total VRAM usage (order matters - more specific first)
    patterns = [
        # Megatron format: "mem-reserved-GB: 78.0" (caching allocator reserved ≈ system VRAM)
        (r'mem-reserved-GB[:\s]+([0-9.]+)', 'gb'),
        # Megatron throughput line: "| mem-reserved-GB: 78.0 |"
        (r'\|\s*mem-reserved-GB[:\s]+([0-9.]+)\s*\|', 'gb'),
        # Megatron format: "memory (GB) | reserved: 78.0"
        (r'memory\s*\(GB\)\s*\|\s*reserved[:\s]+([0-9.]+)', 'gb'),
        # Primus/ROCm: "hip mem usage/free/total/usage_ratio: 72.50GiB/..."
        (r'hip mem usage[^:]*:\s*([0-9.]+)\s*GiB', 'gib'),
        # HIP/ROCm format: "hip mem usage: 72.5 GB"
        (r'hip mem usage[^:]*:\s*([0-9.]+)\s*GB', 'gb'),
        # Generic: "memory: 72.5 GB" or "memory usage: 72.5 GB"
        (r'memory\s*(?:usage)?[:\s]+([0-9.]+)\s*GB', 'gb'),
    ]
    return _extract_memory_values(log_file, patterns)


def parse_memory_log(log_file: str, num_steps: int = None) -> Optional[Dict[str, Any]]:
    """Parse rocm-smi or nvidia-smi memory log to extract memory values.

    Converts all values to decimal GB (1 GB = 1e9 bytes) for consistency
    with torch.cuda.memory_allocated() / 1e9 used elsewhere.

    Args:
        log_file: Path to memory log file
        num_steps: If provided, interpolate memory values to match step count

    Returns:
        Dict with memory_values array and summary stats, or None if no data found.
    """
    if not os.path.exists(log_file):
        return None

    raw_values = []

    with open(log_file, 'r') as f:
        for line in f:
            # rocm-smi format: "VRAM Total Used Memory (B): 73014444032"
            match = re.search(r'Used.*\(B\)[:\s]+(\d+)', line)
            if match:
                bytes_val = int(match.group(1))
                raw_values.append(bytes_val / 1e9)  # bytes -> GB
                continue

            # rocm-smi format: "Used: 69632 MB" (GPU tools report MiB)
            match = re.search(r'Used[:\s]+(\d+)\s*MB', line, re.IGNORECASE)
            if match:
                mib_val = int(match.group(1))
                raw_values.append(mib_val * 1048576 / 1e9)  # MiB -> bytes -> GB
                continue

            # nvidia-smi CSV format: "0, 65432" (index, memory_mib)
            match = re.search(r'^\d+,\s*(\d+)', line)
            if match:
                mib_val = int(match.group(1))
                raw_values.append(mib_val * 1048576 / 1e9)  # MiB -> bytes -> GB
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


def get_parallelism_config(strategy: str, model: str, platform: str) -> Dict[str, Any]:
    return {
        "strategy": strategy or "unknown",
        "tensor_parallel_size": int(os.environ.get("TP", 1)),
        "pipeline_parallel_size": int(os.environ.get("PP", 1)),
        "data_parallel_size": int(os.environ.get("DP", 4)),
        "gradient_accumulation_steps": int(os.environ.get("GA", 32)),
    }


def print_summary(results: Dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY - Platform: {results['platform'].upper()}")
    print(f"{'='*60}")
    print(f"Device: {results['gpu_info'].get('device_name', 'Unknown')}")
    print(f"GPUs: {results['training_config']['num_gpus']}")
    print(f"Total Steps: {results['performance_metrics']['total_steps']}")
    print(f"Avg Step Time: {results['performance_metrics']['avg_step_time_seconds']:.3f}s")

    if results['performance_metrics'].get('tokens_per_second'):
        print(f"\nThroughput Metrics:")
        print(f"  Total Throughput: {results['performance_metrics']['tokens_per_second']:,.0f} tokens/sec")
        print(f"  Per-GPU Throughput: {results['performance_metrics']['tokens_per_second_per_gpu']:,.0f} tokens/sec/GPU")

    print(f"{'='*60}\n")
