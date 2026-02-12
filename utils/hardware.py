"""Hardware detection and GPU/device utilities for Qwen-Bench."""
from typing import Any, Dict


# Known GPU core counts (CUDA cores for NVIDIA, stream processors for AMD)
NVIDIA_CORES = {
    "h100": 16896,
    "h100 sxm5": 16896,
    "h100 pcie": 14592,
    "a100": 6912,
    "a100-sxm4": 6912,
    "a100-pcie": 6912,
    "v100": 5120,
    "v100-sxm2": 5120,
    "v100-pcie": 5120,
    "a40": 10752,
    "a30": 10752,
    "a10": 9216,
    "rtx 4090": 16384,
    "rtx 3090": 10496,
    "rtx 3080": 8704,
}

AMD_CORES = {
    "mi300x": 19456,
    "mi300a": 19456,
    "mi250x": 14080 * 2,
    "mi250": 13312 * 2,
    "mi210": 13312,
    "mi100": 7680,
    "instinct mi300x": 19456,
    "instinct mi250x": 14080 * 2,
    "instinct mi250": 13312 * 2,
    "instinct mi210": 13312,
    "instinct mi100": 7680,
}


def get_gpu_core_count(device_name: str, device_props=None) -> int:
    """Look up known GPU core count by device name.

    Falls back to SM count * 128 for unknown NVIDIA GPUs when device_props is available.
    """
    name_lower = device_name.lower()

    for gpu_name, cores in NVIDIA_CORES.items():
        if gpu_name in name_lower:
            return cores

    for gpu_name, cores in AMD_CORES.items():
        if gpu_name in name_lower:
            return cores

    if device_props is not None and hasattr(device_props, "multi_processor_count"):
        return device_props.multi_processor_count * 128

    return 0


def detect_gpu_info() -> Dict[str, Any]:
    """Detect GPU information using PyTorch.

    Returns a dict with device_count, device_name, total_memory_gb, gpu_cores,
    pytorch_version, software_stack (cuda/rocm), and software_version.
    """
    try:
        import torch
    except ImportError:
        return {
            "device_count": 0,
            "device_name": "N/A (torch not available)",
            "total_memory_gb": 0,
            "gpu_cores": 0,
            "pytorch_version": "N/A",
            "software_stack": "N/A",
            "software_version": "N/A",
        }

    if not torch.cuda.is_available():
        return {
            "device_count": 0,
            "device_name": "N/A (no GPU)",
            "total_memory_gb": 0,
            "gpu_cores": 0,
            "pytorch_version": torch.__version__,
            "software_stack": "cpu",
            "software_version": "N/A",
        }

    device_props = torch.cuda.get_device_properties(0)
    device_name = torch.cuda.get_device_name(0)
    gpu_cores = get_gpu_core_count(device_name, device_props)
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    software_stack = "rocm" if is_rocm else "cuda"
    software_version = torch.version.hip if is_rocm else torch.version.cuda

    return {
        "device_count": torch.cuda.device_count(),
        "device_name": device_name,
        "total_memory_gb": round(device_props.total_memory / 1e9, 2),
        "gpu_cores": gpu_cores,
        "pytorch_version": torch.__version__,
        "software_stack": software_stack,
        "software_version": software_version,
    }


def detect_platform() -> str:
    """Return 'cuda', 'rocm', or 'cpu' based on available hardware."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    return "rocm" if is_rocm else "cuda"
