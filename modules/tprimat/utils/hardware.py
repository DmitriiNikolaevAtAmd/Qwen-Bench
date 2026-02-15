"""Hardware detection and GPU/device utilities."""
from typing import Dict, Any
import torch


def get_gpu_core_count(device_name: str, device_props) -> int:
    device_name_lower = device_name.lower()

    nvidia_cores = {
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

    amd_cores = {
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

    for gpu_name, cores in nvidia_cores.items():
        if gpu_name in device_name_lower:
            return cores

    for gpu_name, cores in amd_cores.items():
        if gpu_name in device_name_lower:
            return cores

    if hasattr(device_props, 'multi_processor_count'):
        return device_props.multi_processor_count * 128

    return 0


def detect_gpu_info() -> Dict[str, Any]:
    gpu_info = {}

    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False

    if not TORCH_AVAILABLE:
        gpu_info = {
            "device_count": "N/A",
            "device_name": "AMD GPU (from log)",
            "total_memory_gb": 192,
            "gpu_cores": 19456,
            "pytorch_version": "N/A",
            "software_stack": "rocm",
            "software_version": "N/A",
        }
        return gpu_info

    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        device_name = torch.cuda.get_device_name(0)
        gpu_cores = get_gpu_core_count(device_name, device_props)
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        software_stack = "rocm" if is_rocm else "cuda"
        software_version = torch.version.hip if is_rocm else torch.version.cuda

        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "device_name": device_name,
            "total_memory_gb": device_props.total_memory / 1e9,
            "gpu_cores": gpu_cores,
            "pytorch_version": torch.__version__,
            "software_stack": software_stack,
            "software_version": software_version,
        }
    else:
        gpu_info = {
            "device_count": "N/A",
            "device_name": "Unknown (from logs)",
            "total_memory_gb": "N/A",
            "gpu_cores": 0,
            "pytorch_version": torch.__version__ if TORCH_AVAILABLE else "N/A",
            "software_stack": "rocm",
            "software_version": "N/A",
        }

    return gpu_info
