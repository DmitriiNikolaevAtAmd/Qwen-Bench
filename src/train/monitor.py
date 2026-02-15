"""Benchmark callback for Megatron-Core training.

Collects per-step metrics (loss, learning rate, grad norm, step time, memory)
and writes a JSON report at the end of training. The JSON is later consumed
by the evaluate stage to build the comparison dashboard.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch


def round_floats(obj: Any, precision: int = 5) -> Any:
    """Round all floats in a nested structure."""
    if isinstance(obj, float):
        if abs(obj) < 0.001 and obj != 0:
            return round(obj, 10)
        return round(obj, precision)
    elif isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, precision) for item in obj]
    return obj


class BenchmarkCallback:
    """Collects training metrics during Megatron-Core pretrain runs.

    Designed to be driven manually from the training log output rather than
    via a Trainer callback hook (Megatron-Core ``pretrain()`` does not expose
    a callback API).  Instead, the evaluate stage parses the Megatron log and
    populates this object post-hoc.

    Alternatively the worker can call ``record_step()`` inside the forward
    step when a custom training loop is used.
    """

    def __init__(
        self,
        output_dir: str = "./output",
        platform: str = "auto",
        model_name: Optional[str] = None,
        framework: str = "megatron",
        dataset: Optional[str] = None,
        warmup_steps: int = 3,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if platform == "auto":
            if torch.cuda.is_available():
                is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
                self.platform = "rocm" if is_rocm else "cuda"
            else:
                self.platform = "cpu"
        else:
            self.platform = platform

        self.warmup_steps = warmup_steps
        self.model_name = model_name
        self.framework = framework
        self.dataset = dataset or os.environ.get("DATASET", "benchmark")

        # Collected series
        self.step_times: list[float] = []
        self.loss_values: list[float] = []
        self.learning_rates: list[float] = []
        self.grad_norms: list[float] = []
        self.memory_allocated: list[float] = []
        self.memory_reserved: list[float] = []

        # Scalar metadata
        self.gpu_info: dict[str, Any] = {}
        self.num_gpus: int = 0
        self.global_batch_size: Optional[int] = None
        self.micro_batch_size: Optional[int] = None
        self.sequence_length: Optional[int] = None
        self.total_params: Optional[int] = None
        self.trainable_params: Optional[int] = None
        self.train_start_time: Optional[float] = None
        self.total_time: Optional[float] = None

        # Parallelism metadata
        self.tensor_model_parallel_size: int = 1
        self.pipeline_model_parallel_size: int = 1
        self.data_parallel_size: int = 1
        self.gradient_accumulation_steps: int = 1

    # ------------------------------------------------------------------
    # Manual recording API (for custom loops or post-hoc parsing)
    # ------------------------------------------------------------------

    def record_step(
        self,
        step_time: Optional[float] = None,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        mem_allocated_gb: Optional[float] = None,
        mem_reserved_gb: Optional[float] = None,
    ) -> None:
        """Record metrics for a single training step."""
        if step_time is not None:
            self.step_times.append(step_time)
        if loss is not None:
            self.loss_values.append(loss)
        if lr is not None:
            self.learning_rates.append(lr)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
        if mem_allocated_gb is not None:
            self.memory_allocated.append(mem_allocated_gb)
        if mem_reserved_gb is not None:
            self.memory_reserved.append(mem_reserved_gb)

    def detect_gpu(self) -> None:
        """Populate gpu_info from the current CUDA device."""
        if not torch.cuda.is_available():
            return
        self.num_gpus = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        self.platform = "rocm" if is_rocm else "cuda"
        self.gpu_info = {
            "device_count": self.num_gpus,
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gb": round(props.total_memory / 1e9, 2),
            "pytorch_version": torch.__version__,
            "software_stack": "rocm" if is_rocm else "cuda",
            "software_version": torch.version.hip if is_rocm else torch.version.cuda,
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def build_results(self, max_steps: Optional[int] = None) -> dict[str, Any]:
        """Build the benchmark results dictionary."""
        total_time = self.total_time or 0.0

        # Skip first step (JIT/compilation warmup)
        if len(self.step_times) > 1:
            step_times_no_warmup = self.step_times[1:]
        else:
            step_times_no_warmup = self.step_times

        avg_step_time = (
            sum(step_times_no_warmup) / len(step_times_no_warmup)
            if step_times_no_warmup else 0
        )
        steps_per_second = (
            len(step_times_no_warmup) / sum(step_times_no_warmup)
            if step_times_no_warmup else 0
        )

        tokens_per_second = None
        tokens_per_second_per_gpu = None
        if self.global_batch_size and self.sequence_length and avg_step_time > 0:
            tokens_per_step = self.global_batch_size * self.sequence_length
            tokens_per_second = tokens_per_step / avg_step_time
            tokens_per_second_per_gpu = (
                tokens_per_second / self.num_gpus if self.num_gpus else None
            )

        results: dict[str, Any] = {
            "platform": self.platform,
            "dataset": self.dataset,
            "gpu_info": self.gpu_info,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "total_params": self.total_params,
                "trainable_params": self.trainable_params,
            },
            "parallelism_config": {
                "tensor_model_parallel_size": self.tensor_model_parallel_size,
                "pipeline_model_parallel_size": self.pipeline_model_parallel_size,
                "data_parallel_size": self.data_parallel_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
            },
            "training_config": {
                "max_steps": max_steps or len(self.step_times),
                "global_batch_size": self.global_batch_size or "N/A",
                "micro_batch_size": self.micro_batch_size or "N/A",
                "sequence_length": self.sequence_length or "N/A",
                "num_gpus": self.num_gpus,
            },
            "performance_metrics": {
                "total_steps": len(self.step_times),
                "total_time_seconds": total_time,
                "avg_step_time_seconds": avg_step_time,
                "min_step_time_seconds": min(step_times_no_warmup) if step_times_no_warmup else 0,
                "max_step_time_seconds": max(step_times_no_warmup) if step_times_no_warmup else 0,
                "steps_per_second": steps_per_second,
                "tokens_per_second": tokens_per_second,
                "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
            },
            "step_times": self.step_times,
            "loss_values": self.loss_values,
            "learning_rates": self.learning_rates,
            "grad_norms": self.grad_norms,
        }

        if self.memory_allocated:
            results["memory_metrics"] = {
                "peak_memory_allocated_gb": max(self.memory_allocated),
                "avg_memory_allocated_gb": sum(self.memory_allocated) / len(self.memory_allocated),
                "min_memory_allocated_gb": min(self.memory_allocated),
                "measurement_method": "torch.cuda.memory_allocated",
            }
        if self.memory_reserved:
            results.setdefault("memory_metrics", {}).update({
                "peak_memory_reserved_gb": max(self.memory_reserved),
                "avg_memory_reserved_gb": sum(self.memory_reserved) / len(self.memory_reserved),
                "min_memory_reserved_gb": min(self.memory_reserved),
                "reserved_measurement_method": "torch.cuda.mem_get_info",
            })

        return results

    def save(self, max_steps: Optional[int] = None) -> Path:
        """Build results and write to a JSON file. Returns the file path.

        Filename convention: train_{platform}_{framework}_{model}_{dataset}.json
        (e.g. train_cuda_megatron_qwen_bc.json)
        """
        results = self.build_results(max_steps)
        results = round_floats(results)

        parts = ["train"]
        parts.append(self.platform or "unknown")
        if self.framework:
            parts.append(self.framework)
        if self.model_name:
            parts.append(self.model_name)
        if self.dataset:
            parts.append(self.dataset)
        filename = "_".join(parts) + ".json"

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        return filepath
