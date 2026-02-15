"""
Benchmark callback for LLaMA Factory / HuggingFace Trainer.

Records per-step timing, memory, loss, learning rate, and gradient norms,
then writes a structured JSON report at the end of training.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from util.hardware import detect_gpu_info, detect_platform, get_gpu_core_count
from util.logging import round_floats

try:
    from transformers import (
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
except ImportError:
    raise ImportError(
        "transformers is required for BenchmarkCallback. "
        "Install it with: pip install transformers"
    )


class BenchmarkCallback(TrainerCallback):
    """HuggingFace Trainer callback that benchmarks LLaMA Factory training.

    Records per-step wall-clock time, GPU memory, loss, learning rate, and
    gradient norms. Writes a JSON report to output_dir on training end.

    Args:
        output_dir: Directory for the JSON results file.
        platform: Platform label ('auto' detects cuda/rocm/cpu).
        model_name: Model identifier for the results filename.
        dataset: Dataset name for the results filename.
        warmup_steps: Number of initial steps to skip in timing measurements
            (CUDA/ROCm kernels compile during early steps).
    """

    def __init__(
        self,
        output_dir: str = "./output",
        platform: str = "auto",
        model_name: Optional[str] = None,
        dataset: Optional[str] = None,
        warmup_steps: int = 3,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.platform = detect_platform() if platform == "auto" else platform
        self.model_name = model_name
        self.dataset = dataset or os.environ.get("DATASET", "")
        self.warmup_steps = warmup_steps

        # Metrics accumulators
        self.step_times: list[float] = []
        self.memory_allocated: list[float] = []
        self.memory_reserved: list[float] = []
        self.loss_values: list[float] = []
        self.learning_rates: list[float] = []
        self.grad_norms: list[float] = []

        # Internal state
        self._step_start: Optional[float] = None
        self._train_start: Optional[float] = None
        self._measured_step_count = 0
        self.gpu_info: dict = {}
        self.num_gpus = 1

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._train_start = time.time()
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.gpu_info = detect_gpu_info()

        # Global batch size for throughput calculation
        self._global_batch_size = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.num_gpus
        )
        self._seq_length = int(os.environ.get("CUTOFF_LEN", "2048"))

        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main:
            print(f"\n{'=' * 60}")
            print(f"BENCHMARK START - Platform: {self.platform.upper()}")
            print(f"{'=' * 60}")
            for key, value in self.gpu_info.items():
                print(f"  {key}: {value}")
            print(f"{'=' * 60}\n")

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Skip warmup steps for timing (kernel compilation phase)
        if state.global_step < self.warmup_steps:
            self._step_start = None
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._step_start is None:
            is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if is_main and state.global_step < self.warmup_steps:
                print(
                    f"  [warmup] Step {state.global_step + 1}/{self.warmup_steps} (un-timed)"
                )
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_time = time.time() - self._step_start
        self.step_times.append(step_time)
        self._measured_step_count += 1

        # Extract metrics from Trainer log_history
        if state.log_history:
            latest = state.log_history[-1]
            if "loss" in latest:
                self.loss_values.append(float(latest["loss"]))
            if "learning_rate" in latest:
                self.learning_rates.append(float(latest["learning_rate"]))
            if "grad_norm" in latest:
                self.grad_norms.append(float(latest["grad_norm"]))

        # GPU memory
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1e9
            free, total = torch.cuda.mem_get_info()
            mem_used = (total - free) / 1e9
            self.memory_allocated.append(mem_alloc)
            self.memory_reserved.append(mem_used)

        # Periodic progress logging
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main and self._measured_step_count > 0 and self._measured_step_count % 10 == 0:
            recent = self.step_times[-10:]
            avg = sum(recent) / len(recent)
            loss_str = f" | Loss: {self.loss_values[-1]:.4f}" if self.loss_values else ""
            mem_str = (
                f" | Mem: {self.memory_allocated[-1]:.1f}GB"
                if self.memory_allocated
                else ""
            )
            print(
                f"  [{self.platform.upper()}] Step {state.global_step:4d} | "
                f"Time: {step_time:.3f}s | Avg: {avg:.3f}s{loss_str}{mem_str}"
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        is_main = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not is_main:
            return

        total_time = time.time() - self._train_start if self._train_start else 0

        if len(self.step_times) < 2:
            print("[BenchmarkCallback] Not enough measured steps to report.")
            return

        # Skip first measured step (extra JIT overhead after warmup)
        measured = self.step_times[1:]
        avg_step = sum(measured) / len(measured)
        steps_per_sec = len(measured) / sum(measured)

        tokens_per_sec = None
        tokens_per_sec_gpu = None
        if self._global_batch_size and self._seq_length:
            tokens_per_step = self._global_batch_size * self._seq_length
            tokens_per_sec = tokens_per_step / avg_step
            tokens_per_sec_gpu = tokens_per_sec / self.num_gpus if self.num_gpus else None

        results = {
            "platform": self.platform,
            "dataset": self.dataset,
            "gpu_info": self.gpu_info,
            "timestamp": datetime.now().isoformat(),
            "training_config": {
                "max_steps": args.max_steps,
                "num_train_epochs": args.num_train_epochs,
                "global_batch_size": self._global_batch_size,
                "micro_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "sequence_length": self._seq_length,
                "num_gpus": self.num_gpus,
                "learning_rate": args.learning_rate,
                "bf16": args.bf16,
                "fp16": args.fp16,
                "deepspeed": args.deepspeed or None,
            },
            "performance_metrics": {
                "total_steps": len(self.step_times),
                "total_time_seconds": round(total_time, 2),
                "avg_step_time_seconds": round(avg_step, 5),
                "min_step_time_seconds": round(min(measured), 5),
                "max_step_time_seconds": round(max(measured), 5),
                "steps_per_second": round(steps_per_sec, 5),
                "tokens_per_second": round(tokens_per_sec, 2) if tokens_per_sec else None,
                "tokens_per_second_per_gpu": (
                    round(tokens_per_sec_gpu, 2) if tokens_per_sec_gpu else None
                ),
            },
            "step_times": self.step_times,
            "loss_values": self.loss_values,
            "learning_rates": self.learning_rates,
            "grad_norms": self.grad_norms,
        }

        # Memory metrics
        if self.memory_allocated:
            results["memory_metrics"] = {
                "peak_memory_allocated_gb": round(max(self.memory_allocated), 2),
                "avg_memory_allocated_gb": round(
                    sum(self.memory_allocated) / len(self.memory_allocated), 2
                ),
                "measurement_method": "torch.cuda.memory_allocated",
            }
        if self.memory_reserved:
            results.setdefault("memory_metrics", {}).update(
                {
                    "peak_memory_reserved_gb": round(max(self.memory_reserved), 2),
                    "avg_memory_reserved_gb": round(
                        sum(self.memory_reserved) / len(self.memory_reserved), 2
                    ),
                    "reserved_measurement_method": "torch.cuda.mem_get_info",
                }
            )

        # Write results JSON
        suffix = f"_{self.dataset}" if self.dataset else ""
        model_tag = self.model_name or "qwen"
        filename = f"bench_{self.platform}_{model_tag}{suffix}.json"
        filepath = self.output_dir / filename

        results_rounded = round_floats(results, precision=5)
        with open(filepath, "w") as f:
            json.dump(results_rounded, f, indent=2)

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK COMPLETE - {self.platform.upper()}")
        print(f"{'=' * 60}")
        print(f"  GPUs: {self.num_gpus}")
        print(f"  Steps: {len(self.step_times)} (measured)")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Avg step: {avg_step:.3f}s")

        if tokens_per_sec:
            print(f"  Throughput: {tokens_per_sec:,.0f} tok/s total, "
                  f"{tokens_per_sec_gpu:,.0f} tok/s/GPU")

        mem = results.get("memory_metrics", {})
        if mem:
            print(f"  Memory: {mem.get('peak_memory_allocated_gb', 'N/A')} GB peak, "
                  f"{mem.get('avg_memory_allocated_gb', 'N/A')} GB avg")

        print(f"  Results: {filepath}")
        print(f"{'=' * 60}\n")
