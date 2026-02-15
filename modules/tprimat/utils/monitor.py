"""Benchmark callbacks and callback integration (Lightning, Hugging Face Trainer)."""
import time
import json
import os
from datetime import datetime
from pathlib import Path
import torch
from lightning.pytorch.callbacks import Callback

from utils.hardware import get_gpu_core_count
from utils.logging import round_floats

try:
    from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    TrainerCallback = object


class BenchmarkCallback(Callback):
    def __init__(self, output_dir: str = "./output", platform: str = "auto", model_name: str = None,
                 parallel_strategy: str = "unknown", framework: str = None, dataset: str = None,
                 warmup_steps: int = 3):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if platform == "auto":
            if torch.cuda.is_available():
                is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
                self.platform = "amd" if is_rocm else "nvd"
            else:
                self.platform = "cpu"
        else:
            self.platform = platform

        self.warmup_steps = warmup_steps
        self._batch_count = 0
        self.step_times = []
        self.memory_allocated = []
        self.memory_reserved = []
        self.memory_allocated_per_gpu = []
        self.loss_values = []
        self.learning_rates = []
        self.grad_norms = []
        self.step_start_time = None
        self.train_start_time = None
        self.gpu_info = {}
        self.global_batch_size = None
        self.sequence_length = None
        self.num_gpus = None
        self.model_name = model_name
        self.parallel_strategy = parallel_strategy
        self.framework = framework
        self.dataset = dataset or os.environ.get("DATASET", "bc")

    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

        if hasattr(trainer, 'datamodule'):
            self.global_batch_size = getattr(trainer.datamodule, 'global_batch_size', None)
            self.sequence_length = getattr(trainer.datamodule, 'seq_length',
                                          getattr(trainer.datamodule, 'sequence_length', 2048))

        # Count model parameters
        self.total_params = sum(p.numel() for p in pl_module.parameters())
        self.trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            device_name = torch.cuda.get_device_name(0)
            gpu_cores = get_gpu_core_count(device_name, device_props)
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            software_stack = "prim" if is_rocm else "nemo"
            software_version = torch.version.hip if is_rocm else torch.version.cuda

            self.gpu_info = {
                "device_count": self.num_gpus,
                "device_name": device_name,
                "total_memory_gb": device_props.total_memory / 1e9,
                "gpu_cores": gpu_cores,
                "pytorch_version": torch.__version__,
                "software_stack": software_stack,
                "software_version": software_version,
            }
        if trainer.is_global_zero:
            software_stack = self.gpu_info.get("software_stack", "unknown")
            print(f"\n{'='*60}")
            print(f"BENCHMARK START - Platform: {self.platform.upper()} ({software_stack.upper()})")
            print(f"{'='*60}")
            for key, value in self.gpu_info.items():
                print(f"{key}: {value}")
            print(f"[DIAG] Total parameters: {self.total_params:,}")
            print(f"[DIAG] Trainable parameters: {self.trainable_params:,}")
            print(f"{'='*60}\n")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_count += 1

        # Log token IDs from the first 3 micro-batches for data pipeline verification
        if self._batch_count <= 3 and trainer.is_global_zero and batch is not None:
            try:
                tokens = None
                if isinstance(batch, dict):
                    tokens = batch.get('tokens', batch.get('input_ids', batch.get('text', None)))
                elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                    tokens = batch[0]
                elif torch.is_tensor(batch):
                    tokens = batch

                if tokens is not None and torch.is_tensor(tokens):
                    first_seq = tokens[0] if tokens.dim() > 1 else tokens
                    ids = first_seq[:32].tolist()
                    print(f"[DIAG] Batch {self._batch_count} tokens (first 32): {ids}")
                    print(f"[DIAG] Batch {self._batch_count} shape: {list(tokens.shape)}, "
                          f"dtype: {tokens.dtype}, min: {tokens.min().item()}, max: {tokens.max().item()}")
                else:
                    print(f"[DIAG] Batch {self._batch_count}: type={type(batch).__name__}, "
                          f"could not extract token IDs")
            except Exception as e:
                print(f"[DIAG] Batch {self._batch_count}: error extracting tokens: {e}")

        # During warmup steps, skip timing (CUDA kernels are still compiling)
        if self._batch_count <= self.warmup_steps:
            self.step_start_time = None
            self._step_lr = None
            return
        self.step_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Save LR at the START of the step (before scheduler advances)
        # so the recorded value matches the LR actually used for this update.
        try:
            self._step_lr = trainer.optimizers[0].param_groups[0]['lr']
        except (IndexError, KeyError, AttributeError):
            self._step_lr = None

    def _extract_loss(self, trainer, pl_module, outputs) -> float:
        """Extract training loss using all available sources.

        Priority (highest to lowest):
          1. trainer.callback_metrics['reduced_train_loss'] — Megatron's true reduced lm loss
          2. trainer.callback_metrics['train_loss']
          3. pl_module.loss_mean (NeMo internal)
          4. outputs.loss / outputs['loss'] (Lightning default — may be scaled/transformed)

        Returns the loss as a Python float, or None if nothing found.
        """
        loss = None
        source = None

        # --- Source 1: callback_metrics (most reliable for NeMo/Megatron) ---
        cb = getattr(trainer, 'callback_metrics', {}) or {}
        for key in ('reduced_train_loss', 'train_loss', 'loss'):
            val = cb.get(key)
            if val is not None:
                loss = val.item() if torch.is_tensor(val) else float(val)
                source = f'callback_metrics[{key!r}]'
                break

        # --- Source 2: pl_module.loss_mean (NeMo stores average loss here) ---
        if loss is None:
            for attr in ('loss_mean', '_loss_mean'):
                val = getattr(pl_module, attr, None)
                if val is not None:
                    loss = val.item() if torch.is_tensor(val) else float(val)
                    source = f'pl_module.{attr}'
                    break

        # --- Source 3: outputs (Lightning default) ---
        if loss is None and outputs is not None:
            if isinstance(outputs, dict):
                val = outputs.get('loss')
            elif hasattr(outputs, 'loss'):
                val = outputs.loss
            elif torch.is_tensor(outputs):
                val = outputs
            else:
                val = None
            if val is not None:
                loss = val.item() if torch.is_tensor(val) else float(val)
                source = 'outputs.loss'

        # --- Diagnostic: log all sources for the first 3 measured steps ---
        measured_step = len(self.step_times) + 1  # about to append
        if trainer.is_global_zero and measured_step <= 3:
            print(f"[LOSS-DIAG] Step {measured_step} | chosen source: {source} = {loss}")
            # Dump every candidate so we can compare
            diag_parts = []
            for key in ('reduced_train_loss', 'train_loss', 'loss'):
                val = cb.get(key)
                if val is not None:
                    diag_parts.append(f"cb[{key!r}]={val.item() if torch.is_tensor(val) else val:.6f}")
            for attr in ('loss_mean', '_loss_mean'):
                val = getattr(pl_module, attr, None)
                if val is not None:
                    diag_parts.append(f"pl.{attr}={val.item() if torch.is_tensor(val) else val:.6f}")
            if outputs is not None:
                out_type = type(outputs).__name__
                if isinstance(outputs, dict):
                    out_keys = list(outputs.keys())
                    diag_parts.append(f"outputs=dict(keys={out_keys})")
                    for k, v in outputs.items():
                        vv = v.item() if torch.is_tensor(v) else v
                        diag_parts.append(f"  outputs[{k!r}]={vv}")
                elif torch.is_tensor(outputs):
                    diag_parts.append(f"outputs=tensor({outputs.item():.6f})")
                elif hasattr(outputs, '__dict__'):
                    diag_parts.append(f"outputs={out_type}(attrs={list(outputs.__dict__.keys())})")
                    for k, v in outputs.__dict__.items():
                        vv = v.item() if torch.is_tensor(v) else v
                        diag_parts.append(f"  outputs.{k}={vv}")
                else:
                    diag_parts.append(f"outputs={out_type}({outputs})")
            # Logged metrics
            lm = getattr(trainer, 'logged_metrics', {}) or {}
            if lm:
                for k, v in lm.items():
                    vv = v.item() if torch.is_tensor(v) else v
                    diag_parts.append(f"logged[{k!r}]={vv}")
            print(f"[LOSS-DIAG]   all: {' | '.join(diag_parts)}")

        return loss

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Skip recording during warmup steps
        if self.step_start_time is None:
            if trainer.is_global_zero and self._batch_count <= self.warmup_steps:
                print(f"[{self.platform.upper()}] Warmup step {self._batch_count}/{self.warmup_steps} (un-timed)")
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)

        loss = self._extract_loss(trainer, pl_module, outputs)
        if loss is not None:
            self.loss_values.append(float(loss))

        if torch.cuda.is_available():
            # PyTorch: memory currently held by tensors (torch.cuda.memory_allocated)
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            # System: total VRAM in use (matches nvidia-smi/rocm-smi), sampled per iteration
            free, total = torch.cuda.mem_get_info()
            mem_reserved = (total - free) / 1e9
            self.memory_allocated.append(mem_allocated)
            self.memory_reserved.append(mem_reserved)

            if torch.distributed.is_initialized():
                try:
                    curr_mem = torch.tensor([mem_allocated], device=f"cuda:{torch.cuda.current_device()}")
                    world_size = torch.distributed.get_world_size()
                    all_mems = [torch.zeros(1, device=f"cuda:{torch.cuda.current_device()}") for _ in range(world_size)]
                    torch.distributed.all_gather(all_mems, curr_mem)
                    per_gpu_mems = [m.item() for m in all_mems]
                    self.memory_allocated_per_gpu.append(per_gpu_mems)
                except Exception:
                    self.memory_allocated_per_gpu.append([mem_allocated])
            else:
                self.memory_allocated_per_gpu.append([mem_allocated])

        # Use LR saved at batch start (before scheduler advanced)
        if self._step_lr is not None:
            self.learning_rates.append(float(self._step_lr))

        # Gradient norm — try callback_metrics first (NeMo/Megatron), then logged_metrics
        gn = None
        cb = getattr(trainer, 'callback_metrics', {}) or {}
        for gn_key in ('grad_norm', 'gradient_norm', 'global_grad_norm'):
            val = cb.get(gn_key)
            if val is not None:
                gn = val.item() if torch.is_tensor(val) else float(val)
                break
        if gn is None:
            lm = getattr(trainer, 'logged_metrics', {}) or {}
            for gn_key in ('grad_norm', 'gradient_norm', 'global_grad_norm'):
                val = lm.get(gn_key)
                if val is not None:
                    gn = val.item() if torch.is_tensor(val) else float(val)
                    break
        if gn is not None:
            self.grad_norms.append(gn)

        measured_step = len(self.step_times)
        if trainer.is_global_zero and measured_step > 0 and measured_step % 10 == 0:
            recent_times = self.step_times[-10:]
            avg_time = sum(recent_times) / len(recent_times)

            loss_str = ""
            if self.loss_values:
                recent_loss = self.loss_values[-1] if self.loss_values else 0
                loss_str = f" | Loss: {recent_loss:.4f}"

            if torch.cuda.is_available():
                avg_mem = sum(self.memory_allocated[-10:]) / len(self.memory_allocated[-10:])
                print(f"[{self.platform.upper()}] Step {measured_step:3d} | "
                      f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s | "
                      f"Memory: {avg_mem:.2f}GB{loss_str}")
            else:
                print(f"[{self.platform.upper()}] Step {measured_step:3d} | "
                      f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s{loss_str}")

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        total_time = time.time() - self.train_start_time

        if len(self.step_times) > 1:
            # Skip first step (JIT/compilation warmup), matching extract.py
            step_times_no_warmup = self.step_times[1:]
            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)

            tokens_per_second = None
            tokens_per_second_per_gpu = None

            if self.global_batch_size and self.sequence_length:
                tokens_per_step = self.global_batch_size * self.sequence_length
                tokens_per_second = tokens_per_step / avg_step_time
                tokens_per_second_per_gpu = tokens_per_second / self.num_gpus if self.num_gpus else None

            parallelism_info = {}
            parallel_strategy = os.environ.get('PARALLEL', 'unknown')
            parallelism_info["strategy_name"] = parallel_strategy

            try:
                if hasattr(trainer, 'strategy') and hasattr(trainer.strategy, 'tensor_model_parallel_size'):
                    parallelism_info.update({
                        "tensor_model_parallel_size": trainer.strategy.tensor_model_parallel_size,
                        "pipeline_model_parallel_size": trainer.strategy.pipeline_model_parallel_size,
                        "data_parallel_size": self.num_gpus // (
                            trainer.strategy.tensor_model_parallel_size *
                            trainer.strategy.pipeline_model_parallel_size
                        ),
                    })
                    if self.global_batch_size and hasattr(trainer.datamodule, 'micro_batch_size'):
                        parallelism_info["gradient_accumulation_steps"] = self.global_batch_size // (
                            trainer.datamodule.micro_batch_size * parallelism_info["data_parallel_size"]
                        )
            except Exception:
                pass

            results = {
                "platform": self.platform,
                "dataset": self.dataset,
                "gpu_info": self.gpu_info,
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "total_params": self.total_params,
                    "trainable_params": self.trainable_params,
                },
                "parallelism_config": parallelism_info,
                "training_config": {
                    "max_steps": trainer.max_steps,
                    "global_batch_size": self.global_batch_size or 'N/A',
                    "micro_batch_size": getattr(trainer.datamodule, 'micro_batch_size', 'N/A'),
                    "sequence_length": self.sequence_length or 'N/A',
                    "num_gpus": self.num_gpus,
                    "parallel_strategy": self.parallel_strategy,
                },
                "performance_metrics": {
                    "total_steps": len(self.step_times),
                    "total_time_seconds": total_time,
                    "avg_step_time_seconds": avg_step_time,
                    "min_step_time_seconds": min(step_times_no_warmup),
                    "max_step_time_seconds": max(step_times_no_warmup),
                    "steps_per_second": steps_per_second,
                    "tokens_per_second": tokens_per_second,
                    "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                    "throughput_per_gpu_core": steps_per_second / self.gpu_info["gpu_cores"] if self.gpu_info.get("gpu_cores", 0) > 0 else 0,
                },
                "step_times": self.step_times,
                "loss_values": self.loss_values if self.loss_values else [],
                "learning_rates": self.learning_rates if self.learning_rates else [],
                "grad_norms": self.grad_norms if self.grad_norms else [],
            }

            # PyTorch: tensor allocations via torch.cuda.memory_allocated()
            if self.memory_allocated:
                results["memory_metrics"] = {
                    "peak_memory_allocated_gb": max(self.memory_allocated),
                    "avg_memory_allocated_gb": sum(self.memory_allocated) / len(self.memory_allocated),
                    "min_memory_allocated_gb": min(self.memory_allocated),
                    "measurement_method": "torch.cuda.memory_allocated",
                }
            # System: total VRAM via mem_get_info(), sampled per iteration
            if self.memory_reserved:
                results.setdefault("memory_metrics", {}).update({
                    "peak_memory_reserved_gb": max(self.memory_reserved),
                    "avg_memory_reserved_gb": sum(self.memory_reserved) / len(self.memory_reserved),
                    "min_memory_reserved_gb": min(self.memory_reserved),
                    "reserved_measurement_method": "torch.cuda.mem_get_info",
                })

            dataset_suffix = f"_{self.dataset}" if self.dataset else ""
            if self.framework and self.model_name:
                filename = f"train_{self.framework}_{self.model_name}{dataset_suffix}.json"
            elif self.model_name:
                software_stack = self.gpu_info.get("software_stack", self.platform)
                filename = f"train_{software_stack}_{self.model_name}{dataset_suffix}.json"
            else:
                software_stack = self.gpu_info.get("software_stack", self.platform)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"train_{software_stack}_{timestamp}{dataset_suffix}.json"

            filepath = self.output_dir / filename
            results_rounded = round_floats(results, precision=5)

            with open(filepath, 'w') as f:
                json.dump(results_rounded, f, indent=2)

            software_stack = self.gpu_info.get("software_stack", self.platform)
            print(f"\n{'='*60}")
            print(f"BENCHMARK COMPLETE - Platform: {self.platform.upper()} ({software_stack.upper()})")
            print(f"{'='*60}")
            print(f"GPUs: {self.num_gpus}")
            print(f"Total Steps: {results['performance_metrics']['total_steps']}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Avg Step Time: {results['performance_metrics']['avg_step_time_seconds']:.3f}s")

            if results['performance_metrics']['tokens_per_second']:
                print(f"\nThroughput Metrics:")
                print(f"  Total Throughput: {results['performance_metrics']['tokens_per_second']:,.0f} tokens/sec")
                print(f"  Per-GPU Throughput: {results['performance_metrics']['tokens_per_second_per_gpu']:,.0f} tokens/sec/GPU")
                print(f"  (Global batch size: {self.global_batch_size}, Sequence length: {self.sequence_length})")
            else:
                print(f"Throughput: {results['performance_metrics']['steps_per_second']:.3f} steps/s")

            mem_metrics = results.get('memory_metrics', {})
            if mem_metrics:
                alloc = mem_metrics.get('avg_memory_allocated_gb')
                resv = mem_metrics.get('avg_memory_reserved_gb')
                print(f"\nMemory (avg per GPU):")
                if alloc:
                    print(f"  PyTorch: {alloc:.2f} GB")
                if resv:
                    print(f"  System (VRAM used): {resv:.2f} GB")

            print(f"\nResults saved to: {filepath}")
            print(f"{'='*60}\n")


class BenchmarkCallbackTran(TrainerCallback):
    def __init__(self, output_dir: str = "./output", platform: str = "auto", model_name: str = None,
                 parallel_strategy: str = "unknown", framework: str = None, dataset: str = None):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if platform == "auto":
            if torch.cuda.is_available():
                is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
                self.platform = "amd" if is_rocm else "nvd"
            else:
                self.platform = "cpu"
        else:
            self.platform = platform

        self.step_times = []
        self.memory_allocated = []
        self.memory_reserved = []
        self.loss_values = []
        self.learning_rates = []
        self.grad_norms = []
        self.step_start_time = None
        self.train_start_time = None
        self.gpu_info = {}
        self.global_batch_size = None
        self.sequence_length = None
        self.num_gpus = None
        self.model_name = model_name
        self.parallel_strategy = parallel_strategy
        self.framework = framework
        self.dataset = dataset or os.environ.get("DATASET", "bc")

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self.train_start_time = time.time()
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.global_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * self.num_gpus
        self.sequence_length = 2048

        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            device_name = torch.cuda.get_device_name(0)
            gpu_cores = get_gpu_core_count(device_name, device_props)
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            software_stack = "prim" if is_rocm else "nemo"
            software_version = torch.version.hip if is_rocm else torch.version.cuda

            self.gpu_info = {
                "device_count": self.num_gpus,
                "device_name": device_name,
                "total_memory_gb": device_props.total_memory / 1e9,
                "gpu_cores": gpu_cores,
                "pytorch_version": torch.__version__,
                "software_stack": software_stack,
                "software_version": software_version,
            }

        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main_process:
            software_stack = self.gpu_info.get("software_stack", "unknown")
            print(f"\n{'='*60}")
            print(f"BENCHMARK START - Platform: {self.platform.upper()} ({software_stack.upper()})")
            print(f"{'='*60}")
            for key, value in self.gpu_info.items():
                print(f"{key}: {value}")
            print(f"{'='*60}\n")

    def on_step_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self.step_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)

        if len(state.log_history) > 0:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                self.loss_values.append(float(latest_log['loss']))
            if 'learning_rate' in latest_log:
                self.learning_rates.append(float(latest_log['learning_rate']))
            # HF Trainer logs grad_norm when max_grad_norm > 0
            if 'grad_norm' in latest_log:
                self.grad_norms.append(float(latest_log['grad_norm']))

        if torch.cuda.is_available():
            # PyTorch: memory currently held by tensors
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            # System: total VRAM in use (matches nvidia-smi/rocm-smi), sampled per iteration
            free, total = torch.cuda.mem_get_info()
            mem_reserved = (total - free) / 1e9
            self.memory_allocated.append(mem_allocated)
            self.memory_reserved.append(mem_reserved)

        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_main_process and state.global_step > 0 and state.global_step % 10 == 0:
            recent_times = self.step_times[-10:]
            avg_time = sum(recent_times) / len(recent_times)

            loss_str = ""
            if self.loss_values:
                recent_loss = self.loss_values[-1]
                loss_str = f" | Loss: {recent_loss:.4f}"

            if torch.cuda.is_available():
                avg_mem = sum(self.memory_allocated[-10:]) / len(self.memory_allocated[-10:])
                print(f"[{self.platform.upper()}] Step {state.global_step:3d} | "
                      f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s | "
                      f"Memory: {avg_mem:.2f}GB{loss_str}")

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not is_main_process:
            return

        total_time = time.time() - self.train_start_time

        if len(self.step_times) > 1:
            step_times_no_warmup = self.step_times[1:]
            avg_step_time = sum(step_times_no_warmup) / len(step_times_no_warmup)
            steps_per_second = len(step_times_no_warmup) / sum(step_times_no_warmup)

            tokens_per_second = None
            tokens_per_second_per_gpu = None

            if self.global_batch_size and self.sequence_length:
                tokens_per_step = self.global_batch_size * self.sequence_length
                tokens_per_second = tokens_per_step / avg_step_time
                tokens_per_second_per_gpu = tokens_per_second / self.num_gpus if self.num_gpus else None

            results = {
                "platform": self.platform,
                "dataset": self.dataset,
                "gpu_info": self.gpu_info,
                "timestamp": datetime.now().isoformat(),
                "training_config": {
                    "max_steps": args.max_steps,
                    "global_batch_size": self.global_batch_size,
                    "micro_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "sequence_length": self.sequence_length,
                    "num_gpus": self.num_gpus,
                    "parallel_strategy": self.parallel_strategy,
                },
                "performance_metrics": {
                    "total_steps": len(self.step_times),
                    "total_time_seconds": total_time,
                    "avg_step_time_seconds": avg_step_time,
                    "min_step_time_seconds": min(step_times_no_warmup),
                    "max_step_time_seconds": max(step_times_no_warmup),
                    "steps_per_second": steps_per_second,
                    "tokens_per_second": tokens_per_second,
                    "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                    "throughput_per_gpu_core": steps_per_second / self.gpu_info["gpu_cores"] if self.gpu_info.get("gpu_cores", 0) > 0 else 0,
                },
                "step_times": self.step_times,
                "loss_values": self.loss_values if self.loss_values else [],
                "learning_rates": self.learning_rates if self.learning_rates else [],
                "grad_norms": self.grad_norms if self.grad_norms else [],
            }

            # PyTorch: tensor allocations via torch.cuda.memory_allocated()
            if self.memory_allocated:
                results["memory_metrics"] = {
                    "peak_memory_allocated_gb": max(self.memory_allocated),
                    "avg_memory_allocated_gb": sum(self.memory_allocated) / len(self.memory_allocated),
                    "min_memory_allocated_gb": min(self.memory_allocated),
                    "measurement_method": "torch.cuda.memory_allocated",
                }
            # System: total VRAM via mem_get_info(), sampled per iteration
            if self.memory_reserved:
                results.setdefault("memory_metrics", {}).update({
                    "peak_memory_reserved_gb": max(self.memory_reserved),
                    "avg_memory_reserved_gb": sum(self.memory_reserved) / len(self.memory_reserved),
                    "min_memory_reserved_gb": min(self.memory_reserved),
                    "reserved_measurement_method": "torch.cuda.mem_get_info",
                })

            dataset_suffix = f"_{self.dataset}" if self.dataset else ""
            if self.framework and self.model_name:
                filename = f"train_{self.framework}_{self.model_name}{dataset_suffix}.json"
            elif self.model_name:
                software_stack = self.gpu_info.get("software_stack", self.platform)
                filename = f"train_{software_stack}_{self.model_name}{dataset_suffix}.json"
            else:
                software_stack = self.gpu_info.get("software_stack", self.platform)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"train_{software_stack}_{timestamp}{dataset_suffix}.json"

            filepath = self.output_dir / filename
            results_rounded = round_floats(results, precision=5)

            with open(filepath, 'w') as f:
                json.dump(results_rounded, f, indent=2)

            software_stack = self.gpu_info.get("software_stack", self.platform)
            print(f"\n{'='*60}")
            print(f"BENCHMARK COMPLETE - Platform: {self.platform.upper()} ({software_stack.upper()})")
            print(f"{'='*60}")
            print(f"GPUs: {self.num_gpus}")
            print(f"Total Steps: {results['performance_metrics']['total_steps']}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Avg Step Time: {results['performance_metrics']['avg_step_time_seconds']:.3f}s")

            if results['performance_metrics']['tokens_per_second']:
                print(f"\nThroughput Metrics:")
                print(f"  Total Throughput: {results['performance_metrics']['tokens_per_second']:,.0f} tokens/sec")
                print(f"  Per-GPU Throughput: {results['performance_metrics']['tokens_per_second_per_gpu']:,.0f} tokens/sec/GPU")
                print(f"  (Global batch size: {self.global_batch_size}, Sequence length: {self.sequence_length})")
            else:
                print(f"Throughput: {results['performance_metrics']['steps_per_second']:.3f} steps/s")

            mem_metrics = results.get('memory_metrics', {})
            if mem_metrics:
                alloc = mem_metrics.get('avg_memory_allocated_gb')
                resv = mem_metrics.get('avg_memory_reserved_gb')
                print(f"\nMemory (avg per GPU):")
                if alloc:
                    print(f"  PyTorch: {alloc:.2f} GB")
                if resv:
                    print(f"  System (VRAM used): {resv:.2f} GB")

            print(f"\nResults saved to: {filepath}")
            print(f"{'='*60}\n")
