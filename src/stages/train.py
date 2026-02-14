"""NeMo/Megatron training stage.

Builds a pretrain recipe from Hydra configuration and executes
via nemo_run. All hyperparameters are sourced from structured
YAML configs instead of environment variables.
"""
import gc
import importlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.profiler
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src import console

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightning callbacks
# ---------------------------------------------------------------------------

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:
    from typing import Any

    class Callback:  # type: ignore[no-redef]
        """Stub when Lightning is not installed."""
        def on_train_start(self, trainer: Any, pl_module: Any) -> None: ...
        def on_train_end(self, trainer: Any, pl_module: Any) -> None: ...
        def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None: ...
        def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None: ...


class GCCallback(Callback):
    """Disable garbage collection during training to avoid step-time spikes."""

    def on_train_start(self, trainer, pl_module):
        gc.disable()
        logger.info("GC disabled for training")

    def on_train_end(self, trainer, pl_module):
        gc.enable()
        gc.collect()
        logger.info("GC re-enabled after training")


class ProfilerCallback(Callback):
    """PyTorch/Kineto profiler callback producing TraceLens-compatible traces."""

    def __init__(
        self,
        profile_dir: Path,
        model_name: str,
        step_start: int,
        step_end: int,
        rank_only: int = 0,
    ):
        self.profile_dir = Path(profile_dir)
        self.model_name = model_name
        self.step_start = step_start
        self.step_end = step_end
        self.rank_only = rank_only
        self.profiler = None

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank != self.rank_only:
            return
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        wait = self.step_start
        active = max(1, self.step_end - self.step_start + 1)

        logger.info(
            "PyTorch profiler: steps %d-%d, output %s",
            self.step_start,
            self.step_end,
            self.profile_dir,
        )

        def trace_handler(prof):
            trace_file = self.profile_dir / f"trace_{self.model_name}_rank0.json"
            prof.export_chrome_trace(str(trace_file))
            logger.info("Trace exported to %s", trace_file)
            try:
                stacks_file = self.profile_dir / f"stacks_{self.model_name}_rank0.txt"
                prof.export_stacks(str(stacks_file), "self_cuda_time_total")
            except Exception:
                pass

        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait, warmup=0, active=active, repeat=1,
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=False,
            with_modules=True,
        )
        self.profiler.__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.profiler is not None and trainer.global_rank == self.rank_only:
            self.profiler.step()

    def on_train_end(self, trainer, pl_module):
        if self.profiler is not None and trainer.global_rank == self.rank_only:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
            logger.info("Profiler traces saved to %s", self.profile_dir)


class BenchmarkCallback(Callback):
    """Records per-step timing and GPU memory for benchmarking."""

    def __init__(
        self,
        output_dir: str,
        model_name: str,
        dataset: str,
        global_batch_size: int,
        seq_length: int,
        warmup_steps: int = 0,
        platform: str = "auto",
    ):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.seq_length = seq_length
        self.warmup_steps = warmup_steps
        self.platform = platform

        self.step_times: list[float] = []
        self.memory_allocated: list[float] = []
        self.memory_reserved: list[float] = []
        self._step_start: float | None = None
        self._train_start: float | None = None
        self._num_gpus = 1

    def on_train_start(self, trainer, pl_module):
        self._train_start = time.time()
        self._num_gpus = (
            torch.cuda.device_count() if torch.cuda.is_available() else 1
        )
        if self.platform == "auto":
            is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
            self.platform = "rocm" if is_rocm else "cuda"

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step < self.warmup_steps:
            self._step_start = None
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._step_start is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.step_times.append(time.time() - self._step_start)

        if torch.cuda.is_available():
            self.memory_allocated.append(torch.cuda.memory_allocated() / 1e9)
            free, total = torch.cuda.mem_get_info()
            self.memory_reserved.append((total - free) / 1e9)

    def on_train_end(self, trainer, pl_module):
        is_main = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        if not is_main or len(self.step_times) < 2:
            return

        total_time = time.time() - self._train_start if self._train_start else 0
        measured = self.step_times[1:]
        avg_step = sum(measured) / len(measured)
        steps_per_sec = len(measured) / sum(measured)

        tokens_per_step = self.global_batch_size * self.seq_length
        tokens_per_sec = tokens_per_step / avg_step
        tokens_per_sec_gpu = tokens_per_sec / self._num_gpus

        results = {
            "platform": self.platform,
            "model": self.model_name,
            "dataset": self.dataset,
            "timestamp": datetime.now().isoformat(),
            "gpu_count": self._num_gpus,
            "training_config": {
                "global_batch_size": self.global_batch_size,
                "sequence_length": self.seq_length,
            },
            "performance": {
                "total_steps": len(self.step_times),
                "total_time_s": round(total_time, 2),
                "avg_step_s": round(avg_step, 5),
                "min_step_s": round(min(measured), 5),
                "max_step_s": round(max(measured), 5),
                "steps_per_sec": round(steps_per_sec, 5),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "tokens_per_sec_per_gpu": round(tokens_per_sec_gpu, 2),
            },
            "step_times": [round(t, 5) for t in self.step_times],
        }

        if self.memory_allocated:
            results["memory"] = {
                "peak_allocated_gb": round(max(self.memory_allocated), 2),
                "avg_allocated_gb": round(
                    sum(self.memory_allocated) / len(self.memory_allocated), 2,
                ),
                "peak_reserved_gb": round(max(self.memory_reserved), 2),
            }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / f"bench_{self.platform}_{self.model_name}_{self.dataset}.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Benchmark results saved to %s", filepath)


# ---------------------------------------------------------------------------
# NeMo recipe lookup
# ---------------------------------------------------------------------------

RECIPES = {
    "llama31_8b": ("nemo.collections.llm", "llama31_8b", "pretrain_recipe"),
    "qwen25_7b": ("nemo.collections.llm", "qwen25_7b", "pretrain_recipe"),
}


def _get_tokenizer_path(cfg: DictConfig) -> str:
    """Resolve tokenizer: prefer local cache, fallback to HuggingFace."""
    data_dir = Path(cfg.paths.data_dir)
    local = data_dir / "tokenizers" / cfg.model.name
    if local.is_dir():
        logger.info("Using local tokenizer: %s", local)
        return str(local)
    logger.info("Using HuggingFace tokenizer: %s", cfg.model.tokenizer_path)
    return cfg.model.tokenizer_path


def _ensure_hf_token(repo_id: str) -> None:
    """Ensure HF_TOKEN is set for gated repos."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    elif repo_id.startswith("meta-llama/"):
        raise RuntimeError(
            f"HF_TOKEN required for gated repo: {repo_id}. "
            "Set HF_TOKEN in the environment or in secrets.env."
        )


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------

def _step(cfg: DictConfig, n: int, title: str, detail: str = "") -> None:
    c = cfg.theme.colors
    console.print(
        f"[{c.train}]{n}.[/{c.train}] [bold]{title}[/bold]  [dim]{detail}[/dim]"
    )
    console.print()


def _kv(cfg: DictConfig, key: str, val: str) -> None:
    c = cfg.theme.colors
    console.print(f"  [{c.success}]{key}[/{c.success}]  {val}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(cfg: DictConfig) -> None:
    """Run NeMo/Megatron training from Hydra configuration."""
    c = cfg.theme.colors
    t = cfg.training

    # -- Display training config panel ----------------------------------------
    train_dict = OmegaConf.to_container(t, resolve=True)
    yaml_str = yaml.dump(train_dict, default_flow_style=False, sort_keys=False)
    console.print(Panel(
        Syntax(
            yaml_str, "yaml",
            theme=cfg.theme.syntax,
            line_numbers=False,
            background_color="default",
        ),
        title=f"[{c.train}]Training[/{c.train}]",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()

    # -- 1. Environment setup -------------------------------------------------
    _step(cfg, 1, "Setup", "seeds, CUDA, environment")

    seed = int(cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    num_gpus = torch.cuda.device_count()
    _kv(cfg, "cuda", f"{num_gpus} GPU(s)")
    _kv(cfg, "seed", str(seed))
    console.print()

    # -- 2. Tokenizer ---------------------------------------------------------
    _step(cfg, 2, "Tokenizer", cfg.model.display_name)

    tokenizer_path = _get_tokenizer_path(cfg)
    _ensure_hf_token(tokenizer_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True,
    )
    vocab_size = len(tokenizer)
    base_vocab_size = getattr(tokenizer, "vocab_size", vocab_size)

    _kv(cfg, "path", tokenizer_path)
    _kv(cfg, "vocab", f"{vocab_size:,}")
    if vocab_size != base_vocab_size:
        _kv(cfg, "base", f"{base_vocab_size:,}")
    console.print()

    # -- 3. Recipe ------------------------------------------------------------
    recipe_key = cfg.model.recipe
    _step(cfg, 3, "Recipe", f"{recipe_key} pretrain")

    if recipe_key not in RECIPES:
        raise ValueError(
            f"Unknown recipe: {recipe_key}. "
            f"Supported: {list(RECIPES.keys())}"
        )

    mod_path, attr, fn_name = RECIPES[recipe_key]
    mod = importlib.import_module(mod_path)
    recipe_mod = getattr(mod, attr)
    recipe_fn = getattr(recipe_mod, fn_name)

    recipe = recipe_fn(
        name=f"{recipe_key}_pretrain",
        dir=str(cfg.paths.data_dir),
        num_nodes=1,
        num_gpus_per_node=num_gpus,
    )

    recipe.model.config.vocab_size = vocab_size
    recipe.model.config.init_method_std = float(t.init_method_std)

    _kv(cfg, "model", cfg.model.display_name)
    _kv(cfg, "recipe", recipe_key)
    _kv(cfg, "init_std", str(t.init_method_std))
    console.print()

    # -- 4. Parallelism (Section 2) -------------------------------------------
    tp = int(t.parallel.tensor)
    pp = int(t.parallel.pipeline)
    dp = int(t.parallel.data)
    cp = int(t.parallel.context)
    sp = bool(t.parallel.sequence)
    vp = t.parallel.get("virtual_pipeline", None)
    _step(cfg, 4, "Parallelism", f"TP={tp}  PP={pp}  DP={dp}  CP={cp}")

    from nemo.lightning import MegatronStrategy

    # DDP config from Section 3 precision params
    try:
        from megatron.core.distributed import DistributedDataParallelConfig

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=bool(t.grad_reduce_in_fp32),
            overlap_grad_reduce=bool(t.overlap_grad_reduce),
            overlap_param_gather=bool(t.overlap_param_gather),
            use_distributed_optimizer=bool(t.distributed_optimizer),
        )
        _kv(cfg, "ddp", f"dist_optim={t.distributed_optimizer}  overlap_grad={t.overlap_grad_reduce}  overlap_param={t.overlap_param_gather}")
    except ImportError:
        ddp_config = "megatron"
        _kv(cfg, "ddp", "megatron default")

    recipe.trainer.strategy = MegatronStrategy(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        virtual_pipeline_model_parallel_size=vp,
        sequence_parallel=sp,
        ddp=ddp_config,
    )

    _kv(cfg, "sequence_parallel", str(sp))
    console.print()

    # -- 5. Data (Section 1) --------------------------------------------------
    dataset_name = str(t.dataset)
    data_format = str(t.data_format)
    data_split = str(t.data_split)
    data_dir = Path(cfg.paths.data_dir)

    # Resolve data path: explicit override or default per format
    if t.get("data_path") and t.data_path is not None:
        data_path = Path(str(t.data_path))
    elif data_format == "energon":
        data_path = data_dir / "webdataset"
    else:
        data_path = data_dir / f"{dataset_name}-train"

    _step(cfg, 5, "Data", f"{dataset_name}  format={data_format}")

    if data_format == "energon":
        # ── Energon: WebDataset .tar shards + .nv-meta/ ──
        nv_meta = data_path / ".nv-meta"
        if not nv_meta.is_dir():
            raise FileNotFoundError(
                f"Energon metadata not found: {nv_meta}\n"
                "Run data preparation first: make data"
            )
        for required in ("dataset.yaml", "split.yaml"):
            if not (nv_meta / required).exists():
                raise FileNotFoundError(
                    f"Missing {nv_meta / required}\n"
                    "Re-run: make data"
                )

        num_workers = int(t.get("num_workers", 4))

        try:
            from nemo.collections.multimodal.data.energon import EnergonDataModule

            recipe.data = EnergonDataModule(
                path=str(data_path),
                micro_batch_size=int(t.micro_batch_size),
                global_batch_size=int(t.global_batch_size),
                num_workers=num_workers,
                seq_length=int(t.seq_length),
            )
        except ImportError:
            from nemo.collections.multimodal.data.energon import (
                EnergonMultiModalDataModule,
            )

            recipe.data = EnergonMultiModalDataModule(
                path=str(data_path),
                micro_batch_size=int(t.micro_batch_size),
                global_batch_size=int(t.global_batch_size),
                num_workers=num_workers,
                seq_length=int(t.seq_length),
            )

        _kv(cfg, "format", "energon (WebDataset + .nv-meta)")
        _kv(cfg, "path", str(data_path))
        _kv(cfg, "workers", str(num_workers))

    elif data_format == "megatron":
        # ── Megatron binary: .bin/.idx tokenized data ──
        bin_path = str(data_path)
        for ext in (".idx", ".bin"):
            fpath = bin_path + ext
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Dataset not found: {fpath}\n"
                    "Run data preparation first."
                )

        if t.verify_data:
            try:
                from data.verify_data import verify_dataset

                logger.info(
                    "Verifying dataset (samples=%d, full_scan=%s)",
                    t.verify_samples,
                    t.verify_full_scan,
                )
                ok = verify_dataset(
                    bin_path,
                    tokenizer_path,
                    int(t.verify_samples),
                    bool(t.verify_full_scan),
                )
                if not ok:
                    raise RuntimeError("Dataset verification failed")
            except ImportError:
                logger.warning(
                    "data.verify_data not available, skipping verification"
                )

        from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule

        index_cache_dir = str(data_dir / "index_cache")
        os.makedirs(index_cache_dir, exist_ok=True)

        recipe.data = PreTrainingDataModule(
            paths=bin_path,
            seq_length=int(t.seq_length),
            micro_batch_size=int(t.micro_batch_size),
            global_batch_size=int(t.global_batch_size),
            split=data_split,
            seed=seed,
            index_mapping_dir=index_cache_dir,
        )

        _kv(cfg, "format", "megatron (.bin/.idx)")
        _kv(cfg, "path", bin_path)
        _kv(cfg, "split", data_split)

    else:
        raise ValueError(
            f"Unknown data_format: '{data_format}'. "
            "Must be 'energon' or 'megatron'."
        )

    _kv(cfg, "samples", f"{t.num_samples:,}")
    _kv(
        cfg,
        "batching",
        f"MBS={t.micro_batch_size}  GBS={t.global_batch_size}  GA={t.gradient_accumulation}  SL={t.seq_length}",
    )
    console.print()

    # -- 6. Optimizer & scheduler (Section 1) ---------------------------------
    train_iters = int(t.train_iters)
    _step(
        cfg, 6, "Optimizer",
        f"lr={t.learning_rate}  scheduler={t.lr_scheduler}  warmup={t.warmup_steps}",
    )

    recipe.trainer.max_steps = train_iters
    recipe.optim.config.lr = float(t.learning_rate)
    recipe.optim.config.min_lr = float(t.min_lr)
    recipe.optim.config.weight_decay = float(t.weight_decay)
    recipe.optim.config.adam_beta1 = float(t.beta1)
    recipe.optim.config.adam_beta2 = float(t.beta2)

    # NeMo warmup formula uses (warmup_steps+1) as divisor;
    # subtract 1 to align with Megatron's lr = max_lr * iter / warmup_iters
    recipe.optim.lr_scheduler.warmup_steps = max(int(t.warmup_steps) - 1, 0)
    recipe.optim.lr_scheduler.constant_steps = 0
    recipe.optim.lr_scheduler.max_steps = train_iters
    recipe.optim.lr_scheduler.min_lr = float(t.min_lr)

    _kv(cfg, "adam", f"beta1={t.beta1}  beta2={t.beta2}")
    _kv(cfg, "weight_decay", str(t.weight_decay))
    _kv(cfg, "steps", f"{train_iters}  warmup={t.warmup_steps}")
    _kv(cfg, "min_lr", str(t.min_lr))
    console.print()

    # -- 7. Precision (Section 3) ---------------------------------------------
    _step(cfg, 7, "Precision", str(t.precision))

    # FP8
    if t.fp8_hybrid:
        recipe.model.config.fp8 = "hybrid"
    else:
        recipe.model.config.fp8 = None
    recipe.model.config.fp8_param = bool(t.fp8_param)

    # Residual connection precision
    recipe.model.config.fp32_residual_connection = bool(t.fp32_residual_connection)

    _kv(cfg, "compute", str(t.precision))
    _kv(cfg, "fp8_hybrid", str(t.fp8_hybrid))
    _kv(cfg, "fp8_param", str(t.fp8_param))
    _kv(cfg, "fp32_residual", str(t.fp32_residual_connection))
    _kv(cfg, "dist_optimizer", str(t.distributed_optimizer))
    _kv(cfg, "grad_reduce_fp32", str(t.grad_reduce_in_fp32))
    console.print()

    # -- 8. Fusions & Attention (Section 4) -----------------------------------
    f = t.fusions
    _step(cfg, 8, "Fusions", "configurable per-platform")

    recipe.model.config.bias_activation_fusion = bool(f.bias_activation)
    recipe.model.config.bias_dropout_fusion = bool(f.bias_dropout)
    recipe.model.config.masked_softmax_fusion = bool(f.masked_softmax)
    recipe.model.config.persist_layer_norm = bool(f.persist_layer_norm)
    recipe.model.config.apply_rope_fusion = bool(f.apply_rope)
    recipe.model.config.cross_entropy_loss_fusion = bool(f.cross_entropy_loss)
    recipe.model.config.gradient_accumulation_fusion = bool(f.gradient_accumulation)

    # Attention backend (TransformerEngine env vars for NVIDIA)
    if f.nvte_fused_attn:
        os.environ["NVTE_FUSED_ATTN"] = "1"
    else:
        os.environ.pop("NVTE_FUSED_ATTN", None)

    if f.nvte_flash_attn:
        os.environ["NVTE_FLASH_ATTN"] = "1"
    else:
        os.environ.pop("NVTE_FLASH_ATTN", None)

    os.environ.pop("NVTE_UNFUSED_ATTN", None)
    recipe.model.config.attention_backend = "auto"

    # AMD-specific Primus flags
    if f.primus_turbo:
        os.environ["ENABLE_PRIMUS_TURBO"] = "1"
    if f.turbo_attention:
        os.environ["USE_TURBO_ATTENTION"] = "1"

    enabled = [k for k in (
        "bias_activation", "bias_dropout", "masked_softmax",
        "persist_layer_norm", "apply_rope", "cross_entropy_loss",
        "gradient_accumulation", "flash_attn",
    ) if getattr(f, k, False)]
    _kv(cfg, "enabled", ", ".join(enabled) if enabled else "none")
    _kv(cfg, "nvte", f"fused={f.nvte_fused_attn}  flash={f.nvte_flash_attn}")
    if f.primus_turbo or f.turbo_attention:
        _kv(cfg, "amd", f"primus_turbo={f.primus_turbo}  turbo_attn={f.turbo_attention}")
    console.print()

    # -- 9. Recompute ---------------------------------------------------------
    rc = t.recompute
    recipe.model.config.recompute_granularity = str(rc.granularity) if rc.granularity else None
    recipe.model.config.recompute_method = str(rc.method) if rc.method else None
    recipe.model.config.recompute_num_layers = int(rc.num_layers) if rc.num_layers else None

    # -- 10. Checkpointing & logging ------------------------------------------
    recipe.trainer.enable_checkpointing = bool(t.checkpointing)
    if not t.checkpointing:
        recipe.log.ckpt = None
        recipe.resume = None
    recipe.log.tensorboard = None
    recipe.log.wandb = None
    recipe.trainer.val_check_interval = train_iters + 1
    recipe.trainer.check_val_every_n_epoch = None
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.num_sanity_val_steps = 0

    # -- 11. Callbacks --------------------------------------------------------
    _step(cfg, 9, "Callbacks", "benchmark, GC, profiling")

    if recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []

    recipe.trainer.callbacks.append(BenchmarkCallback(
        output_dir=str(cfg.paths.output_dir),
        model_name=cfg.model.name,
        dataset=dataset_name,
        global_batch_size=int(t.global_batch_size),
        seq_length=int(t.seq_length),
        warmup_steps=0,
    ))
    recipe.trainer.callbacks.append(GCCallback())
    cb_names = ["benchmark", "gc"]

    if t.profiling.enabled:
        profile_dir = Path(cfg.paths.output_dir) / "profiles"
        try:
            from megatron.bridge.training.config import ProfilingConfig

            profile_dir.mkdir(parents=True, exist_ok=True)
            recipe.profiling = ProfilingConfig(
                use_pytorch_profiler=True,
                profile_step_start=int(t.profiling.step_start),
                profile_step_end=int(t.profiling.step_end),
                profile_ranks=[0],
                record_shapes=True,
                record_memory_history=False,
                memory_snapshot_path=str(
                    profile_dir / "memory_snapshot.pickle"
                ),
            )
            cb_names.append("bridge-profiler")
        except (ImportError, AttributeError):
            recipe.trainer.callbacks.append(ProfilerCallback(
                profile_dir=profile_dir,
                model_name=cfg.model.name,
                step_start=int(t.profiling.step_start),
                step_end=int(t.profiling.step_end),
            ))
            cb_names.append("pytorch-profiler")

    _kv(cfg, "active", ", ".join(cb_names))
    console.print()

    # -- 12. Summary ----------------------------------------------------------
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row("model", cfg.model.display_name)
    summary.add_row("recipe", recipe_key)
    summary.add_row("precision", f"{t.precision}  fp8={t.fp8_hybrid}")
    summary.add_row("parallelism", f"TP={tp}  PP={pp}  DP={dp}  CP={cp}  SP={sp}")
    summary.add_row("batching", f"MBS={t.micro_batch_size}  GBS={t.global_batch_size}  GA={t.gradient_accumulation}")
    summary.add_row("optimizer", f"lr={t.learning_rate}  wd={t.weight_decay}  {t.lr_scheduler}")
    summary.add_row("steps", f"{train_iters}  warmup={t.warmup_steps}")
    summary.add_row("dataset", f"{dataset_name}  split={data_split}  samples={t.num_samples}")
    summary.add_row("recompute", str(rc.granularity))
    summary.add_row("checkpointing", str(t.checkpointing))

    console.print(Panel(
        summary,
        title=f"[{c.train}]Training[/{c.train}]",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()

    # -- 13. Run training -----------------------------------------------------
    _step(cfg, 10, "Train", f"{cfg.model.display_name} ({train_iters} steps)")

    import nemo_run

    nemo_run.run(recipe, direct=True)

    _kv(cfg, "complete", cfg.model.display_name)
