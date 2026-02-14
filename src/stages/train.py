"""Megatron-Core training stage.

Converts Hydra configuration to Megatron command-line arguments
and launches the pretrain worker via torchrun.
"""
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src import console

logger = logging.getLogger(__name__)

WORKER_SCRIPT = Path(__file__).resolve().parent.parent / "worker.py"


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def _get_tokenizer_path(cfg: DictConfig) -> str:
    """Resolve tokenizer: prefer local cache, fallback to HuggingFace."""
    data_dir = Path(cfg.paths.data_dir)
    local = data_dir / "tokenizers" / cfg.model.name
    if local.is_dir():
        return str(local)
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
# Synthetic Megatron binary data generator (benchmarking)
# ---------------------------------------------------------------------------

def _ensure_megatron_data(cfg: DictConfig, vocab_size: int) -> None:
    """Ensure Megatron binary (.bin/.idx) data exists.

    For benchmarking, generates synthetic random-token data using
    Megatron's own ``MMapIndexedDatasetBuilder`` so the format is
    guaranteed to match the reader.
    """
    from megatron.core.datasets.indexed_dataset import MMapIndexedDatasetBuilder

    data_dir = Path(cfg.paths.data_dir)
    t = cfg.training
    prefix = data_dir / f"{t.dataset}-train"
    bin_path = Path(f"{prefix}.bin")
    idx_path = Path(f"{prefix}.idx")

    num_samples = int(t.num_samples)
    seq_length = int(t.seq_length)

    # Validate existing files by checking .bin size matches expectations
    expected_bin_size = num_samples * seq_length * np.dtype(np.int32).itemsize
    if bin_path.exists() and idx_path.exists():
        if bin_path.stat().st_size == expected_bin_size:
            _kv(cfg, "data", f"found {bin_path.name} + {idx_path.name}")
            return
        _kv(cfg, "data", "removing mismatched data files, will regenerate")
        bin_path.unlink(missing_ok=True)
        idx_path.unlink(missing_ok=True)

    _kv(cfg, "generating",
         f"{num_samples:,} synthetic samples  seq_length={seq_length}  vocab={vocab_size:,}")

    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(cfg.seed)

    builder = MMapIndexedDatasetBuilder(str(bin_path), dtype=np.int32)
    for _ in range(num_samples):
        tokens = torch.from_numpy(
            rng.randint(0, vocab_size, size=seq_length).astype(np.int32)
        )
        builder.add_item(tokens)
        builder.end_document()
    builder.finalize(str(idx_path))

    bin_mb = bin_path.stat().st_size / (1024 * 1024)
    idx_mb = idx_path.stat().st_size / (1024 * 1024)
    _kv(cfg, "data", f"{bin_path.name} ({bin_mb:.1f} MB)  {idx_path.name} ({idx_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Megatron args builder
# ---------------------------------------------------------------------------

def _build_megatron_args(cfg: DictConfig, tokenizer_path: str, num_gpus: int) -> list[str]:
    """Convert Hydra config to Megatron-style command-line arguments."""
    t = cfg.training
    m = cfg.model.architecture
    data_dir = Path(cfg.paths.data_dir)

    args: list[str] = []

    def add(flag: str, value=None):
        args.append(flag)
        if value is not None:
            args.append(str(value))

    # -- Model architecture ---------------------------------------------------
    add("--num-layers", m.num_layers)
    add("--hidden-size", m.hidden_size)
    add("--ffn-hidden-size", m.ffn_hidden_size)
    add("--num-attention-heads", m.num_attention_heads)
    if m.num_query_groups != m.num_attention_heads:
        add("--group-query-attention")
        add("--num-query-groups", m.num_query_groups)
    add("--max-position-embeddings", m.max_position_embeddings)
    add("--init-method-std", t.init_method_std)
    add("--normalization", m.normalization)
    add("--norm-epsilon", m.norm_epsilon)
    if m.swiglu:
        add("--swiglu")
    if m.rotary:
        add("--use-rotary-position-embeddings")
    if m.untie_embeddings_and_output_weights:
        add("--untie-embeddings-and-output-weights")

    # -- Training algorithm ---------------------------------------------------
    add("--seq-length", t.seq_length)
    add("--micro-batch-size", t.micro_batch_size)
    add("--global-batch-size", t.global_batch_size)
    add("--lr", t.learning_rate)
    add("--min-lr", t.min_lr)
    add("--weight-decay", t.weight_decay)
    add("--adam-beta1", t.beta1)
    add("--adam-beta2", t.beta2)
    add("--lr-warmup-iters", t.warmup_steps)
    add("--train-iters", t.train_iters)
    add("--lr-decay-iters", t.train_iters)
    add("--lr-decay-style", t.lr_scheduler)
    add("--seed", cfg.seed)
    add("--log-interval", 10)
    add("--eval-interval", t.train_iters)  # evaluate once at the end
    add("--eval-iters", 0)                 # no eval samples (100/0/0 split)

    # -- Parallelism ----------------------------------------------------------
    add("--tensor-model-parallel-size", t.parallel.tensor)
    add("--pipeline-model-parallel-size", t.parallel.pipeline)
    if int(t.parallel.context) > 1:
        add("--context-parallel-size", t.parallel.context)
    if t.parallel.sequence:
        add("--sequence-parallel")

    # -- Precision ------------------------------------------------------------
    precision = str(t.precision).lower()
    if precision == "bf16":
        add("--bf16")
    elif precision == "fp16":
        add("--fp16")
    if t.fp32_residual_connection:
        add("--fp32-residual-connection")
    if t.distributed_optimizer:
        add("--use-distributed-optimizer")
    if t.overlap_grad_reduce:
        add("--overlap-grad-reduce")
    if t.overlap_param_gather:
        add("--overlap-param-gather")

    # -- Fusions (Megatron uses --no-* flags to disable) ----------------------
    f = t.fusions
    if not f.bias_activation:
        add("--no-bias-gelu-fusion")
    if not f.bias_dropout:
        add("--no-bias-dropout-fusion")
    if not f.masked_softmax:
        add("--no-masked-softmax-fusion")
    if not f.persist_layer_norm:
        add("--no-persist-layer-norm")
    if not f.apply_rope:
        add("--no-rope-fusion")
    if not f.gradient_accumulation:
        add("--no-gradient-accumulation-fusion")

    # -- Recompute ------------------------------------------------------------
    rc = t.recompute
    if rc.granularity:
        add("--recompute-granularity", rc.granularity)
    if rc.method:
        add("--recompute-method", rc.method)
    if rc.num_layers:
        add("--recompute-num-layers", rc.num_layers)

    # -- Transformer implementation (local = no TransformerEngine dependency) --
    add("--transformer-impl", "local")

    # -- Tokenizer ------------------------------------------------------------
    add("--tokenizer-type", "HuggingFaceTokenizer")
    add("--tokenizer-model", tokenizer_path)

    # -- Data -----------------------------------------------------------------
    data_format = str(t.data_format)
    if t.get("data_path") and t.data_path is not None:
        data_path = str(t.data_path)
    elif data_format == "megatron":
        data_path = str(data_dir / f"{t.dataset}-train")
    else:
        data_path = str(data_dir / "webdataset")

    add("--data-path", data_path)
    add("--split", t.data_split)
    add("--data-cache-path", str(data_dir / "index_cache"))

    # -- Checkpointing (disabled for benchmarking) ----------------------------
    if not t.checkpointing:
        add("--no-save-optim")
        add("--no-save-rng")
        add("--no-load-optim")
        add("--no-load-rng")

    return args


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(cfg: DictConfig) -> None:
    """Run Megatron-Core training from Hydra configuration."""
    c = cfg.theme.colors
    t = cfg.training
    m = cfg.model

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

    # -- 1. Environment -------------------------------------------------------
    _step(cfg, 1, "Environment", "CUDA, seeds")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    num_gpus = torch.cuda.device_count()
    _kv(cfg, "cuda", f"{num_gpus} GPU(s)")
    _kv(cfg, "seed", str(cfg.seed))

    # Attention env vars (NVIDIA TransformerEngine)
    f = t.fusions
    if f.nvte_fused_attn:
        os.environ["NVTE_FUSED_ATTN"] = "1"
    if f.nvte_flash_attn:
        os.environ["NVTE_FLASH_ATTN"] = "1"
    os.environ.pop("NVTE_UNFUSED_ATTN", None)

    # AMD Primus flags
    if f.primus_turbo:
        os.environ["ENABLE_PRIMUS_TURBO"] = "1"
    if f.turbo_attention:
        os.environ["USE_TURBO_ATTENTION"] = "1"

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    console.print()

    # -- 2. Tokenizer ---------------------------------------------------------
    _step(cfg, 2, "Tokenizer", m.display_name)

    tokenizer_path = _get_tokenizer_path(cfg)
    _ensure_hf_token(tokenizer_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True,
    )
    vocab_size = len(tokenizer)

    _kv(cfg, "path", tokenizer_path)
    _kv(cfg, "vocab", f"{vocab_size:,}")
    console.print()

    # -- 3. Ensure training data exists ---------------------------------------
    data_format = str(t.data_format)
    if data_format == "megatron":
        _step(cfg, 3, "Data", "Megatron binary (.bin/.idx)")
        _ensure_megatron_data(cfg, vocab_size)
        console.print()

    # -- 4. Build Megatron args -----------------------------------------------
    _step(cfg, 4, "Args", "Hydra config -> Megatron CLI")

    megatron_args = _build_megatron_args(cfg, tokenizer_path, num_gpus)

    args_str = " \\\n    ".join(megatron_args)
    _kv(cfg, "args", f"{len(megatron_args)} flags")
    console.print()

    # -- 5. Summary -----------------------------------------------------------
    arch = m.architecture
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row("model", f"{m.display_name}  ({arch.num_layers}L, {arch.hidden_size}H, {arch.num_attention_heads}A)")
    summary.add_row("precision", f"{t.precision}  fp8={t.fp8_hybrid}")
    summary.add_row("parallelism", f"TP={t.parallel.tensor}  PP={t.parallel.pipeline}  DP={t.parallel.data}  CP={t.parallel.context}  SP={t.parallel.sequence}")
    summary.add_row("batching", f"MBS={t.micro_batch_size}  GBS={t.global_batch_size}  GA={t.gradient_accumulation}  SL={t.seq_length}")
    summary.add_row("optimizer", f"lr={t.learning_rate}  wd={t.weight_decay}  {t.lr_scheduler}  warmup={t.warmup_steps}")
    summary.add_row("steps", str(t.train_iters))
    summary.add_row("dataset", f"{t.dataset}  format={t.data_format}  split={t.data_split}")
    summary.add_row("recompute", str(t.recompute.granularity))
    summary.add_row("fusions", f"nvte_fused={f.nvte_fused_attn}  nvte_flash={f.nvte_flash_attn}  primus={f.primus_turbo}")

    console.print(Panel(
        summary,
        title=f"[{c.train}]Training[/{c.train}]",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()

    # -- 6. Launch torchrun ---------------------------------------------------
    _step(cfg, 6, "Train", f"{m.display_name} ({t.train_iters} steps, {num_gpus} GPUs)")

    worker = str(WORKER_SCRIPT)
    if num_gpus > 1:
        cmd = [
            "torchrun",
            "--nproc_per_node", str(num_gpus),
            worker,
        ] + megatron_args
    else:
        cmd = [sys.executable, worker] + megatron_args

    _kv(cfg, "launcher", "torchrun" if num_gpus > 1 else "python")
    _kv(cfg, "worker", worker)
    console.print()

    t0 = time.time()
    result = subprocess.run(cmd, env=os.environ.copy())

    elapsed = time.time() - t0

    if result.returncode != 0:
        console.print(
            f"  [{c.error}]failed[/{c.error}]  exit code {result.returncode}"
        )
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    _kv(cfg, "complete", f"{m.display_name}  {elapsed:.1f}s")
