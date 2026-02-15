"""Training stage orchestrator.

Sets up the environment, resolves the tokenizer, builds Megatron CLI
arguments, and launches the pretrain worker via torchrun.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from rich.panel import Panel
from rich.table import Table

from src import console
from src.train.args import build_megatron_args
from src.train.tokenizer import ensure_hf_token, get_tokenizer_path

WORKER_SCRIPT = Path(__file__).resolve().parent.parent / "train" / "worker.py"


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
    """Run Megatron-Core training from Hydra configuration."""
    c = cfg.theme.colors
    t = cfg.training
    m = cfg.model

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

    # expandable_segments is only supported on CUDA, not ROCm/HIP
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    if not is_rocm:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Required by Megatron when using tensor or context parallelism
    if int(t.parallel.tensor) > 1 or int(t.parallel.context) > 1:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    console.print()

    # -- 2. Tokenizer ---------------------------------------------------------
    _step(cfg, 2, "Tokenizer", m.display_name)

    tokenizer_path = get_tokenizer_path(cfg)
    ensure_hf_token(tokenizer_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True,
    )
    vocab_size = len(tokenizer)

    # If tokenizer was loaded from a remote HF repo, save it locally so the
    # torchrun workers (all ranks) can load from cache without network access.
    data_dir = Path(cfg.paths.data_dir)
    local_tok_dir = data_dir / "tokenizers" / cfg.model.name
    if not local_tok_dir.is_dir():
        local_tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(local_tok_dir))
        tokenizer_path = str(local_tok_dir)
        _kv(cfg, "cached", str(local_tok_dir))
    elif str(local_tok_dir) != tokenizer_path:
        tokenizer_path = str(local_tok_dir)

    _kv(cfg, "path", tokenizer_path)
    _kv(cfg, "vocab", f"{vocab_size:,}")
    console.print()

    # -- 3. Build Megatron args -----------------------------------------------
    _step(cfg, 3, "Args", "Hydra config -> Megatron CLI")

    megatron_args = build_megatron_args(cfg, tokenizer_path, num_gpus)

    _kv(cfg, "args", f"{len(megatron_args)} flags")
    console.print()

    # -- 4. Summary -----------------------------------------------------------
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
    summary.add_row("dataset", f"{t.dataset}  split={t.data_split}")
    summary.add_row("recompute", str(t.recompute.granularity))
    summary.add_row("fusions", f"nvte_fused={f.nvte_fused_attn}  nvte_flash={f.nvte_flash_attn}  primus={f.primus_turbo}")
    prof = t.profiling
    if prof.enabled:
        summary.add_row("profiling", f"steps {prof.step_start}..{prof.step_end}")
    else:
        summary.add_row("profiling", "disabled")

    console.print(Panel(
        summary,
        title=f"[{c.train}]Training[/{c.train}]",
        border_style="dim",
        padding=(1, 2),
    ))
    console.print()

    # -- 5. Launch torchrun ---------------------------------------------------
    _step(cfg, 5, "Train", f"{m.display_name} ({t.train_iters} steps, {num_gpus} GPUs)")

    worker = str(WORKER_SCRIPT)
    # Pick a free port to avoid EADDRINUSE from stale processes
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.bind(("", 0))
        master_port = str(_s.getsockname()[1])

    if num_gpus > 1:
        cmd = [
            "torchrun",
            "--nproc_per_node", str(num_gpus),
            "--master_port", master_port,
            worker,
        ] + megatron_args
    else:
        cmd = [sys.executable, worker] + megatron_args

    _kv(cfg, "launcher", "torchrun" if num_gpus > 1 else "python")
    _kv(cfg, "worker", worker)
    console.print()

    # Workers should read from local cache only -- no HF Hub network calls.
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    # Save training output to log file (like tprimat: training_{platform}_{framework}_{model}_{dataset}.log)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    platform_tag = "rocm" if is_rocm else "cuda"
    model_tag = m.get("name", "model")
    dataset_tag = t.get("dataset", "benchmark")
    log_file = output_dir / f"training_{platform_tag}_megatron_{model_tag}_{dataset_tag}.log"

    t0 = time.time()

    # Tee: stream to both console and log file
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
    )
    with open(log_file, "w") as lf:
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
    proc.wait()

    elapsed = time.time() - t0

    if proc.returncode != 0:
        console.print(
            f"  [{c.error}]failed[/{c.error}]  exit code {proc.returncode}"
        )
        raise RuntimeError(f"Training failed with exit code {proc.returncode}")

    _kv(cfg, "log", str(log_file))
    _kv(cfg, "complete", f"{m.display_name}  {elapsed:.1f}s")
