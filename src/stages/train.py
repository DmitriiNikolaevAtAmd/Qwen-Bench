"""Megatron-Core training stage.

Converts Hydra configuration to Megatron command-line arguments
and launches the pretrain worker via torchrun.
"""
import json
import logging
import os
import struct
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

def _expected_idx_size(num_sequences: int) -> int:
    """Return the exact expected .idx file size for the canonical Megatron format.

    Layout: magic(9) + version(Q=8) + dtype(B=1) + num_seq(Q=8) + num_doc(Q=8)
            + sizes(num_seq*4) + pointers(num_seq*8) + doc_idx((num_seq+1)*8)
    """
    header = 9 + 8 + 1 + 8 + 8  # = 34
    return header + num_sequences * 4 + num_sequences * 8 + (num_sequences + 1) * 8


def _write_megatron_idx(idx_path: Path, sizes: list[int], doc_idx: list[int]) -> None:
    """Write a Megatron MMapIndexedDataset .idx file."""
    dtype_size = np.dtype(np.int32).itemsize
    with open(idx_path, "wb") as index:
        index.write(b"MMIDIDX\x00\x00")                       # 9 bytes  magic
        index.write(struct.pack("<Q", 1))                      # 8 bytes  version
        index.write(struct.pack("<B", 4))                      # 1 byte   dtype (int32)
        index.write(struct.pack("<Q", len(sizes)))             # 8 bytes  num sequences
        index.write(struct.pack("<Q", len(doc_idx)))           # 8 bytes  num documents

        np.array(sizes, dtype=np.int32).tofile(index)          # sizes

        pointers = np.zeros(len(sizes), dtype=np.int64)        # pointers
        for i in range(1, len(sizes)):
            pointers[i] = pointers[i - 1] + sizes[i - 1] * dtype_size
        pointers.tofile(index)

        np.array(doc_idx, dtype=np.int64).tofile(index)        # doc indices


def _tokenize_captions_to_megatron(
    cfg: DictConfig,
    jsonl_path: Path,
    bin_path: Path,
    idx_path: Path,
    seq_length: int,
    tokenizer,
) -> None:
    """Tokenize pseudo_camera captions from JSONL into Megatron binary format.

    Reads each ``{"image": ..., "caption": ...}`` line, tokenizes the caption
    text, pads or truncates to ``seq_length``, and writes the result as a
    Megatron MMapIndexedDataset (.bin + .idx).
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dtype = np.int32

    captions: list[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(json.loads(line)["caption"])

    num_samples = len(captions)
    _kv(cfg, "tokenizing",
         f"{num_samples:,} captions from {jsonl_path.name}  seq_length={seq_length}")

    sizes: list[int] = []
    doc_idx: list[int] = [0]

    with open(bin_path, "wb") as data_file:
        for caption in captions:
            token_ids = tokenizer.encode(caption, add_special_tokens=True)
            # Truncate or pad to exact seq_length
            if len(token_ids) > seq_length:
                token_ids = token_ids[:seq_length]
            elif len(token_ids) < seq_length:
                token_ids = token_ids + [pad_id] * (seq_length - len(token_ids))
            tokens = np.array(token_ids, dtype=dtype)
            data_file.write(tokens.tobytes(order="C"))
            sizes.append(seq_length)
            doc_idx.append(len(sizes))

    _write_megatron_idx(idx_path, sizes, doc_idx)

    bin_mb = bin_path.stat().st_size / (1024 * 1024)
    idx_mb = idx_path.stat().st_size / (1024 * 1024)
    _kv(cfg, "data", f"{bin_path.name} ({bin_mb:.1f} MB)  {idx_path.name} ({idx_mb:.2f} MB)")


def _generate_synthetic_data(
    cfg: DictConfig,
    bin_path: Path,
    idx_path: Path,
    num_samples: int,
    seq_length: int,
    vocab_size: int,
) -> None:
    """Generate synthetic random-token Megatron binary data for benchmarking."""
    dtype = np.int32

    _kv(cfg, "generating",
         f"{num_samples:,} synthetic samples  seq_length={seq_length}  vocab={vocab_size:,}")

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(cfg.seed)

    sizes: list[int] = []
    doc_idx: list[int] = [0]

    with open(bin_path, "wb") as data_file:
        for _ in range(num_samples):
            tokens = rng.randint(0, vocab_size, size=seq_length, dtype=dtype)
            data_file.write(tokens.tobytes(order="C"))
            sizes.append(seq_length)
            doc_idx.append(len(sizes))

    _write_megatron_idx(idx_path, sizes, doc_idx)

    bin_mb = bin_path.stat().st_size / (1024 * 1024)
    idx_mb = idx_path.stat().st_size / (1024 * 1024)
    _kv(cfg, "data", f"{bin_path.name} ({bin_mb:.1f} MB)  {idx_path.name} ({idx_mb:.2f} MB)")


def _ensure_megatron_data(cfg: DictConfig, vocab_size: int, tokenizer=None) -> None:
    """Ensure Megatron binary (.bin/.idx) data exists.

    When ``dataset == "pseudo_camera"`` and the raw JSONL is available,
    tokenizes real captions.  Otherwise generates synthetic random-token
    data for benchmarking.
    """
    data_dir = Path(cfg.paths.data_dir)
    t = cfg.training
    prefix = data_dir / f"{t.dataset}-train"
    bin_path = Path(f"{prefix}.bin")
    idx_path = Path(f"{prefix}.idx")

    num_samples = int(t.num_samples)
    seq_length = int(t.seq_length)
    dtype_size = np.dtype(np.int32).itemsize

    # Validate existing files by exact expected sizes
    expected_bin = num_samples * seq_length * dtype_size
    expected_idx = _expected_idx_size(num_samples)

    if bin_path.exists() and idx_path.exists():
        if (bin_path.stat().st_size == expected_bin
                and idx_path.stat().st_size == expected_idx):
            _kv(cfg, "data", f"found {bin_path.name} + {idx_path.name}")
            return
        _kv(cfg, "data", "removing mismatched data files, will regenerate")
        bin_path.unlink(missing_ok=True)
        idx_path.unlink(missing_ok=True)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Use real caption data when available, otherwise synthetic
    dataset_name = str(t.dataset)
    jsonl_path = data_dir / "pseudo-camera-raw.jsonl"

    if dataset_name == "pseudo_camera" and jsonl_path.exists() and tokenizer is not None:
        _tokenize_captions_to_megatron(
            cfg, jsonl_path, bin_path, idx_path, seq_length, tokenizer,
        )
    else:
        _generate_synthetic_data(
            cfg, bin_path, idx_path, num_samples, seq_length, vocab_size,
        )


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

    # Set both names -- ROCm/HIP may still look for the old CUDA name
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Required by Megatron when using tensor or context parallelism
    if int(t.parallel.tensor) > 1 or int(t.parallel.context) > 1:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
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
        _ensure_megatron_data(cfg, vocab_size, tokenizer=tokenizer)
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
    # Pick a free port to avoid EADDRINUSE from stale processes
    import socket
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

    t0 = time.time()
    result = subprocess.run(cmd, env=os.environ.copy())

    elapsed = time.time() - t0

    if result.returncode != 0:
        console.print(
            f"  [{c.error}]failed[/{c.error}]  exit code {result.returncode}"
        )
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    _kv(cfg, "complete", f"{m.display_name}  {elapsed:.1f}s")
