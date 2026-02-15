"""Build Megatron-style command-line arguments from Hydra config."""
from pathlib import Path

from omegaconf import DictConfig


def build_megatron_args(cfg: DictConfig, tokenizer_path: str, num_gpus: int) -> list[str]:
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

    # -- Data (Energon / WebDataset) ------------------------------------------
    if t.get("data_path") and t.data_path is not None:
        data_path = str(t.data_path)
    else:
        data_path = str(data_dir / "webdataset")

    add("--data-path", data_path)
    add("--split", t.data_split)
    add("--data-cache-path", str(data_dir / "index_cache"))

    # -- Profiling (disabled by default) ---------------------------------------
    prof = t.profiling
    if prof.enabled:
        add("--profile")
        add("--profile-step-start", prof.step_start)
        add("--profile-step-end", prof.step_end)

    # -- Checkpointing (disabled for benchmarking) ----------------------------
    if not t.checkpointing:
        add("--no-save-optim")
        add("--no-save-rng")
        add("--no-load-optim")
        add("--no-load-rng")

    return args
