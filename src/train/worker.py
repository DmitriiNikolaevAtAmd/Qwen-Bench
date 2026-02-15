"""Megatron-Core GPT pretrain worker.

Launched via torchrun by the training stage orchestrator.
All configuration is passed as Megatron-style command-line arguments.
"""
import gc
from functools import partial

import torch


def model_provider(pre_process=True, post_process=True, **kwargs):
    """Build GPTModel from Megatron args.

    Latest Megatron-LM passes ``config`` as a keyword argument built from
    the global args.  When provided we use it directly; otherwise we fall
    back to constructing one ourselves (for older Megatron versions).
    """
    from megatron.training import get_args
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    args = get_args()

    config = kwargs.get("config")
    if config is None:
        config = TransformerConfig(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            ffn_hidden_size=args.ffn_hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_query_groups=getattr(
                args, "num_query_groups", args.num_attention_heads
            ),
            use_cpu_initialization=True,
            init_method_std=args.init_method_std,
            bf16=args.bf16,
            fp16=args.fp16,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            sequence_parallel=args.sequence_parallel,
            fp32_residual_connection=args.fp32_residual_connection,
            recompute_granularity=args.recompute_granularity,
            recompute_method=args.recompute_method,
            recompute_num_layers=args.recompute_num_layers,
        )

    transformer_layer_spec = get_gpt_layer_local_spec()

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=True,
    )

    return model


def get_batch(data_iterator):
    """Extract batch from data iterator."""
    data = next(data_iterator)

    tokens = data["tokens"].long().cuda()
    labels = data["labels"].long().cuda()
    loss_mask = data["loss_mask"].float().cuda()
    attention_mask = data.get("attention_mask")
    position_ids = data.get("position_ids")

    if attention_mask is not None:
        attention_mask = attention_mask.cuda()
    if position_ids is not None:
        position_ids = position_ids.long().cuda()

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    """Compute cross-entropy loss with masking."""
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return total_loss, {"lm loss": total_loss}


def forward_step(data_iterator, model):
    """Forward pass: tokens -> model -> loss."""
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator
    )
    output_tensor = model(
        tokens, position_ids, attention_mask, labels=labels
    )
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build GPT datasets from Megatron binary (.bin/.idx) files."""
    from megatron.training import get_args, get_tokenizer
    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import (
        GPTDataset,
        GPTDatasetConfig,
    )

    from megatron.core import parallel_state as mpu

    args = get_args()
    tokenizer = get_tokenizer()

    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=(args.data_path, None),
        split=args.split,
        path_to_cache=args.data_cache_path,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        data_parallel_size=mpu.get_data_parallel_world_size(),
    )

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        lambda: mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage(),
        config,
    ).build()

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    gc.disable()

    # On ROCm (no CUDA runtime), skip legacy fused-kernel compilation
    # which crashes because CUDA_HOME / nvcc are absent.
    import torch
    if not torch.version.cuda:
        import megatron.legacy.fused_kernels as _fk
        _fk.load = lambda args: None

    from megatron.training import pretrain
    from megatron.core.enums import ModelType

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={
            "tokenizer_type": "HuggingFaceTokenizer",
            "no_save_optim": True,
            "no_save_rng": True,
            "no_load_optim": True,
            "no_load_rng": True,
        },
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
