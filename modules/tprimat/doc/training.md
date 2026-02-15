# Training Configuration

## 1. Training Algorithm (apple-to-apple)

| Parameter                  | NVIDIA           | AMD              |
| -------------------------- | ---------------- | ---------------- |
| Seed                       | 42               | 42               |
| Micro Batch Size (MBS)     | 2                | 2                |
| Sequence Length (SL)       | 2048             | 2048             |
| Gradient Accumulation (GA) | 4                | 4                |
| Global Batch Size (GBS)    | 64               | 64               |
| Peak Learning Rate         | 3e-4             | 3e-4             |
| LR Decay                   | cosine, min_lr=0 | cosine, min_lr=0 |
| LR Warmup Steps            | 50               | 50               |
| Train Iters                | 500              | 500              |
| Weight Decay               | 0.1              | 0.1              |
| Adam Beta1 / Beta2         | 0.9 / 0.95       | 0.9 / 0.95       |
| init_method_std            | 0.02             | 0.02             |
| Gradient Checkpointing     | Disabled         | Disabled         |
| random_data_seed           | 42               | 42               |
| Data Split                 | 100,0,0          | 100,0,0          |
| Dataset                    | BookCorpus (bc)  | BookCorpus (bc)  |
| tokenizer_type             | HFTokenizer      | HFTokenizer      |
| Checkpointing/Logging      | Disabled         | Disabled         |
| Number of Samples          | 50000            | 50000            |

## 2. Parallelism and Precision (apple-to-apple)

| Parameter                    | NVIDIA | AMD   |
| ---------------------------- | ------ | ----- |
| tensor_model_parallel_size   | 1      | 1     |
| pipeline_model_parallel_size | 1      | 1     |
| data_parallel_size           | 8      | 8     |
| gradient_accumulation        | 8      | 8     |
| sequence_parallel            | False  | False |
| context_parallel_size        | 1      | -     |
| precision                    | bf16   | bf16  |

## 3. Precision (apple-to-apple)

| Parameter                 | NVIDIA | AMD   |
| ------------------------- | ------ | ----- |
| hybrid_fp8                | False  | -     |
| fp32_residual_connection  | False  | False |
| Distributed Optimizer     | Yes    | Yes   |
| use_distributed_optimizer | True   | True  |
| grad_reduce_in_fp32       | False  | False |
| overlap_grad_reduce       | False  | -     |
| overlap_param_gather      | False  | -     |

## 4. Fused Ops and Flash Attn (best-vs-best)

| Parameter                 | NVIDIA | AMD  |
| ------------------------- | ------ | ---- |
| bias_activation_fusion    | True   | True |
| bias_dropout_fusion       | True   | True |
| masked_softmax_fusion     | True   | True |
| persist_layer_norm        | True   | True |
| apply_rope_fusion         | True   | True |
| cross_entropy_loss_fusion | True   | True |
| use_flash_attn            | True   | True |
| NVTE_FUSED_ATTN           | 1      | -    |
| NVTE_FLASH_ATTN           | 1      | -    |
| enable_primus_turbo       | -      | True |
| use_turbo_attention       | -      | True |