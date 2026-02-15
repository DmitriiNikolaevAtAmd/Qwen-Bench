"""Energon/WebDataset GPT dataset for Megatron-Core pretraining.

Reads caption text from WebDataset .tar shards, tokenizes into a
continuous token stream, and serves fixed-length sequences for causal
language modelling.
"""
import glob
import logging
import os
import tarfile

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EnergonGPTDataset(Dataset):
    """GPT pretraining dataset backed by Energon/WebDataset shards.

    Reads ``.txt`` entries from ``.tar`` shards in the ``train/``
    subdirectory, tokenizes them into a single token stream separated
    by end-of-document tokens, then slices the stream into fixed-length
    samples of ``seq_length + 1`` (the extra token provides the label
    for the last position).
    """

    def __init__(self, data_path: str, tokenizer, seq_length: int) -> None:
        train_dir = os.path.join(data_path, "train")
        shard_files = sorted(glob.glob(os.path.join(train_dir, "*.tar")))

        if not shard_files:
            raise FileNotFoundError(
                f"No .tar shards found in {train_dir}. "
                "Run 'stage=data' first to create WebDataset shards."
            )

        # Tokenize all captions into a continuous token stream
        eod_id = tokenizer.eod
        all_tokens: list[int] = []

        for shard_file in shard_files:
            with tarfile.open(shard_file, "r") as tar:
                for member in sorted(tar.getmembers(), key=lambda m: m.name):
                    if member.name.endswith(".txt"):
                        f = tar.extractfile(member)
                        if f is not None:
                            text = f.read().decode("utf-8").strip()
                            if text:
                                tokens = tokenizer.tokenize(text)
                                all_tokens.extend(tokens)
                                all_tokens.append(eod_id)

        # Slice into fixed-length samples (+1 for the labels shift)
        sample_len = seq_length + 1
        num_samples = len(all_tokens) // sample_len

        if num_samples == 0:
            raise ValueError(
                f"Not enough tokens ({len(all_tokens):,}) for even one "
                f"sample of length {sample_len:,}. Need more data or a "
                f"shorter seq_length."
            )

        self.data = np.array(
            all_tokens[: num_samples * sample_len], dtype=np.int64,
        ).reshape(num_samples, sample_len)
        self.seq_length = seq_length

        logger.info(
            "EnergonGPTDataset: %d shards, %d tokens, %d samples "
            "(seq_length=%d)",
            len(shard_files), len(all_tokens), num_samples, seq_length,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        tokens = torch.tensor(sample[:-1], dtype=torch.long)
        labels = torch.tensor(sample[1:], dtype=torch.long)
        loss_mask = torch.ones(self.seq_length, dtype=torch.float32)

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
        }
