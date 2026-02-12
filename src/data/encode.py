#!/usr/bin/env python3
import argparse
import json
import os
import struct
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

DATA_DIR = os.environ.get("DATA_DIR", "/data/tprimat")
DATA_SAMPLES = int(os.environ.get("DATA_SAMPLES", 500000))
TRAIN_SPLIT = float(os.environ.get("TRAIN_SPLIT", 0.9))  # default if env not set

DTYPE = np.dtype(np.int32)
DTYPE_CODE = 4


def write_megatron_dataset(output_prefix: str, tokens_array: np.ndarray, seq_length: int):
    """Write tokens to Megatron indexed dataset format (.bin and .idx files)."""
    num_sequences = len(tokens_array) // seq_length
    if num_sequences == 0:
        raise ValueError(f"Not enough tokens ({len(tokens_array)}) for seq_length {seq_length}")
    
    tokens_used = num_sequences * seq_length
    tokens_array = tokens_array[:tokens_used]
    
    output_path = Path(output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    bin_path = Path(f"{output_prefix}.bin")
    idx_path = Path(f"{output_prefix}.idx")
    
    with open(bin_path, "wb") as f:
        tokens_array.tofile(f)
    
    bytes_per_seq = seq_length * DTYPE.itemsize
    
    # Megatron IndexedDataset format:
    # 1. sequence_lengths (int32 array of length num_sequences)
    # 2. sequence_pointers (int64 array of length num_sequences)  
    # 3. document_indices (int64 array of length num_documents + 1)
    num_documents = num_sequences
    sequence_lengths = np.full(num_sequences, seq_length, dtype=np.int32)
    sequence_pointers = np.arange(num_sequences, dtype=np.int64) * bytes_per_seq
    document_indices = np.arange(num_documents + 1, dtype=np.int64)
    
    assert sequence_lengths.shape[0] == document_indices[-1], \
        f"Format error: {sequence_lengths.shape[0]} != {document_indices[-1]}"
    
    with open(idx_path, "wb") as f:
        f.write(b'MMIDIDX\x00\x00')  # 9-byte header
        f.write(struct.pack('<Q', 1))  # version
        f.write(struct.pack('<B', DTYPE_CODE))  # dtype code
        f.write(struct.pack('<Q', len(sequence_lengths)))  # sequence_count
        f.write(struct.pack('<Q', len(document_indices)))  # document_count
        f.write(sequence_lengths.tobytes())
        f.write(sequence_pointers.tobytes())
        f.write(document_indices.tobytes())
    
    return num_sequences


def tokenize_documents(input_file: str, tokenizer_name: str, max_samples: int = None):
    """Tokenize documents from JSONL file and return all tokens."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    vocab_size = len(tokenizer)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError(f"{tokenizer_name} has no eos_token_id configured")
    
    all_tokens = []
    doc_count = 0
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and doc_count >= max_samples:
                break
            
            line = line.strip()
            if not line:
                continue
            
            doc = json.loads(line)
            text = doc.get("text", "")
            doc_count += 1
            
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if tokens:
                max_token = max(tokens)
                min_token = min(tokens)
                if min_token < 0 or max_token >= vocab_size:
                    raise ValueError(
                        f"Token id out of range for tokenizer {tokenizer_name}: "
                        f"min={min_token} max={max_token} vocab={vocab_size}"
                    )
            all_tokens.extend(tokens)
            all_tokens.append(eos_token_id)
    
    print(f"Tokenized {doc_count} documents -> {len(all_tokens)} tokens")
    return np.array(all_tokens, dtype=DTYPE)


# Tokenizer paths: prefer local, fall back to HuggingFace
LOCAL_TOKENIZERS = {
    "llama": f"{DATA_DIR}/tokenizers/llama",
    "qwen": f"{DATA_DIR}/tokenizers/qwen",
}

HF_TOKENIZERS = {
    "llama": "meta-llama/Llama-3.1-8B",
    "qwen": "Qwen/Qwen2.5-7B",
}


def get_tokenizer_path(model_name: str) -> str:
    """Get tokenizer path, preferring local if it exists."""
    local_path = LOCAL_TOKENIZERS.get(model_name)
    if local_path and os.path.isdir(local_path):
        print(f"Using local tokenizer: {local_path}")
        return local_path
    hf_path = HF_TOKENIZERS.get(model_name)
    if hf_path:
        print(f"Using HuggingFace tokenizer: {hf_path}")
        return hf_path
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Encode JSONL to Megatron format with train/test split")
    parser.add_argument(
        "--input",
        type=str,
        default=f"{DATA_DIR}/pseudo-camera.jsonl",
        help="Input clean JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DATA_DIR,
        help="Output directory for encoded datasets",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="pc",
        help="Base name for output files (default: pc)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DATA_SAMPLES,
        help=f"Maximum number of documents to process (default: {DATA_SAMPLES})",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=TRAIN_SPLIT,
        help=f"Fraction for training data (default: {TRAIN_SPLIT})",
    )
    
    args = parser.parse_args()
    
    print(f"Processing up to {args.max_samples} documents from {args.input}")
    print(f"Train/test split: {args.train_split:.0%} / {1-args.train_split:.0%}")
    
    # Tokenize all documents (use llama tokenizer by default)
    tokenizer_path = get_tokenizer_path("llama")
    all_tokens = tokenize_documents(args.input, tokenizer_path, args.max_samples)
    
    # Calculate split point (align to sequence boundary)
    total_sequences = len(all_tokens) // args.seq_length
    train_sequences = int(total_sequences * args.train_split)
    test_sequences = total_sequences - train_sequences
    
    train_tokens = train_sequences * args.seq_length
    test_tokens = test_sequences * args.seq_length
    
    print(f"Total sequences: {total_sequences}")
    print(f"  Train: {train_sequences} sequences ({train_tokens} tokens)")
    print(f"  Test:  {test_sequences} sequences ({test_tokens} tokens)")
    
    # Split tokens
    train_array = all_tokens[:train_tokens]
    test_array = all_tokens[train_tokens:train_tokens + test_tokens]
    
    # Write train dataset
    train_prefix = f"{args.output_dir}/{args.output_name}-train"
    train_seqs = write_megatron_dataset(train_prefix, train_array, args.seq_length)
    print(f"Written train: {train_prefix}.{{bin,idx}} ({train_seqs} sequences)")
    
    # Write test dataset (used for both validation and test)
    test_prefix = f"{args.output_dir}/{args.output_name}-test"
    test_seqs = write_megatron_dataset(test_prefix, test_array, args.seq_length)
    print(f"Written test:  {test_prefix}.{{bin,idx}} ({test_seqs} sequences)")


if __name__ == "__main__":
    main()
