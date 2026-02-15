#!/usr/bin/env python3
"""
Cross-platform data pipeline diagnostic tool.

Reads Megatron-format binary dataset files (.bin/.idx) and prints token IDs
for the first N documents/sequences. Run on both AMD and NVIDIA platforms to
verify that the same data is being presented to the model.

Usage:
    python3 evaluate/fingerprint.py [--data-prefix /data/tprimat/bc-train] \
        [--seq-len 2048] [--num-samples 5] [--tokens-per-sample 32]

Output:
    Prints a fingerprint (hash) plus the first token IDs of each sample.
    If fingerprints match across platforms, data pipelines are equivalent.
"""
import argparse
import hashlib
import os
import struct
import sys
from pathlib import Path

import numpy as np


# Megatron indexed dataset dtypes (must match megatron/core/datasets/indexed_dataset.py)
DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,  # legacy, not used for tokens
    7: np.float64,
    8: np.uint16,
}


def read_index(idx_path: str):
    """Read a Megatron .idx index file and return (dtype, sizes, pointers)."""
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        assert magic == b"MMIDIDX\x00\x00", f"Bad magic in {idx_path}: {magic!r}"
        version = struct.unpack("<Q", f.read(8))[0]
        assert version == 1, f"Unsupported version {version}"
        dtype_code = struct.unpack("<B", f.read(1))[0]
        dtype = DTYPES[dtype_code]
        num_docs = struct.unpack("<Q", f.read(8))[0]
        # sizes: number of tokens per document
        sizes = np.frombuffer(f.read(num_docs * 4), dtype=np.int32)
        # pointers: byte offset into .bin for each document
        pointers = np.frombuffer(f.read(num_docs * 8), dtype=np.int64)
        # doc_idx: document indices (num_docs entries)
        doc_idx = np.frombuffer(f.read((num_docs + 1) * 8), dtype=np.int64)
    return dtype, sizes, pointers, doc_idx


def read_tokens(bin_path: str, dtype, pointer: int, length: int):
    """Read `length` tokens starting at byte `pointer` from the .bin file."""
    token_size = np.dtype(dtype).itemsize
    with open(bin_path, "rb") as f:
        f.seek(pointer)
        data = f.read(length * token_size)
    return np.frombuffer(data, dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description="Verify Megatron binary data pipeline")
    parser.add_argument(
        "--data-prefix",
        default=None,
        help="Data prefix (e.g. /data/tprimat/bc-train). "
             "Auto-detected from config.env if not provided.",
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to inspect")
    parser.add_argument("--tokens-per-sample", type=int, default=32, help="Tokens to print per sample")
    args = parser.parse_args()

    # Resolve data prefix
    data_prefix = args.data_prefix
    if data_prefix is None:
        # Try to read from config.env
        config_path = Path(__file__).parent.parent / "config.env"
        if config_path.exists():
            data_dir = None
            dataset = "bc"
            with open(config_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DATA_DIR="):
                        data_dir = line.split("=", 1)[1].strip()
                    elif line.startswith("DATASET="):
                        dataset = line.split("=", 1)[1].strip()
            if data_dir:
                data_prefix = os.path.join(data_dir, f"{dataset}-train")
    
    if not data_prefix:
        print("ERROR: Could not determine data prefix. Use --data-prefix.")
        sys.exit(1)

    idx_path = data_prefix + ".idx"
    bin_path = data_prefix + ".bin"

    for path in [idx_path, bin_path]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    print(f"Data prefix: {data_prefix}")
    print(f"Index file:  {idx_path} ({os.path.getsize(idx_path):,} bytes)")
    print(f"Binary file: {bin_path} ({os.path.getsize(bin_path):,} bytes)")
    print()

    # Read index
    dtype, sizes, pointers, doc_idx = read_index(idx_path)
    num_docs = len(sizes)
    total_tokens = int(sizes.sum())

    print(f"Dataset info:")
    print(f"  Documents:    {num_docs:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Token dtype:  {dtype.__name__}")
    print(f"  Avg doc len:  {total_tokens / num_docs:.1f} tokens")
    print(f"  Min doc len:  {sizes.min()}")
    print(f"  Max doc len:  {sizes.max()}")
    print()

    # Compute overall data fingerprint (hash of first 1MB of .bin)
    with open(bin_path, "rb") as f:
        chunk = f.read(1024 * 1024)
    data_hash = hashlib.sha256(chunk).hexdigest()[:16]
    print(f"Data fingerprint (SHA256 of first 1MB): {data_hash}")
    print()

    # Print first N samples
    n = min(args.num_samples, num_docs)
    k = args.tokens_per_sample

    print(f"First {n} documents (showing first {k} token IDs each):")
    print("-" * 72)

    sample_hashes = []
    for i in range(n):
        doc_len = int(sizes[i])
        tokens = read_tokens(bin_path, dtype, int(pointers[i]), min(doc_len, k))
        token_list = tokens.tolist()
        
        # Hash full document for verification
        full_tokens = read_tokens(bin_path, dtype, int(pointers[i]), doc_len)
        doc_hash = hashlib.md5(full_tokens.tobytes()).hexdigest()[:8]
        sample_hashes.append(doc_hash)

        print(f"  Doc {i:4d} | len={doc_len:6d} | hash={doc_hash} | tokens={token_list}")

    print("-" * 72)

    # Combined fingerprint of first N documents
    combined = ":".join(sample_hashes)
    combined_hash = hashlib.md5(combined.encode()).hexdigest()[:12]
    print(f"\nCombined fingerprint (first {n} docs): {combined_hash}")
    print()

    # Simulate Megatron sequence construction
    # Megatron concatenates documents and splits into seq_len chunks
    print(f"Simulated Megatron sequences (seq_len={args.seq_len}):")
    print("-" * 72)

    # Build a token stream from concatenated documents
    max_tokens = args.seq_len * args.num_samples * 2  # read enough for N sequences
    token_stream = []
    for i in range(min(num_docs, 200)):  # read up to 200 docs
        doc_len = int(sizes[i])
        tokens = read_tokens(bin_path, dtype, int(pointers[i]), doc_len)
        token_stream.extend(tokens.tolist())
        if len(token_stream) >= max_tokens:
            break

    # Split into sequences
    seq_hashes = []
    for s in range(min(args.num_samples, len(token_stream) // args.seq_len)):
        start = s * args.seq_len
        seq_tokens = token_stream[start : start + args.seq_len]
        seq_hash = hashlib.md5(bytes(str(seq_tokens), "utf-8")).hexdigest()[:8]
        seq_hashes.append(seq_hash)
        print(f"  Seq {s:3d} | hash={seq_hash} | first {k}: {seq_tokens[:k]}")

    print("-" * 72)
    if seq_hashes:
        seq_combined = ":".join(seq_hashes)
        seq_fp = hashlib.md5(seq_combined.encode()).hexdigest()[:12]
        print(f"\nSequence fingerprint (first {len(seq_hashes)} seqs): {seq_fp}")

    print()
    print("Compare these fingerprints across platforms.")
    print("If they match, both platforms see identical data.")


if __name__ == "__main__":
    main()
