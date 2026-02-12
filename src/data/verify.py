#!/usr/bin/env python3
import argparse
import os
import struct
import random
import sys
import numpy as np
from pathlib import Path

DATA_DIR = os.environ.get("DATA_DIR", "/data/tprimat")

DTYPE = np.dtype(np.int32)
DTYPE_CODE = 4


def verify_dataset(
    input_prefix: str,
    tokenizer_name: str,
    num_samples: int,
    full_scan: bool = False,
) -> bool:
    bin_path = Path(f"{input_prefix}.bin")
    idx_path = Path(f"{input_prefix}.idx")
    is_nemo_format = "-nemo" in input_prefix
    
    errors = []
    warnings = []
    
    print(f"Verifying {input_prefix}...")
    print(f"  Format: {'NeMo' if is_nemo_format else 'Mega'}")
    
    if not bin_path.exists():
        print(f"FAIL: Binary file not found: {bin_path}")
        return False
    if not idx_path.exists():
        print(f"FAIL: Index file not found: {idx_path}")
        return False
    
    print(f"\n[1/5] Checking index file...")
    try:
        with open(idx_path, 'rb') as f:
            # Megatron _INDEX_HEADER is exactly 9 bytes: b"MMIDIDX\x00\x00"
            magic = f.read(9)
            if magic != b'MMIDIDX\x00\x00':
                errors.append(f"Invalid magic: {magic}")
            
            version = struct.unpack('<Q', f.read(8))[0]
            if version != 1:
                errors.append(f"Unsupported version: {version}")
            
            dtype_code = struct.unpack('<B', f.read(1))[0]
            if dtype_code != DTYPE_CODE:
                warnings.append(f"Unexpected dtype code: {dtype_code} (expected {DTYPE_CODE})")
            
            num_seqs = struct.unpack('<Q', f.read(8))[0]
            num_docs = struct.unpack('<Q', f.read(8))[0]
            
            # num_docs should be num_seqs + 1 (len(document_indices) for N sequences)
            if num_docs != num_seqs + 1:
                warnings.append(f"num_docs ({num_docs}) != num_seqs + 1 ({num_seqs + 1})")
            
            # Megatron-compatible layout (matches Megatron _IndexReader):
            # 1. sequence_lengths (int32 * num_seqs)
            # 2. sequence_pointers (int64 * num_seqs)
            # 3. document_indices (int64 * num_docs) -- Megatron reads count=document_count
            lengths = np.frombuffer(f.read(num_seqs * 4), dtype=np.int32)
            pointers = np.frombuffer(f.read(num_seqs * 8), dtype=np.int64)
            doc_indices = np.frombuffer(f.read(num_docs * 8), dtype=np.int64)
            
            format_name = "NeMo" if is_nemo_format else "Mega"
            print(f"  {format_name} format (Megatron-compatible):")
            print(f"    - lengths array: {len(lengths)} elements")
            print(f"    - pointers array: {len(pointers)} elements")
            print(f"    - doc_indices array: {len(doc_indices)} elements")
            
            # Verify Megatron's critical assertion: sequence_lengths.shape[0] == document_indices[-1]
            if len(lengths) != doc_indices[-1]:
                errors.append(f"Megatron assertion FAILED: len(lengths)={len(lengths)} != doc_indices[-1]={doc_indices[-1]}")
            else:
                print(f"  Megatron assertion PASSED: len(lengths)={len(lengths)} == doc_indices[-1]={doc_indices[-1]}")
            
            expected_doc_indices = np.arange(num_docs, dtype=np.int64)
            if not np.array_equal(doc_indices, expected_doc_indices):
                errors.append(f"doc_indices malformed: expected [0..{num_docs-1}], got [{doc_indices[0]}..{doc_indices[-1]}]")
        
        print(f"  Sequences: {num_seqs:,}")
        print(f"  Seq length: {lengths[0] if len(lengths) > 0 else 'N/A'}")
        
    except Exception as e:
        errors.append(f"Failed to read index: {e}")
        print(f"FAIL: {e}")
        return False
    
    print(f"\n[2/5] Checking binary file...")
    bin_size = bin_path.stat().st_size
    seq_length = int(lengths[0]) if len(lengths) > 0 else 0
    bytes_per_seq = seq_length * DTYPE.itemsize
    expected_size = num_seqs * bytes_per_seq
    
    print(f"  Binary size: {bin_size:,} bytes ({bin_size / 1024 / 1024:.1f} MB)")
    
    if bin_size != expected_size:
        errors.append(f"Binary size {bin_size} != expected {expected_size}")
    
    if not np.all(lengths == seq_length):
        errors.append(f"Non-uniform sequence lengths found")
    
    print(f"\n[3/5] Checking pointers...")
    if np.any(pointers < 0):
        errors.append("Negative pointers found")
    if np.any(pointers >= bin_size):
        errors.append("Out-of-bounds pointers found")
    if np.any(pointers % DTYPE.itemsize != 0):
        errors.append("Unaligned pointers found")
    
    expected_pointers = np.arange(num_seqs, dtype=np.int64) * bytes_per_seq
    if not np.array_equal(pointers, expected_pointers):
        warnings.append("Non-sequential pointer layout")
    
    print(f"  Pointer range: [{pointers.min():,}, {pointers.max():,}]")
    
    print(f"\n[4/5] Validating tokens...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        vocab_size = len(tokenizer)
        print(f"  Tokenizer: {tokenizer_name}")
        print(f"  Vocab size: {vocab_size:,} (base: {tokenizer.vocab_size:,})")
    except Exception as e:
        warnings.append(f"Could not load tokenizer: {e}")
        vocab_size = None
    
    print(f"\n[5/5] Sampling {num_samples} sequences...")
    
    sample_indices = random.sample(range(num_seqs), min(num_samples, num_seqs))
    decode_errors = 0
    
    with open(bin_path, 'rb') as f:
        for i, idx in enumerate(sample_indices):
            offset = int(pointers[idx])
            f.seek(offset)
            data = f.read(bytes_per_seq)
            tokens = np.frombuffer(data, dtype=DTYPE)
            
            if len(tokens) != seq_length:
                errors.append(f"Sequence {idx}: got {len(tokens)} tokens, expected {seq_length}")
                continue
            
            if np.any(tokens < 0):
                errors.append(f"Sequence {idx}: negative tokens found")
            
            if vocab_size and np.any(tokens >= vocab_size):
                errors.append(f"Sequence {idx}: tokens exceed vocab size")
            
            if tokenizer:
                try:
                    text = tokenizer.decode(tokens[:50], skip_special_tokens=False)
                    if i < 3:
                        preview = text[:80].replace('\n', ' ')
                        print(f"  Sample {idx}: \"{preview}...\"")
                except Exception:
                    decode_errors += 1
    
    if decode_errors > 0:
        warnings.append(f"{decode_errors} sequences failed to decode")
    
    if full_scan:
        print(f"\n[6/6] Full token range scan...")
        try:
            total_tokens = bin_size // DTYPE.itemsize
            token_memmap = np.memmap(bin_path, dtype=DTYPE, mode="r", shape=(total_tokens,))
            min_token = int(token_memmap.min())
            max_token = int(token_memmap.max())
            print(f"  Token range: [{min_token}, {max_token}] over {total_tokens:,} tokens")
            if min_token < 0:
                errors.append(f"Full scan: negative token id found (min={min_token})")
            if vocab_size and max_token >= vocab_size:
                errors.append(
                    f"Full scan: token id exceeds vocab size (max={max_token}, vocab={vocab_size})"
                )
        except Exception as e:
            warnings.append(f"Full scan failed: {e}")

    print("\n" + "=" * 50)
    
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
    
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    
    if errors:
        print(f"\nFAIL: {len(errors)} error(s) found")
        return False
    else:
        print(f"\nPASS: Dataset verified OK")
        if warnings:
            print(f"  ({len(warnings)} warning(s))")
        return True


DATASETS = {
    "pc-train": ("pc-train", "meta-llama/Llama-3.1-8B"),
    "pc-test": ("pc-test", "meta-llama/Llama-3.1-8B"),
}


def main():
    parser = argparse.ArgumentParser(description="Verify encoded Megatron dataset")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DATA_DIR,
        help="Directory containing encoded datasets",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of random sequences to validate (default: 100)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Tokenizer to use for validation",
    )
    
    args = parser.parse_args()
    
    all_success = True
    for split_name, (dataset_name, default_tokenizer) in DATASETS.items():
        input_prefix = f"{args.input_dir}/{dataset_name}"
        
        # Check if dataset exists
        if not Path(f"{input_prefix}.idx").exists():
            print(f"\nSkipping {split_name}: {input_prefix}.idx not found")
            continue
        
        tokenizer = args.tokenizer or default_tokenizer
        print(f"\n{'='*60}")
        print(f"Verifying {split_name.upper()} dataset: {input_prefix}")
        print(f"{'='*60}")
        success = verify_dataset(input_prefix, tokenizer, args.samples, full_scan=True)
        if not success:
            all_success = False
    
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
