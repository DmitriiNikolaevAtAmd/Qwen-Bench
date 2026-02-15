#!/usr/bin/env python3
"""
Indexed Dataset Loader for datasets created by data/encode.py
Format: MMIDIDX header + sequence_count + document_count + seq_lengths + seq_pointers + doc_indices
"""
import struct
import numpy as np
import torch
from pathlib import Path


class IndexedDataset:
    """
    Loader for indexed datasets (binary format with .bin and .idx files)
    Compatible with datasets created by data/encode.py
    """

    def __init__(self, path):
        self.path = Path(path)
        self.bin_path = Path(str(path) + '.bin')
        self.idx_path = Path(str(path) + '.idx')

        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {self.bin_path}")
        if not self.idx_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.idx_path}")

        # Read index file
        with open(self.idx_path, 'rb') as f:
            # Read header: magic number (9 bytes)
            magic = f.read(9)
            assert magic == b'MMIDIDX\x00\x00', f"Invalid index file format: {magic}"

            # Read version (8 bytes)
            version = struct.unpack('<Q', f.read(8))[0]
            assert version == 1, f"Unsupported index version: {version}"

            # Read dtype code (1 byte)
            dtype_code = struct.unpack('<B', f.read(1))[0]
            self.dtype = self._code_to_dtype(dtype_code)

            # Read counts (order matches encode.py: sequence_count, document_count)
            self.num_seqs = struct.unpack('<Q', f.read(8))[0]
            self.num_docs = struct.unpack('<Q', f.read(8))[0]

            # Read arrays in order written by encode.py:
            # 1. sequence_lengths (int32 array of length num_seqs)
            # 2. sequence_pointers (int64 array of length num_seqs)
            # 3. document_indices (int64 array of length num_docs)
            self.seq_lengths = np.frombuffer(
                f.read(self.num_seqs * 4),
                dtype=np.int32
            ).copy()

            self.seq_pointers = np.frombuffer(
                f.read(self.num_seqs * 8),
                dtype=np.int64
            ).copy()

            self.doc_idx = np.frombuffer(
                f.read(self.num_docs * 8),
                dtype=np.int64
            ).copy()

        # Track binary file size for sanity checks
        self.bin_size = self.bin_path.stat().st_size
        self.dtype_size = np.dtype(self.dtype).itemsize

        # Don't open binary file here - will open lazily per process
        self._bin_file = None

    def _code_to_dtype(self, code):
        """Convert dtype code to numpy dtype"""
        dtype_map = {
            1: np.uint8,
            2: np.int8,
            3: np.int16,
            4: np.int32,
            5: np.int64,
            6: np.float32,
            7: np.float64,
            8: np.uint16,
        }
        return dtype_map.get(code, np.int64)

    @property
    def bin_file(self):
        """Lazy file handle - opens once per process"""
        if self._bin_file is None:
            self._bin_file = open(self.bin_path, 'rb', buffering=8192*1024)
        return self._bin_file

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx):
        """Get sequence at index"""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Get pointer (byte offset) and length (number of elements)
        pointer_bytes = int(self.seq_pointers[idx])
        num_elements = int(self.seq_lengths[idx])

        if num_elements <= 0:
            return torch.tensor([0], dtype=torch.long)

        bytes_to_read = num_elements * self.dtype_size

        # Validate read bounds
        if pointer_bytes < 0 or pointer_bytes + bytes_to_read > self.bin_size:
            raise IOError(
                f"Read out of bounds: offset={pointer_bytes}, "
                f"size={bytes_to_read}, file_size={self.bin_size}"
            )

        # Read data using pread for thread safety
        import os
        try:
            fd = self.bin_file.fileno()
            data = os.pread(fd, bytes_to_read, pointer_bytes)
        except (AttributeError, OSError):
            self.bin_file.seek(pointer_bytes, os.SEEK_SET)
            data = self.bin_file.read(bytes_to_read)

        if len(data) != bytes_to_read:
            raise IOError(f"Expected to read {bytes_to_read} bytes but got {len(data)}")

        tokens = np.frombuffer(data, dtype=self.dtype)
        return torch.from_numpy(tokens.astype(np.int64))

    def __del__(self):
        if hasattr(self, '_bin_file') and self._bin_file is not None:
            self._bin_file.close()

    def __getstate__(self):
        """Handle pickling for multiprocessing"""
        state = self.__dict__.copy()
        state['_bin_file'] = None
        return state

    def __setstate__(self, state):
        """Handle unpickling for multiprocessing"""
        self.__dict__.update(state)


def check_dataset_exists(path):
    """Check if indexed dataset files exist"""
    bin_path = Path(str(path) + '.bin')
    idx_path = Path(str(path) + '.idx')
    return bin_path.exists() and idx_path.exists()
