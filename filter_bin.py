#!/usr/bin/env python3

# Filter one Megatron-LM preprocessed dataset (.bin and .idx files)
# against another, removing documents that appear in both and writing
# a Megatron-LM dataset with those that only appear in the first.

import os
import sys
import struct
import hashlib
import logging

import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from argparse import ArgumentParser
from tqdm import tqdm

from rbloom import Bloom


# Megatron-LM indexed dataset `.idx` file header
_INDEX_HEADER = b'MMIDIDX\x00\x00'


# Megatron-LM indexed dataset `.idx` file data type mapping
DTYPE_MAP = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}

INVERSE_DTYPE_MAP = { v: k for k, v in DTYPE_MAP.items() }


def bin_and_idx_paths(bin_path):
    root, ext = os.path.splitext(bin_path)
    if ext != '.bin':
        raise ValueError(f'{bin_path} has extension {ext}, expected .bin')
    return bin_path, root+'.idx'


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('full', help='path to .bin with full data')
    ap.add_argument('subset', help='path to .bin with subset of data')
    ap.add_argument('out', help='path to .bin to output')
    ap.add_argument('--quiet', action='store_true')
    return ap.parse_args()


class IndexReader(object):
    """Reads Megatron-LM `.idx` file header and provides some additional
    information.

    Mostly a reduced and simplified version of Megatron _IndexReader
    (from megatron/core/datasets/indexed_dataset.py).

    Parameters
    ----------
    path : str
        Path to the `.idx` file.

    """
    def __init__(self, path: str):
        self.path = path

        logging.info(f'reading .idx {path}')
        with open(path, 'rb') as f:
            header = f.read(len(_INDEX_HEADER))
            assert header == _INDEX_HEADER, f'bad .idx header in {path}'

            version = struct.unpack('<Q', f.read(8))[0]
            assert version == 1, f'bad .idx version in {path}'

            code = struct.unpack('<B', f.read(1))[0]
            self.dtype = DTYPE_MAP[code]

            self.sequence_count = struct.unpack('<Q', f.read(8))[0]
            self.document_count = struct.unpack('<Q', f.read(8))[0]
            logging.info(f'sequence_count: {self.sequence_count}')
            logging.info(f'document_count: {self.document_count}')

            logging.info('reading sequence lengths')
            self.sequence_lengths = np.fromfile(f, np.int32, self.sequence_count)
            logging.info(f'sequence_lengths (shape {self.sequence_lengths.shape}): {self.sequence_lengths}')

            logging.info('reading sequence pointers')
            self.sequence_pointers = np.fromfile(f, np.int64, self.sequence_count)
            logging.info(f'sequence_pointers (shape {self.sequence_pointers.shape}): {self.sequence_pointers}')

            logging.info('reading document indices')
            self.document_indices = np.fromfile(f, np.int64, self.document_count)
            logging.info(f'document_indices (shape {self.document_indices.shape}): {self.document_indices}')


class BinIterator:
    def __init__(self, path, idx):
        self.path = path
        self.idx = idx

    def __iter__(self):
        with open(self.path, 'rb') as f:
            for i in range(self.idx.sequence_count):
                offset = self.idx.sequence_pointers[i]
                length = self.idx.sequence_lengths[i]
                f.seek(offset)
                yield np.fromfile(f, self.idx.dtype, length)


def hash_array(a):
    return hashlib.sha256(np.ascontiguousarray(a)).digest()


def main():
    args = parse_args()

    loglevel = logging.ERROR if args.quiet else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=loglevel)

    # Load index for subset data
    subset_bin_fn, subset_idx_fn = bin_and_idx_paths(args.subset)
    subset_idx = IndexReader(subset_idx_fn)

    # Store hashes in bloom filter
    bloom_filter = Bloom(subset_idx.sequence_count, 0.01)
    for data in tqdm(BinIterator(subset_bin_fn, subset_idx),
                     total=subset_idx.sequence_count,
                     desc="Hashing subset"):
        bloom_filter.add(hash_array(data))
    subset_idx = None

    # Load index of full data
    full_bin_fn, full_idx_fn = bin_and_idx_paths(args.full)
    full_idx = IndexReader(full_idx_fn)

    # Process full data documents, writing out ones not found in the
    # bloom filter and storing lengths for output index
    out_bin_fn, out_idx_fn = bin_and_idx_paths(args.out)
    out_sequence_lengths = []
    hits, misses = 0, 0
    with open(out_bin_fn, 'wb') as out_bin:
        for data in tqdm(BinIterator(full_bin_fn, full_idx),
                         total=full_idx.sequence_count,
                         desc="Processing full"):
            h = hash_array(data)
            if h in bloom_filter:
                hits += 1
            else:
                misses += 1
                data.tofile(out_bin)
                out_sequence_lengths.append(data.shape[0])
    logging.info(f'{hits} hits, {misses} misses')
    logging.info(f'wrote {out_bin_fn} with {len(out_sequence_lengths)} documents.')

    # write output index (following Megatron-LM _IndexWriter)
    with open(out_idx_fn, 'wb') as out_idx:
        out_idx.write(_INDEX_HEADER)
        out_idx.write(struct.pack('<Q', 1))
        out_idx.write(struct.pack('<B', INVERSE_DTYPE_MAP[full_idx.dtype]))

        sequence_pointers, curr_ptr = [], 0
        itemsize = np.dtype(full_idx.dtype).itemsize
        for length in out_sequence_lengths:
            sequence_pointers.append(curr_ptr)
            curr_ptr += length * itemsize

        document_indices = list(range(len(out_sequence_lengths)+1))

        sequence_count = len(out_sequence_lengths)
        out_idx.write(struct.pack('<Q', sequence_count))
        document_count = len(document_indices)
        out_idx.write(struct.pack('<Q', document_count))

        sequence_lengths = np.array(out_sequence_lengths, dtype=np.int32)
        out_idx.write(sequence_lengths.tobytes(order='C'))
        del out_sequence_lengths

        sequence_pointers = np.array(sequence_pointers, dtype=np.int64)
        out_idx.write(sequence_pointers.tobytes(order='C'))
        del sequence_pointers

        document_indices = np.array(document_indices, dtype=np.int64)
        out_idx.write(document_indices.tobytes(order='C'))
        del document_indices
    logging.info(f'wrote {out_idx_fn}.')
    logging.info('done.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
