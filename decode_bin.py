#!/usr/bin/env python3

# Mostly adapted from parts of
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py and
# https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/indexed_dataset.py

import sys
import struct

import numpy as np

from argparse import ArgumentParser

from transformers import AutoTokenizer


_INDEX_HEADER = b"MMIDIDX\x00\x00"


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


def argparser():
    ap = ArgumentParser()
    ap.add_argument('path', help='data path without suffix (.idx/.bin)')
    ap.add_argument('tokenizer')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    idxfn, binfn = f'{args.path}.idx', f'{args.path}.bin'

    # Load index
    with open(idxfn, 'rb') as f:
        header = f.read(len(_INDEX_HEADER))
        assert header == _INDEX_HEADER
        version = struct.unpack('<Q', f.read(8))
        assert version == (1,)

        code = struct.unpack('<B', f.read(1))[0]
        dtype = DTYPE_MAP[code]

        sequence_count = struct.unpack("<Q", f.read(8))[0]
        document_count = struct.unpack("<Q", f.read(8))[0]

        sequence_lengths = np.fromfile(f, np.int32, sequence_count)
        sequence_pointers = np.fromfile(f, np.int64, sequence_count)
        document_indices = np.fromfile(f, np.int64, document_count)

    # Load and print decoded .bin records
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(binfn, 'rb') as f:
        for i in range(sequence_count):
            offset, length = sequence_pointers[i], sequence_lengths[i]
            f.seek(offset)
            data = np.fromfile(f, dtype, length)
            print('-'* 30, i, '-'*30)
            print(tokenizer.decode(data))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
