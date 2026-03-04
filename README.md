# Megatron binary data filter

Special-purpose tool for filtering Megatron-LM preprocessed data (.bin
and .idx) files. Given two preprocessed datasets, identifies documents
that appear only in one and writes a dataset consisting of those
examples. Membership testing is probabilistic and only guarantees that
the output does not contain any documents in the intersection, but may
not include all documents not in the intersection.

## Quickstart

The directory `data/json` contains small example datasets that overlap
in part, and `data/gpt2-bin` contains the same data preprocessed into
the Megatron binary format with the `gpt2` tokenizer. To test the
filtering with this data, run e.g.

```
python3 filter_bin.py \
    data/gpt2-bin/tiny.bin \
    data/gpt2-bin/tiny-subset.bin \
    filtered.bin
```

This should output in part `5 hits, 5 misses` and write the files
`filtered.bin` and `filtered.idx`. To check the contents of the
output, run

```
python3 decode_bin.py filtered gpt2
```

This should show the five texts that are in `data/json/tiny.jsonl`
but not in `data/json/tiny-subset.jsonl`.
