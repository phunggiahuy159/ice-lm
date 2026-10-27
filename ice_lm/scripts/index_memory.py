#!/usr/bin/env python
"""Measured memory of the ICE-VR index on real code (CPU, no model): is "64.7 B/token" real?

    python scripts/index_memory.py --data_dir data --lengths 4096 16384 32768 --out index_memory.json

For each context length T (the first T held-out code tokens), three numbers:
  counted   ICEIndex.compact_bytes(): what benchmark.py reports (bytes of the compact layout, counted)
  numpy     that layout built for real and measured with .nbytes: token ids (uint16), per order n >= 2 an
            open-addressing hash table (int32 pairs (latest end position, previous one), load factor <= 0.5, the key
            is checked against the token array, so no key bytes), a direct int32 [vocab] x 2 table for n = 1, and
            the VR chain pointers (int32 per order per token). Every position's lookup is checked against the
            Python index, so the layout is complete, not just counted.
  python    tracemalloc peak of building the Python ICEIndex used in this repo (dicts and tuples, an
            implementation artefact)
Bytes per token are reported with and without the fixed vocab-sized order-1 table.
"""
import argparse
import json
import os
import sys
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from ice_lm.data import alpha_mask_for_data, load_meta, open_tokens  # noqa: E402
from ice_lm.ice_features import DEFAULT_ORDERS, segments_from_eos, set_alpha_mask  # noqa: E402
from ice_lm.inference import ICEIndex  # noqa: E402

M64 = (1 << 64) - 1


def ngram_hash(x, t, n):
    h = 1469598103934665603
    for v in x[t - n + 1: t + 1]:
        h = ((h ^ int(v)) * 1099511628211) & M64
    return h


def build_numpy(x, seg, orders, vocab):
    """The compact index as arrays; returns (dict of arrays, lookup(n, t) -> (latest, previous) end position)."""
    T = len(x)
    arrays = {"tokens": x.astype(np.uint16 if vocab <= 65536 else np.uint32)}
    tables = {}
    for n in orders:
        if n == 1:
            tab = np.full((vocab, 2), -1, dtype=np.int32)
            for t in range(T):
                tab[x[t]] = (t, tab[x[t], 0])
            tables[n] = tab
            continue
        keys = {tuple(x[t - n + 1: t + 1]) for t in range(n - 1, T) if seg[t - n + 1] == seg[t]}
        cap = 1 << max(4, int(np.ceil(np.log2(max(1, 2 * len(keys))))))      # load factor <= 0.5
        tab = np.full((cap, 2), -1, dtype=np.int32)
        for t in range(n - 1, T):
            if seg[t - n + 1] != seg[t]:
                continue
            i = ngram_hash(x, t, n) & (cap - 1)
            while tab[i, 0] >= 0 and not np.array_equal(x[tab[i, 0] - n + 1: tab[i, 0] + 1], x[t - n + 1: t + 1]):
                i = (i + 1) & (cap - 1)                                     # linear probing; key = the n-gram
            tab[i] = (t, tab[i, 0])
        tables[n] = tab
    for n, tab in tables.items():
        arrays[f"table_{n}"] = tab
    arrays["chain_prev"] = np.full((len(orders), T), -1, dtype=np.int32)    # VR: previous occurrence per order

    def lookup(n, t):
        tab = tables[n]
        if n == 1:
            return tuple(int(v) for v in tab[x[t]])
        cap = len(tab)
        i = ngram_hash(x, t, n) & (cap - 1)
        while tab[i, 0] >= 0:
            if np.array_equal(x[tab[i, 0] - n + 1: tab[i, 0] + 1], x[t - n + 1: t + 1]):
                return int(tab[i, 0]), int(tab[i, 1])
            i = (i + 1) & (cap - 1)
        return None
    return arrays, lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 32768])
    ap.add_argument("--out", default="index_memory.json")
    a = ap.parse_args()
    meta = load_meta(a.data_dir)
    vocab, eos = meta["vocab_size"], meta["eos_id"]
    set_alpha_mask(alpha_mask_for_data(a.data_dir))
    code = open_tokens(a.data_dir, "val_code")
    rows = []
    for T in a.lengths:
        x = np.asarray(code[:T]).astype(np.int64)
        seg = segments_from_eos(x, eos)
        tracemalloc.start()
        idx = ICEIndex(DEFAULT_ORDERS, eos, vocab, "vr")
        idx.prefill(x.tolist())
        _, py_peak = tracemalloc.get_traced_memory()
        py_now = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        counted = idx.compact_bytes()
        arrays, lookup = build_numpy(x, seg, DEFAULT_ORDERS, vocab)
        for j in range(len(DEFAULT_ORDERS)):
            arrays["chain_prev"][j] = np.asarray(idx.prev[j], dtype=np.int32)   # the VR chain, as stored
        bad = 0
        for j, n in enumerate(DEFAULT_ORDERS):                  # the arrays reproduce the Python tables exactly
            for key, (last, prev) in idx.tables[j].items():
                got = lookup(n, last)
                bad += got is None or got[0] != last
        nb = {k: int(v.nbytes) for k, v in arrays.items()}
        fixed = nb["table_1"]
        total = sum(nb.values())
        r = dict(T=T, distinct=idx.distinct_ngrams(), counted_total=counted["total"],
                 counted_per_token=(counted["total"] - vocab * 8) / T, numpy_bytes=nb, numpy_total=total,
                 numpy_per_token=(total - fixed) / T, numpy_per_token_incl_fixed=total / T,
                 python_bytes_now=py_now, python_peak=py_peak, python_per_token=py_now / T, lookup_mismatches=bad)
        rows.append(r)
        print(f"T={T:6d}  counted {r['counted_per_token']:6.1f} B/tok   numpy {r['numpy_per_token']:6.1f} B/tok "
              f"(+{fixed / 2**10:.0f} KiB fixed; {total / 2**20:.2f} MiB total)   python {r['python_per_token']:7.0f} B/tok"
              f"   lookup mismatches {bad}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
