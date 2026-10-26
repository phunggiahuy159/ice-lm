#!/usr/bin/env python
"""Day 0 (proposal, Step 0 ceiling): no model needed.
How often does an earlier n-gram match exist (coverage), and how often is its successor the true next
token (precision)? Reported per order and by longest-match length, separately for prose and code.
Also: is the lookup's precision predictable from the "trust" signals the model gets (agreement of the two
most recent occurrences, online reliability of the longest-match predictor)? If precision differs a lot
across their buckets, the model has something to learn from when deciding to ignore a lookup.

    python scripts/day0_stats.py --data_dir data --seq_len 4096 --windows 200
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from ice_lm.data import load_meta, open_tokens  # noqa: E402
from ice_lm.ice_features import DEFAULT_ORDERS, compute_ice_features, segments_from_eos  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="data")
ap.add_argument("--seq_len", type=int, default=4096)
ap.add_argument("--windows", type=int, default=200)
ap.add_argument("--out", default=None, help="optional JSON output path")
a = ap.parse_args()

meta = load_meta(a.data_dir)
LEN_BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 32)]
AGREE_NAMES = ["none", "single", "agree", "disagree"]
REL_NAMES = ["no history", "<20%", "20-50%", "50-80%", ">=80%"]
DIST_BINS = [(1, 64), (65, 512), (513, 2048), (2049, 1 << 30)]       # how far back the longest match is
results = {}
for split in ("val_prose", "val_code"):
    data = open_tokens(a.data_dir, split)
    n_win = min(a.windows, (len(data) - 1) // a.seq_len)
    order_hit = np.zeros(len(DEFAULT_ORDERS)); order_cov = np.zeros(len(DEFAULT_ORDERS))
    len_hit = np.zeros(len(LEN_BINS)); len_cnt = np.zeros(len(LEN_BINS))
    ag_hit = np.zeros(4); ag_cnt = np.zeros(4); rel_hit = np.zeros(5); rel_cnt = np.zeros(5)
    d_hit = np.zeros(len(DIST_BINS)); d_cnt = np.zeros(len(DIST_BINS))
    total = 0
    for j in range(n_win):
        w = np.asarray(data[j * a.seq_len: j * a.seq_len + a.seq_len + 1]).astype(np.int64)
        x, y = w[:-1], w[1:]
        f = compute_ice_features(x, segments_from_eos(x, meta["eos_id"]))
        total += len(x)
        for k in range(len(DEFAULT_ORDERS)):
            has = f["ice_succ"][k] >= 0
            order_cov[k] += has.sum()
            order_hit[k] += (f["ice_succ"][k][has] == y[has]).sum()
        ml = f["ice_match_len"]
        for b, (lo, hi) in enumerate(LEN_BINS):
            sel = (ml >= lo) & (ml <= hi)
            len_cnt[b] += sel.sum()
            len_hit[b] += (f["ice_s_star"][sel] == y[sel]).sum()
        has = f["ice_s_star"] >= 0
        right = (f["ice_s_star"] == y) & has
        for b in range(4):
            sel = has & (f["ice_agree_star"] == b)
            ag_cnt[b] += sel.sum(); ag_hit[b] += right[sel].sum()
        for b in range(5):
            sel = has & (f["ice_rel_star"] == b)
            rel_cnt[b] += sel.sum(); rel_hit[b] += right[sel].sum()
        for b, (lo, hi) in enumerate(DIST_BINS):
            sel = has & (f["ice_dist_star"] >= lo) & (f["ice_dist_star"] <= hi)
            d_cnt[b] += sel.sum(); d_hit[b] += right[sel].sum()
    res = dict(tokens=int(total),
               per_order={str(n): dict(coverage=order_cov[k] / total, precision=order_hit[k] / max(1, order_cov[k]))
                          for k, n in enumerate(DEFAULT_ORDERS)},
               longest_match={f"{lo}-{hi}": dict(coverage=len_cnt[b] / total, precision=len_hit[b] / max(1, len_cnt[b]))
                              for b, (lo, hi) in enumerate(LEN_BINS)})
    res["ice_correct_frac"] = float(len_hit.sum() / total)
    res["by_agree"] = {AGREE_NAMES[b]: dict(coverage=ag_cnt[b] / total, precision=ag_hit[b] / max(1, ag_cnt[b]))
                       for b in range(1, 4)}
    res["by_distance"] = {(f"{lo}-{hi}" if hi < 1 << 30 else f">{lo - 1}"):
                          dict(coverage=d_cnt[b] / total, precision=d_hit[b] / max(1, d_cnt[b]),
                               correct_frac=d_hit[b] / total)
                          for b, (lo, hi) in enumerate(DIST_BINS)}
    res["by_reliability"] = {REL_NAMES[b]: dict(coverage=rel_cnt[b] / total, precision=rel_hit[b] / max(1, rel_cnt[b]))
                             for b in range(5)}
    results[split] = res
    print(f"\n=== {split}: {total:,} tokens ===")
    print(" order  coverage  precision")
    for n, r in res["per_order"].items():
        print(f"  {n:>4}   {r['coverage']:7.1%}   {r['precision']:7.1%}")
    print(" longest match  coverage  precision")
    for k, r in res["longest_match"].items():
        print(f"  {k:>11}   {r['coverage']:7.1%}   {r['precision']:7.1%}")
    print(f" tokens predicted correctly by the longest-match successor: {res['ice_correct_frac']:.1%}")
    print(" trust signal of the longest match       coverage  precision")
    for k, r in res["by_agree"].items():
        print(f"  occurrences: {k:<24} {r['coverage']:7.1%}   {r['precision']:7.1%}")
    for k, r in res["by_reliability"].items():
        print(f"  recent hit rate: {k:<20} {r['coverage']:7.1%}   {r['precision']:7.1%}")
    print(" distance of the longest match           coverage  precision  tokens it gets right")
    for k, r in res["by_distance"].items():
        print(f"  {k:<38} {r['coverage']:7.1%}   {r['precision']:7.1%}   {r['correct_frac']:7.1%}")
if a.out:
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
