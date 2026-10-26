#!/usr/bin/env python
"""D3 coverage gate (CPU only, no model): how often is the correct next token inside a candidate set?

    python scripts/coverage_gate.py --data_dir /vol/data --out /vol/coverage_gate.json

For every scored position t (which predicts y = x[t+1]):
  s*  current ICE prediction: successor of the longest match (ice_s_star of compute_ice_features)
  V   12 slots: orders (1,2,3,4,6,8) x {last, second} earlier occurrence of the order-n key ending at t,
      candidate = successor x[p+1]
  VR  V + 22 slots:
      chain   occurrences 3..8 of the longest-matching order (same key, following the per-order prev pointers)
      anchors the 2 most recent alphabetic tokens at a in x[t-7..t]; for each, the latest earlier occurrence q under
              the order-1 key (x[a]) and the order-2 key (x[a-1], x[a]); candidates x[q+1..q+4], every slot with
              source index q+i > t masked
"Alphabetic" is a vocabulary mask: the token piece without the SentencePiece "▁" matches [A-Za-z_][A-Za-z0-9_]*
and contains a letter (identifier-like; digits are separate tokens in the Llama tokenizer).
Matches never cross a document (EOS) boundary. Each source index is asserted to be <= t.

Data:
  CodeRecall  all categories, gaps 256/1024/3072, n=100 per (category, gap), exactly the items run_coderecall builds
              (same seeds, same tokenizer); scored position = the deciding token.
  real code   --windows windows of 4096 tokens from val_code (sequential, packed, EOS-separated like
              eval_loss_splits); every position scored, also split by the s* status (right / wrong / none) and
              copyable (longest match >= 4).
Gate (decision.md, D3): train VR only if VR >= V + 0.30 on at least 2 of {override_diff, copy_diff, conflict_g5}.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from ice_lm.coderecall import CATEGORIES, build_items  # noqa: E402
from ice_lm.data import load_meta, open_tokens  # noqa: E402
from ice_lm.ice_features import (DEFAULT_ORDERS, _P, _prev_occurrence, compute_ice_features,  # noqa: E402
                                 segments_from_eos)

GATE_CATS = ("override_diff", "copy_diff", "conflict_g5")
N_CHAIN = 8              # occurrences 1..8 of each key are followed; V uses 1-2, the deep chain 3-8
N_ANCHORS, ANCHOR_WIN, SPAN = 2, 8, 4


def key_prev(x: np.ndarray, seg: np.ndarray, n: int) -> np.ndarray:
    """prev[t] = most recent p < t whose n tokens ending at p equal those ending at t (same document), else -1.
    Same hash as compute_ice_features, plus an exact check (a collision drops the pointer)."""
    T = len(x)
    prev = np.full(T, -1, dtype=np.int64)
    if n > T:
        return prev
    xu = x.astype(np.uint64) + np.uint64(1)
    su = seg.astype(np.uint64) + np.uint64(1)
    h = np.zeros(T, dtype=np.uint64)
    for i in range(n):
        sh = np.zeros(T, dtype=np.uint64)
        sh[i:] = xu[: T - i]
        h = h * _P + sh
    h = h * _P + su
    valid = np.zeros(T, dtype=bool)
    valid[n - 1:] = seg[n - 1:] == seg[: T - n + 1]
    p = _prev_occurrence(h, valid)
    has = np.nonzero(p >= 0)[0]
    ok = np.ones(len(has), dtype=bool)
    for i in range(n):
        ok &= x[has - i] == x[p[has] - i]
    ok &= seg[p[has]] == seg[has]
    prev[has[ok]] = p[has[ok]]
    return prev


def candidate_sets(x: np.ndarray, seg: np.ndarray, alpha: np.ndarray, orders=DEFAULT_ORDERS):
    """-> dict of candidate arrays [K, T] (token id, -1 = empty slot): 'V' (12), 'chain' (6), 'anchor' (16),
    plus s_star [T] and match_len [T] from compute_ice_features (the current ICE)."""
    x = np.asarray(x, dtype=np.int64)
    T = len(x)
    t_idx = np.arange(T)
    prevs = {n: key_prev(x, seg, n) for n in orders}
    occ = np.full((len(orders), N_CHAIN, T), -1, dtype=np.int64)   # occurrence index of the key ending at t
    for j, n in enumerate(orders):
        pr = prevs[n]
        cur = pr.copy()
        for k in range(N_CHAIN):
            occ[j, k] = cur
            nxt = np.full(T, -1, dtype=np.int64)
            m = cur >= 0
            nxt[m] = pr[cur[m]]
            cur = nxt

    def succ_of(src):                       # src: occurrence index (-1 = none) -> successor token
        out = np.full(src.shape, -1, dtype=np.int64)
        m = src >= 0
        assert np.all(src[m] + 1 <= np.broadcast_to(t_idx, src.shape)[m]), "source index beyond t"
        out[m] = x[src[m] + 1]
        return out

    V = succ_of(occ[:, :2].reshape(-1, T))                          # [12, T]: order-major, (last, second)

    # longest-matching order = highest order with a match (its most recent occurrence has the longest true match)
    has = occ[:, 0] >= 0                                            # [N, T]
    best = np.where(has.any(0), len(orders) - 1 - np.argmax(has[::-1], axis=0), -1)
    chain_src = np.full((N_CHAIN - 2, T), -1, dtype=np.int64)
    m = best >= 0
    chain_src[:, m] = occ[best[m], 2:, t_idx[m]].T
    chain = succ_of(chain_src)                                      # [6, T]

    # anchors: 2 most recent alphabetic tokens in x[t-7..t]
    is_a = alpha[x]
    anc = np.full((N_ANCHORS, T), -1, dtype=np.int64)
    filled = np.zeros(T, dtype=np.int64)
    for d in range(ANCHOR_WIN):                                     # a = t - d, most recent first
        a = t_idx - d
        ok = (a >= 0) & (filled < N_ANCHORS)
        ok[ok] &= is_a[a[ok]] & (seg[a[ok]] == seg[t_idx[ok]])
        for r in range(N_ANCHORS):
            sel = ok & (filled == r)
            anc[r, sel] = a[sel]
        filled += ok
    p1, p2 = prevs[1], prevs.get(2, key_prev(x, seg, 2))
    anchor_src = np.full((N_ANCHORS, 2, SPAN, T), -1, dtype=np.int64)
    for r in range(N_ANCHORS):
        a = anc[r]
        for kk, pr in enumerate((p1, p2)):
            q = np.full(T, -1, dtype=np.int64)
            ma = a >= 0
            q[ma] = pr[a[ma]]
            for i in range(SPAN):
                src = np.where(q >= 0, q + i, -1)                  # candidate x[q+1+i] has source index q+i
                src = np.where((src >= 0) & (src + 1 <= t_idx), src, -1)   # mask a candidate from beyond t
                if i > 0:   # the span stays inside the document of q
                    src = np.where((src >= 0) & (seg[np.clip(src + 1, 0, T - 1)] == seg[np.clip(q, 0, None)]), src, -1)
                anchor_src[r, kk, i] = src
    anchor = succ_of(anchor_src.reshape(-1, T))                    # [16, T]

    f = compute_ice_features(x, seg, orders)
    # sanity: V's 'last' slots are exactly the current ICE successors; s* is the best order's last successor
    assert np.array_equal(V[0::2], f["ice_succ"]), "V(last) != ice_succ"
    s_best = np.where(best >= 0, V[0::2][np.clip(best, 0, None), t_idx], -1)
    assert np.array_equal(s_best, f["ice_s_star"]), "longest-order successor != s*"
    return dict(V=V, chain=chain, anchor=anchor, s_star=f["ice_s_star"], match_len=f["ice_match_len"])


def n_distinct(c: np.ndarray) -> np.ndarray:
    """c [K, P] -> number of distinct valid ids per column."""
    s = np.sort(c, axis=0)
    new = np.concatenate([s[:1] >= 0, (s[1:] != s[:-1]) & (s[1:] >= 0)], axis=0)
    return new.sum(0)


DIGIT = None   # [vocab] bool: single-digit tokens (set in main); CodeRecall deciding tokens are digits


def score(cs, pos: np.ndarray, y: np.ndarray):
    """Per scored position: covered by s* / V / VR (0/1) and candidate counts (all, and digit tokens only: a
    deciding digit is one of 9, so a set holding many digits covers it more easily by chance)."""
    V = cs["V"][:, pos]
    VR = np.concatenate([V, cs["chain"][:, pos], cs["anchor"][:, pos]], 0)

    def dig(c):
        return np.where((c >= 0) & DIGIT[np.clip(c, 0, None)], c, -1)
    return dict(s_star=(cs["s_star"][pos] == y), V=(V == y).any(0), VR=(VR == y).any(0),
                n_V=n_distinct(V), n_VR=n_distinct(VR), dig_V=n_distinct(dig(V)), dig_VR=n_distinct(dig(VR)),
                VR_chain=(np.concatenate([V, cs["chain"][:, pos]], 0) == y).any(0),
                VR_anchor=(np.concatenate([V, cs["anchor"][:, pos]], 0) == y).any(0))


def summarize(rows):
    keys = ("s_star", "V", "VR", "VR_chain", "VR_anchor", "n_V", "n_VR", "dig_V", "dig_VR")
    cat = {k: np.concatenate([r[k] for r in rows]).astype(float) for k in keys}
    out = {k: float(v.mean()) if len(v) else float("nan") for k, v in cat.items()}
    out["n"] = int(len(cat["V"]))
    out["VR_minus_V"] = out["VR"] - out["V"]
    return out


def alpha_mask(tok, V: int) -> np.ndarray:
    pieces = tok.convert_ids_to_tokens(list(range(V)))
    rx = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    m = np.zeros(V, dtype=bool)
    for i, p in enumerate(pieces):
        s = (p or "").replace("▁", "")
        m[i] = bool(s) and rx.fullmatch(s) is not None and any(ch.isalpha() for ch in s)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--gaps", type=int, nargs="+", default=[256, 1024, 3072])
    ap.add_argument("--n", type=int, default=100, help="CodeRecall items per category and gap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=int, default=300, help="held-out code windows")
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--out", default="coverage_gate.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    meta = load_meta(a.data_dir)
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
    eos = meta["eos_id"]
    alpha = alpha_mask(tok, max(meta["vocab_size"], len(tok)))
    global DIGIT
    pieces = tok.convert_ids_to_tokens(list(range(len(alpha))))
    DIGIT = np.array([(p or "").replace("\u2581", "") in set("0123456789") for p in pieces])
    print(f"alphabetic tokens: {int(alpha.sum())} / {len(alpha)}")
    filler = open_tokens(a.data_dir, "val_code")
    t0 = time.time()

    res = {"coderecall": {}, "coderecall_by_gap": {}, "code": {}, "config": vars(a)}
    for cat in CATEGORIES:
        rows = []
        for gap in (a.gaps if cat not in ("increment", "track") else a.gaps[:1]):
            rng = np.random.default_rng([a.seed, CATEGORIES.index(cat), gap])   # identical to run_coderecall
            seqs, spans, decide = build_items(tok, rng, cat, a.n, gap, filler, eos)
            grows = []
            for seq, t1 in zip(seqs, decide):
                x = np.asarray(seq[:-1], dtype=np.int64)            # what score_answers feeds the model
                cs = candidate_sets(x, np.zeros(len(x), dtype=np.int64), alpha)
                pos = np.array([t1 - 1])
                grows.append(score(cs, pos, np.array([seq[t1]])))
            res["coderecall_by_gap"][f"{cat}@{gap}"] = summarize(grows)
            rows += grows
        res["coderecall"][cat] = summarize(rows)
        print(f"[{time.time() - t0:5.0f}s] {cat}: {res['coderecall'][cat]}", flush=True)

    # held-out code: sequential packed windows, every position
    data = open_tokens(a.data_dir, "val_code")
    n_win = min(a.windows, (len(data) - 1) // a.seq_len)
    groups = {g: [] for g in ("all", "copyable", "lk_right", "lk_wrong", "lk_none")}
    for j in range(n_win):
        w = np.asarray(data[j * a.seq_len: j * a.seq_len + a.seq_len + 1]).astype(np.int64)
        x, y = w[:-1], w[1:]
        cs = candidate_sets(x, segments_from_eos(x, eos), alpha)
        pos = np.arange(len(x))
        r = score(cs, pos, y)
        ss = cs["s_star"]
        masks = dict(all=np.ones(len(x), bool), copyable=cs["match_len"] >= 4, lk_right=ss == y,
                     lk_wrong=(ss >= 0) & (ss != y), lk_none=ss < 0)
        for g, m in masks.items():
            groups[g].append({k: v[m] for k, v in r.items()})
    for g, rows in groups.items():
        res["code"][g] = summarize(rows)
    res["code_windows"] = n_win
    res["code_tokens"] = n_win * a.seq_len
    print(f"[{time.time() - t0:5.0f}s] code: {n_win} windows")

    # gate
    passed = [c for c in GATE_CATS if res["coderecall"][c]["VR_minus_V"] >= 0.30]
    res["gate"] = dict(categories=list(GATE_CATS), threshold=0.30, passed_on=passed, passed=len(passed) >= 2)

    hdr = f"{'set':22s}{'n':>6s}{'s*':>8s}{'V':>8s}{'VR':>8s}{'VR-V':>8s}{'V+chn':>8s}{'V+anc':>8s}{'|V|':>7s}{'|VR|':>7s}{'dgV':>6s}{'dgVR':>6s}"
    print("\n== coverage of the correct deciding token (CodeRecall, pooled over gaps)\n" + hdr)

    def line(name, s):
        print(f"{name:22s}{s['n']:6d}{s['s_star']:8.3f}{s['V']:8.3f}{s['VR']:8.3f}{s['VR_minus_V']:+8.3f}"
              f"{s['VR_chain']:8.3f}{s['VR_anchor']:8.3f}{s['n_V']:7.2f}{s['n_VR']:7.2f}{s['dig_V']:6.2f}{s['dig_VR']:6.2f}")
    for cat, s in res["coderecall"].items():
        line(cat, s)
    print("\n== per gap\n" + hdr)
    for k, s in res["coderecall_by_gap"].items():
        line(k, s)
    print(f"\n== held-out code ({n_win} x {a.seq_len} tokens, every position, split by the s* status)\n" + hdr)
    for g, s in res["code"].items():
        line(g, s)
    print(f"\nGATE (VR >= V + 0.30 on >= 2 of {GATE_CATS}): passed on {passed} -> "
          f"{'PASS' if res['gate']['passed'] else 'FAIL'}")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
