#!/usr/bin/env python
"""F1 (decision2.md section 3, CPU, no model): relational rules over the ICE-VR slots and per-group slot precision
as a function of the 0-byte V2 features Delta / n / r.

    python scripts/rule_oracle2.py --data_dir /vol/data --out /vol/diag/rule_oracle2.json [--mask data|coverage]

Slots: compute_ice_features(cand="vr") (K = 34; groups last 6 / second 6 / chain 6 / anchor 16), plus each slot's
source position i_k (c_k = x[i_k], i_k <= t), recomputed here by a copy of ice_features._candidates that also
returns it (asserted identical tok / len / dist). Features (decision2 section 2):
  Delta_k  clip(len_k - max{len_j : j valid OCCURRENCE slot (last/second/chain), c_j != c_k}, -8, 8);
           bucket Delta + 8 (0..16), 17 = no competitor
  n_k      clip(#{j valid : c_j = c_k}, 1, 6)
  r_k      recency rank of m(c_k) = max{i_j : c_j = c_k} among the distinct candidate tokens (newest = 0), clip 0..3
Rules (token picked at a position; only "ambiguous" positions, >= 2 distinct candidate tokens, are scored):
  s_star        current ICE prediction
  longest       longest exact match over all slots (rule_oracle.py); ties -> nearer distance bucket, slot order
  longest_occ   longest over occurrence slots only (anchor len is the anchor's match length); no occurrence slot
                -> falls back to `longest`
  newest        newest distinct token: argmax_k i_k ("latest write")
  newest_occ / newest_anc   the same over occurrence / anchor slots only (-1 if none)
  vote, lenvote as in rule_oracle.py (ties -> longest); oracle = target in the set
Sets: held-out code (val_code windows, all / s* wrong) and CodeRecall (deciding token, per category; items are
exactly run_coderecall's). Per-group precision P(c_k = y) of the valid slots at ambiguous positions vs Delta / n / r.

The module also holds the helpers shared with scripts/token_scorer.py (F2) and scripts/icek_probe.py (run 6):
vr_features, slot_features, code_doc_mask, train_code_windows.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ice_lm import ice_features as IF  # noqa: E402
from ice_lm.ice_features import (ANCHOR_ORDERS, ANCHOR_WINDOW, CHAIN_OCC, N_ANCHORS, SPAN,  # noqa: E402
                                 _dist_bucket, _match_len, compute_ice_features, get_alpha_mask, segments_from_eos,
                                 set_alpha_mask)

K = 34
GROUPS = ("last", "second", "chain", "anchor")
GID = np.array([0, 1] * 6 + [2] * len(CHAIN_OCC) + [3] * (N_ANCHORS * len(ANCHOR_ORDERS) * SPAN))
OCC = GID < 3
D_CLIP = 8
N_DELTA = 2 * D_CLIP + 2          # 0..16 = Delta + 8, 17 = no competitor
D_NONE = N_DELTA - 1
N_CNT = 6                         # n in 1..6 -> 0..5
N_REC = 4                         # r in 0..3
RULES = ("s_star", "longest", "longest_occ", "newest", "newest_occ", "newest_anc", "vote", "lenvote")
assert len(GID) == K


# ---------------------------------------------------------------------------- shared helpers
def _candidates_src(x, seg, cand, orders, prev_ok, len_raw, best, has_match, *_unused, **_unused_kw):
    """Copy of ice_features._candidates that also returns the source positions ice_v_src [K,T] (-1 empty).
    Extra arguments of newer library versions (e.g. v2) are ignored: only tok / len / dist / src are produced."""
    T = len(x)
    t_idx = np.arange(T)
    srcs, lens, dists = [], [], []

    def occ_slot(key, p, length=None):
        p = np.where(p >= 0, p, -1)
        srcs.append(np.where(p >= 0, p + 1, -1))
        lens.append(_match_len(x, seg, key, p) if length is None else length)
        dists.append(_dist_bucket(np.where(p >= 0, key - p, 0)))

    def at(j, p):
        return np.where(p >= 0, prev_ok[j, np.maximum(p, 0)], -1)

    second = np.full((len(orders), T), -1, dtype=np.int64)
    for j in range(len(orders)):
        occ_slot(t_idx, prev_ok[j], len_raw[j])
        second[j] = at(j, prev_ok[j])
        occ_slot(t_idx, second[j])
    if cand == "vr":
        o = np.where(has_match, second[best, t_idx], -1)
        for _ in CHAIN_OCC:
            o = np.where(o >= 0, prev_ok[best, np.maximum(o, 0)], -1)
            occ_slot(t_idx, o)
        is_alpha = get_alpha_mask()[x]
        seg_start = np.searchsorted(seg, seg, side="left")
        anc = np.full((N_ANCHORS, T), -1, dtype=np.int64)
        cnt = np.zeros(T, dtype=np.int64)
        for d in range(ANCHOR_WINDOW):
            a = t_idx - d
            ok = (a >= seg_start) & (cnt < N_ANCHORS)
            ok[ok] = is_alpha[a[ok]]
            anc[cnt[ok], t_idx[ok]] = a[ok]
            cnt += ok
        for r in range(N_ANCHORS):
            a = anc[r]
            for n in ANCHOR_ORDERS:
                q = at(orders.index(n), a)
                length = _match_len(x, seg, np.maximum(a, 0), q)
                dist = _dist_bucket(np.where(q >= 0, a - q, 0))
                for i in range(1, SPAN + 1):
                    ok = (q >= 0) & (q + i <= t_idx)
                    srcs.append(np.where(ok, q + i, -1))
                    lens.append(np.where(ok, length, 0))
                    dists.append(np.where(ok, dist, 0))
    src = np.stack(srcs)
    assert (src <= t_idx).all(), "ICE candidate from the future"
    has = src >= 0
    return dict(ice_v_tok=np.where(has, x[np.maximum(src, 0)], -1),
                ice_v_len=np.where(has, np.stack(lens), 0), ice_v_dist=np.where(has, np.stack(dists), 0),
                ice_v_src=src)


_CHECKED = [False]


def vr_features(x, seg):
    """compute_ice_features(x, seg, cand="vr") + ice_v_src [K,T]. The first call per process also checks that
    tok / len / dist equal the library's."""
    orig = IF._candidates
    IF._candidates = _candidates_src
    try:
        f = compute_ice_features(x, seg, cand="vr")
    finally:
        IF._candidates = orig
    if not _CHECKED[0]:
        g = compute_ice_features(x, seg, cand="vr")
        for k in ("ice_v_tok", "ice_v_len", "ice_v_dist", "ice_s_star"):
            assert np.array_equal(f[k], g[k]), k
        _CHECKED[0] = True
    return f


def slot_features(tok, ln, src):
    """tok / ln / src [K,P] -> dict of [K,P] arrays: valid, delta (bucket 0..17), cnt (0..5 = n-1), rec (0..3),
    rep (first slot of its token), and ndist [P] (number of distinct candidate tokens)."""
    valid = tok >= 0
    same = (tok[:, None] == tok[None]) & valid[:, None] & valid[None]           # [k, j, P]
    occv = valid & OCC[:, None]
    comp = np.where(occv[None] & ~same, ln[None], -1).max(1)                   # best competing occurrence len
    delta = np.where(comp >= 0, np.clip(ln - comp, -D_CLIP, D_CLIP) + D_CLIP, D_NONE)
    cnt = np.clip(same.sum(1), 1, N_CNT) - 1
    m = np.where(same, src[None], -1).max(1)                                    # newest source of token c_k
    tri = np.tri(K, k=-1, dtype=bool)[:, :, None]                               # j < k
    rep = valid & ~(same & tri).any(1)
    rec = ((m[None, :, :] > m[:, None, :]) & rep[None]).sum(1)                 # [k, P]: newer distinct tokens
    return dict(valid=valid, delta=np.where(valid, delta, D_NONE), cnt=np.where(valid, cnt, 0),
                rec=np.where(valid, np.minimum(rec, N_REC - 1), 0), rep=rep, ndist=rep.sum(0), same=same,
                newest_src=m)


def rule_picks(tok, ln, ds, src, sf, s_star):
    """Token picked by every rule at each position [P] (-1 = no pick)."""
    valid = sf["valid"]
    slot = np.arange(K)[:, None]
    key = ln * (18 * K) + (17 - np.minimum(ds, 17)) * K + (K - 1 - slot)          # longest, nearer, slot order
    BIG = 64 * 18 * K

    def pick(score, mask):
        sc = np.where(mask, score, -1)
        i = sc.argmax(0)
        return np.where(mask.any(0), np.take_along_axis(tok, i[None], 0)[0], -1)

    occv = valid & OCC[:, None]
    ancv = valid & ~OCC[:, None]
    out = dict(s_star=s_star)
    out["longest"] = pick(key, valid)
    lo = pick(key, occv)
    out["longest_occ"] = np.where(occv.any(0), lo, out["longest"])
    out["newest"] = pick(src, valid)
    out["newest_occ"] = pick(src, occv)
    out["newest_anc"] = pick(src, ancv)
    out["vote"] = pick(sf["cnt"] * BIG + key, valid)
    lsum = np.where(sf["same"], ln[None], 0).sum(1)
    out["lenvote"] = pick(lsum * BIG + key, valid)
    return out


def code_doc_mask(x, eos, codeish, thr=0.25, min_len=16):
    """[T] bool: position t lies in a document (EOS-separated span inside the window) whose share of code-like
    tokens (newline, indentation runs, punctuation-only pieces) is >= thr. Calibrated on the validation splits:
    thr 0.25 keeps 99.1% of val_code tokens and 0.13% of val_prose tokens. Fragments < min_len tokens: False."""
    seg = segments_from_eos(x, eos)
    c = codeish[x].astype(np.float64)
    n = np.bincount(seg)
    frac = np.bincount(seg, weights=c) / np.maximum(n, 1)
    return ((frac >= thr) & (n >= min_len))[seg]


def codeish_tokens(tok, V):
    pieces = tok.convert_ids_to_tokens(list(range(V)))
    m = np.zeros(V, dtype=bool)
    for i, p in enumerate(pieces):
        p = p or ""
        s = p.replace("▁", "")
        m[i] = p == "<0x0A>" or p.startswith("▁▁") or (
            len(s) > 0 and all(ch in "()[]{}=_:.,#'\"<>+-*/%!&|;@" for ch in s))
    return m


def train_code_windows(data, seq_len, seed, eos, codeish, min_code_frac=0.5, start=0):
    """The training stream of data.TrainWindows (sample index i -> window perm_e[i mod n_win], same RNG), keeping
    windows whose code-document share is >= min_code_frac. Yields (i, window [seq_len+1], code mask [seq_len])."""
    L = seq_len
    n_win = max(1, (len(data) - 1) // L - 1)
    cache = {}
    i = start
    while True:
        epoch, k = divmod(i, n_win)
        if cache.get("epoch") != epoch:
            prng = np.random.default_rng([seed, 7919, epoch])
            cache.update(epoch=epoch, perm=prng.permutation(n_win),
                         shift=0 if epoch == 0 else int(prng.integers(0, L)))
        off = int(cache["perm"][k]) * L + cache["shift"]
        w = np.asarray(data[off: off + L + 1]).astype(np.int64)
        cm = code_doc_mask(w[:-1], eos, codeish)
        if cm.mean() >= min_code_frac:
            yield i, w, cm
        i += 1


def make_alpha(data_dir, kind="data"):
    """data: the anchor mask G_VR was trained with (data.alpha_mask_for_data); coverage: coverage_gate's regex
    mask (what rule_oracle.py used)."""
    if kind == "data":
        from ice_lm.data import alpha_mask_for_data
        return alpha_mask_for_data(data_dir)
    from transformers import AutoTokenizer
    from coverage_gate import alpha_mask
    from ice_lm.data import load_meta
    meta = load_meta(data_dir)
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
    return alpha_mask(tok, max(meta["vocab_size"], len(tok)))


# ---------------------------------------------------------------------------- F1 accumulation
def _empty():
    return dict(n=0, rules={r: 0 for r in RULES}, oracle=0,
                delta=np.zeros((2, 4, N_DELTA), np.int64), cnt=np.zeros((2, 4, N_CNT), np.int64),
                rec=np.zeros((2, 4, N_REC), np.int64), slots=np.zeros((2, 4), np.int64))


def _accumulate(acc, f, pos, y, extra_mask):
    """Add positions pos ([P]) with targets y ([P]) to acc (ambiguous & extra_mask only)."""
    tok, ln, ds, src = (f[k][:, pos] for k in ("ice_v_tok", "ice_v_len", "ice_v_dist", "ice_v_src"))
    sf = slot_features(tok, ln, src)
    sel = (sf["ndist"] >= 2) & extra_mask
    if not sel.any():
        return
    picks = rule_picks(tok, ln, ds, src, sf, f["ice_s_star"][pos])
    acc["n"] += int(sel.sum())
    for r in RULES:
        acc["rules"][r] += int(((picks[r] == y) & sel).sum())
    acc["oracle"] += int(((tok == y[None]).any(0) & sel).sum())
    v = sf["valid"] & sel[None]
    hit = (tok == y[None]) & v
    g = np.broadcast_to(GID[:, None], tok.shape)
    for name, arr in (("delta", sf["delta"]), ("cnt", sf["cnt"]), ("rec", sf["rec"])):
        np.add.at(acc[name][0], (g[v], arr[v]), 1)
        np.add.at(acc[name][1], (g[hit], arr[hit]), 1)
    np.add.at(acc["slots"][0], g[v], 1)
    np.add.at(acc["slots"][1], g[hit], 1)


def _merge(a, b):
    a["n"] += b["n"]
    a["oracle"] += b["oracle"]
    for r in RULES:
        a["rules"][r] += b["rules"][r]
    for k in ("delta", "cnt", "rec", "slots"):
        a[k] += b[k]
    return a


def _summ(acc):
    n = max(1, acc["n"])
    out = dict(n=acc["n"], **{r: acc["rules"][r] / n for r in RULES}, oracle=acc["oracle"] / n)
    prec = {}
    for gi, g in enumerate(GROUPS):
        d = {"all": dict(n=int(acc["slots"][0, gi]), prec=float(acc["slots"][1, gi] / max(1, acc["slots"][0, gi])))}
        for name, labels in (("delta", [str(v) for v in range(-D_CLIP, D_CLIP + 1)] + ["none"]),
                             ("cnt", [str(v) for v in range(1, N_CNT + 1)]),
                             ("rec", [str(v) for v in range(N_REC)])):
            tot, hit = acc[name][0, gi], acc[name][1, gi]
            d[name] = {lab: dict(n=int(tot[i]), prec=float(hit[i] / tot[i]) if tot[i] else None)
                       for i, lab in enumerate(labels)}
        prec[g] = d
    out["precision"] = prec
    return out


def _code_window(args):
    w, eos = args
    x, y = w[:-1], w[1:]
    f = vr_features(x, segments_from_eos(x, eos))
    pos = np.arange(len(x))
    wrong = (f["ice_s_star"] >= 0) & (f["ice_s_star"] != y)
    a_all, a_wrong = _empty(), _empty()
    _accumulate(a_all, f, pos, y, np.ones(len(x), bool))
    _accumulate(a_wrong, f, pos, y, wrong)
    return a_all, a_wrong


def _cr_item(args):
    seq, t1 = args
    x = np.asarray(seq[:-1], dtype=np.int64)
    f = vr_features(x, np.zeros(len(x), dtype=np.int64))
    acc = _empty()
    _accumulate(acc, f, np.array([t1 - 1]), np.array([seq[t1]]), np.ones(1, bool))
    return acc


def _init(alpha):
    set_alpha_mask(alpha)


def _print_prec(title, summ):
    print(f"\n-- per-group slot precision, {title} (n slots / precision)")
    for name, labs in (("delta", None), ("cnt", None), ("rec", None)):
        print(f"  {name}:")
        for g in GROUPS:
            d = summ["precision"][g][name]
            cells = "  ".join(f"{k}:{v['prec']:.2f}({v['n']})" for k, v in d.items() if v["n"] >= 50)
            print(f"    {g:7s} all {summ['precision'][g]['all']['prec']:.3f}  {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--gaps", type=int, nargs="+", default=[256, 1024, 3072])
    ap.add_argument("--mask", default="data", choices=["data", "coverage"])
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="rule_oracle2.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from ice_lm.coderecall import CATEGORIES, build_items
    from ice_lm.data import load_meta, open_tokens
    meta = load_meta(a.data_dir)
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
    eos = meta["eos_id"]
    alpha = make_alpha(a.data_dir, a.mask)
    t0 = time.time()
    res = {"config": vars(a), "code": {}, "coderecall": {}}
    pool = Pool(a.procs, initializer=_init, initargs=(alpha,))

    data = open_tokens(a.data_dir, "val_code")
    n_win = min(a.windows, (len(data) - 1) // a.seq_len)
    jobs = [(np.asarray(data[j * a.seq_len: j * a.seq_len + a.seq_len + 1]).astype(np.int64), eos)
            for j in range(n_win)]
    tot_all, tot_wrong = _empty(), _empty()
    for a_all, a_wrong in pool.imap(_code_window, jobs, chunksize=2):
        _merge(tot_all, a_all)
        _merge(tot_wrong, a_wrong)
    res["code"]["all"] = _summ(tot_all)
    res["code"]["s_star_wrong"] = _summ(tot_wrong)
    print(f"[{time.time() - t0:.0f}s] code: {n_win} windows", flush=True)

    pooled = _empty()
    for cat in CATEGORIES:
        items = []
        for gap in (a.gaps if cat not in ("increment", "track") else a.gaps[:1]):
            rng = np.random.default_rng([0, CATEGORIES.index(cat), gap])   # identical to run_coderecall
            seqs, spans, decide = build_items(tok, rng, cat, a.n, gap, open_tokens(a.data_dir, "val_code"), eos)
            items += list(zip(seqs, decide))
        acc = _empty()
        for r in pool.imap(_cr_item, items, chunksize=4):
            _merge(acc, r)
        res["coderecall"][cat] = _summ(acc)
        if cat in ("override_diff", "conflict_g2", "conflict_g5", "copy_diff"):
            _merge(pooled, acc)
    res["coderecall"]["focus4_pooled"] = _summ(pooled)
    print(f"[{time.time() - t0:.0f}s] coderecall done", flush=True)

    cols = RULES + ("oracle",)
    print(f"\n{'set':30s}{'n':>8s}" + "".join(f"{r[:11]:>12s}" for r in cols))
    for sect in ("code", "coderecall"):
        for k, s in res[sect].items():
            print(f"{sect + ':' + k:30s}{s['n']:8d}" + "".join(f"{s[r]:12.3f}" for r in cols))
    _print_prec("held-out code, ambiguous positions", res["code"]["all"])
    _print_prec("held-out code, s* wrong", res["code"]["s_star_wrong"])
    _print_prec("CodeRecall focus 4 (override_diff, conflict_g2/g5, copy_diff)", res["coderecall"]["focus4_pooled"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved", a.out, f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
