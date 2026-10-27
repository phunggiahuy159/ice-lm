#!/usr/bin/env python
"""F2 (decision2.md section 3, CPU, no backbone): a token-only listwise logistic scorer over the ICE-VR slots,
trained on TRAIN code windows and scored on held-out val_code.

    python scripts/token_scorer.py --data_dir /vol/data --out /vol/diag/token_scorer.json

Score of slot k (all tables are learned; the softmax is over the valid slots only, no sentinel):
    s_k = b[k] + L[k, len_k] + D[g_k, dist_k]                       (base: what ICEPointer's b + L + D express)
        + R[g_k, Delta_k] + A[g_k, n_k] + N[g_k, r_k]               (the 0-byte V2 features, rule_oracle2.py)
Listwise log-loss  -log sum_{k : c_k = y} softmax(s)_k  on positions with >= 2 distinct candidate tokens and y in
the set (the "which candidate is y" label of decision2's aux term). The objective is convex; fitted by L-BFGS.
Train windows: the training stream of data.TrainWindows (seed --seed) restricted to code documents
(rule_oracle2.code_doc_mask: windows with >= 50% code tokens, positions in code documents only).
Held-out: the first --val_windows val_code windows (all positions).
Reported per variant: held-out log-loss (paired bootstrap over windows vs base), token-level rank-1 (the token with
the largest summed slot probability is y) on ambiguous positions, on s*-wrong positions (unconditional, the
rule_oracle convention, oracle ceiling 0.333) and on s*-wrong positions with y in the set.
Decision rule (decision2): keep R/A/N only if they improve held-out log-loss by >= 0.01 nats.
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
import torch  # noqa: E402

from ice_lm.data import load_meta, open_tokens  # noqa: E402
from ice_lm.ice_features import MAX_MATCH, segments_from_eos, set_alpha_mask  # noqa: E402
from rule_oracle2 import (GID, K, N_CNT, N_DELTA, N_REC, codeish_tokens, make_alpha, slot_features,  # noqa: E402
                          train_code_windows, vr_features)

DMAX = 12                    # model.V_DIST_MAX: distance buckets above it are merged, as in ICEPointer
NL = MAX_MATCH + 1
TABLES = dict(L=K * NL, D=4 * (DMAX + 1), R=4 * N_DELTA, A=4 * N_CNT, N=4 * N_REC)
VARIANTS = {"base": (), "+R": ("R",), "+A": ("A",), "+N": ("N",), "+RAN": ("R", "A", "N"),
            "+AN (no R)": ("A", "N"), "+RN (no A)": ("R", "N"), "+RA (no N)": ("R", "A")}


def rows_of(x, y, seg, mask, keep_tok):
    """Feature rows of the ambiguous positions inside mask."""
    f = vr_features(x, seg)
    tok, ln, ds, src = (f[k] for k in ("ice_v_tok", "ice_v_len", "ice_v_dist", "ice_v_src"))
    sf = slot_features(tok, ln, src)
    sel = (sf["ndist"] >= 2) & mask
    p = np.nonzero(sel)[0]
    g = GID[:, None]
    hit = (tok[:, p] == y[None, p]) & sf["valid"][:, p]
    r = dict(L=(np.arange(K)[:, None] * NL + ln[:, p]).T.astype(np.int16),
             D=(g * (DMAX + 1) + np.minimum(ds[:, p], DMAX)).T.astype(np.int16),
             R=(g * N_DELTA + sf["delta"][:, p]).T.astype(np.int16),
             A=(g * N_CNT + sf["cnt"][:, p]).T.astype(np.int16),
             N=(g * N_REC + sf["rec"][:, p]).T.astype(np.int16),
             valid=sf["valid"][:, p].T, hit=hit.T,
             s_wrong=(f["ice_s_star"][p] >= 0) & (f["ice_s_star"][p] != y[p]))
    if keep_tok:
        r["tok"] = tok[:, p].T.astype(np.int32)
    return r


def _work(args):
    w, cm, eos, keep_tok = args
    x, y = w[:-1], w[1:]
    return rows_of(x, y, segments_from_eos(x, eos), cm, keep_tok)


def _init(alpha):
    set_alpha_mask(alpha)


def cat_rows(rows):
    out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    out["win"] = np.concatenate([np.full(len(r["valid"]), i) for i, r in enumerate(rows)])
    return out


class Scorer(torch.nn.Module):
    def __init__(self, feats):
        super().__init__()
        self.feats = ("L", "D") + tuple(feats)
        self.b = torch.nn.Parameter(torch.zeros(K))
        self.t = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.zeros(TABLES[k])) for k in self.feats})

    def forward(self, d):
        s = self.b[None].expand(d["valid"].shape[0], K)
        for k in self.feats:
            s = s + self.t[k][d[k]]
        return s.masked_fill(~d["valid"], float("-inf"))


def to_t(rows, idx=None):
    keys = ("L", "D", "R", "A", "N", "valid", "hit")
    return {k: torch.from_numpy(rows[k] if idx is None else rows[k][idx]).long()
            if k not in ("valid", "hit") else torch.from_numpy(rows[k] if idx is None else rows[k][idx])
            for k in keys}


def listwise_nll(s, hit):
    return s.logsumexp(-1) - s.masked_fill(~hit, float("-inf")).logsumexp(-1)


def fit(train, feats, l2, iters):
    m = Scorer(feats)
    opt = torch.optim.LBFGS(m.parameters(), lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        loss = listwise_nll(m(train), train["hit"]).mean() + l2 * sum((p ** 2).sum() for p in m.parameters())
        loss.backward()
        return loss

    opt.step(closure)
    return m


@torch.no_grad()
def evaluate(m, val, chunk=100_000):
    """Per-row listwise nll (rows with y in the set) and token-level rank-1 correctness (all rows)."""
    nll, top = [], []
    P = len(val["valid"])
    for a in range(0, P, chunk):
        idx = np.arange(a, min(P, a + chunk))
        d = to_t(val, idx)
        s = m(d)
        p = s.softmax(-1)
        tok = torch.from_numpy(val["tok"][idx]).long()
        same = (tok[:, :, None] == tok[:, None, :]) & d["valid"][:, :, None] & d["valid"][:, None, :]
        ptok = (same.float() @ p[:, :, None]).squeeze(-1)                           # P(token of slot k)
        py = (p * d["hit"]).sum(-1)
        pother = ptok.masked_fill(d["hit"] | ~d["valid"], -1.0).amax(-1)
        top.append((py > pother).numpy())
        nll.append(listwise_nll(s, d["hit"]).numpy())
    return np.concatenate(nll), np.concatenate(top)


def boot(diff_sum, cnt, n=2000, seed=0):
    """Paired bootstrap over windows: diff_sum / cnt per window -> mean and 95% CI of the ratio estimator."""
    rng = np.random.default_rng(seed)
    W = len(cnt)
    idx = rng.integers(0, W, (n, W))
    est = diff_sum[idx].sum(1) / np.maximum(1, cnt[idx].sum(1))
    return float(diff_sum.sum() / max(1, cnt.sum())), [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--train_split", default="train",
                    help="train (the training stream, code windows) or val_code (fallback: the val_code windows "
                         "after the --val_windows held-out ones)")
    ap.add_argument("--train_windows", type=int, default=300)
    ap.add_argument("--val_windows", type=int, default=300)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7, help="training-stream seed (fit_scorer.py's default)")
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--mask", default="data", choices=["data", "coverage"])
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="token_scorer.json")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    t0 = time.time()
    from transformers import AutoTokenizer
    meta = load_meta(a.data_dir)
    eos = meta["eos_id"]
    codeish = codeish_tokens(AutoTokenizer.from_pretrained(meta["tokenizer"]), meta["vocab_size"])
    alpha = make_alpha(a.data_dir, a.mask)
    pool = Pool(a.procs, initializer=_init, initargs=(alpha,))

    tr_data = open_tokens(a.data_dir, a.train_split)
    jobs, idxs = [], []
    if a.train_split == "val_code":
        # fallback when train.bin is not available: val_code windows AFTER the held-out ones (disjoint)
        n_all = (len(tr_data) - 1) // a.seq_len
        for j in range(a.val_windows, min(n_all, a.val_windows + a.train_windows)):
            w = np.asarray(tr_data[j * a.seq_len: j * a.seq_len + a.seq_len + 1]).astype(np.int64)
            jobs.append((w, np.ones(a.seq_len, bool), eos, False))
            idxs.append(j)
    else:
        gen = train_code_windows(tr_data, a.seq_len, a.seed, eos, codeish)
        for _ in range(a.train_windows):
            i, w, cm = next(gen)
            jobs.append((w, cm, eos, False))
            idxs.append(i)
    train = cat_rows(pool.map(_work, jobs, chunksize=2))
    n_code = int(sum(j[1].sum() for j in jobs))
    print(f"[{time.time() - t0:.0f}s] train: {len(jobs)} code windows (stream indices {idxs[0]}..{idxs[-1]}, "
          f"{n_code:,} code positions), {len(train['valid']):,} ambiguous rows", flush=True)

    va_data = open_tokens(a.data_dir, "val_code")
    n_val = min(a.val_windows, (len(va_data) - 1) // a.seq_len)
    vjobs = [(np.asarray(va_data[j * a.seq_len: j * a.seq_len + a.seq_len + 1]).astype(np.int64),
              np.ones(a.seq_len, bool), eos, True) for j in range(n_val)]
    val = cat_rows(pool.map(_work, vjobs, chunksize=2))
    pool.close()
    print(f"[{time.time() - t0:.0f}s] held-out: {n_val} val_code windows, {len(val['valid']):,} ambiguous rows",
          flush=True)

    tr_in = np.nonzero(train["hit"].any(1))[0]
    tr = to_t(train, tr_in)
    v_in = val["hit"].any(1)
    res = dict(config=vars(a), n_train_rows=int(len(tr_in)), n_val_rows=int(v_in.sum()),
               n_val_ambiguous=int(len(v_in)), n_val_s_wrong=int(val["s_wrong"].sum()),
               oracle_s_wrong=float(v_in[val["s_wrong"]].mean()), variants={})
    W = n_val
    per = {}
    for name, feats in VARIANTS.items():
        m = fit(tr, feats, a.l2, a.iters)
        with torch.no_grad():
            tr_loss = float(listwise_nll(m(tr), tr["hit"]).mean())
        nll, top = evaluate(m, val)
        sw = val["s_wrong"]
        r = dict(train_logloss=tr_loss, val_logloss=float(nll[v_in].mean()),
                 rank1_ambiguous=float(top.mean()), rank1_s_wrong=float(top[sw].mean()),
                 rank1_s_wrong_y_in_set=float(top[sw & v_in].mean()),
                 rank1_y_in_set=float(top[v_in].mean()))
        per[name] = (nll, top)
        res["variants"][name] = r
        print(f"[{time.time() - t0:.0f}s] {name:12s} " + " ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)

    # paired comparisons vs base (bootstrap over held-out windows)
    win = val["win"]
    cnt_in = np.bincount(win[v_in], minlength=W)
    sw = val["s_wrong"]
    cnt_sw = np.bincount(win[sw], minlength=W)
    for name in VARIANTS:
        if name == "base":
            continue
        dn = np.bincount(win[v_in], weights=per["base"][0][v_in] - per[name][0][v_in], minlength=W)
        dt = np.bincount(win[sw], weights=(per[name][1].astype(float) - per["base"][1])[sw], minlength=W)
        g, ci = boot(dn, cnt_in)
        g2, ci2 = boot(dt, cnt_sw)
        res["variants"][name].update(logloss_gain_vs_base=g, logloss_gain_ci=ci, rank1_s_wrong_gain=g2,
                                     rank1_s_wrong_gain_ci=ci2)
    full = res["variants"]["+RAN"]
    res["decision"] = dict(rule="keep R/A/N only if held-out log-loss improves by >= 0.01 nats",
                           gain_full=full["logloss_gain_vs_base"],
                           keep={k: res["variants"][f"+{k}"]["logloss_gain_vs_base"] >= 0.01 for k in "RAN"},
                           keep_all=full["logloss_gain_vs_base"] >= 0.01)

    print(f"\n{'variant':14s}{'train':>8s}{'val':>8s}{'gain':>8s}{'95% CI':>18s}{'r1 amb':>8s}{'r1 s*w':>8s}"
          f"{'gain':>8s}{'r1 s*w,y':>9s}")
    for name, r in res["variants"].items():
        ci = r.get("logloss_gain_ci", [0, 0])
        print(f"{name:14s}{r['train_logloss']:8.4f}{r['val_logloss']:8.4f}{r.get('logloss_gain_vs_base', 0):+8.4f}"
              f"   [{ci[0]:+.4f},{ci[1]:+.4f}]{r['rank1_ambiguous']:8.3f}{r['rank1_s_wrong']:8.3f}"
              f"{r.get('rank1_s_wrong_gain', 0):+8.3f}{r['rank1_s_wrong_y_in_set']:9.3f}")
    print(f"oracle (y in set) on s*-wrong ambiguous positions: {res['oracle_s_wrong']:.3f}")
    print("decision:", res["decision"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved", a.out, f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
