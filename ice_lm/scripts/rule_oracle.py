#!/usr/bin/env python
"""Free check (CPU, no model): which fixed rule over the ICE-V / VR candidate slots picks the right next token?

    python scripts/rule_oracle.py --data_dir /vol/data --out /vol/diag/rule_oracle.json

Only "ambiguous" positions are scored: the slots hold >= 2 distinct candidate tokens (else every rule agrees).
Rules over the slots of compute_ice_features(cand=...):
  s*        current ICE prediction (successor of the longest match, most recent occurrence)
  longest   the slot with the longest exact match (ice_v_len); ties -> smaller distance bucket, then slot order
  vote      the token held by most slots; ties -> the longest-match rule
  lenvote   the token with the largest summed match length
  oracle    the target is in the set (upper bound for any scorer)
Reported for held-out code (all / s* wrong) and per CodeRecall category (the deciding token), for V and VR slots.
Question it answers (round 2, NEXT_ROUND.md step 3): is "longest match wins" a real-code regularity or only a
CodeRecall template artefact?
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from coverage_gate import alpha_mask  # noqa: E402
from ice_lm.coderecall import CATEGORIES, build_items  # noqa: E402
from ice_lm.data import load_meta, open_tokens  # noqa: E402
from ice_lm.ice_features import compute_ice_features, n_slots, segments_from_eos, set_alpha_mask  # noqa: E402

RULES = ("s_star", "longest", "vote", "lenvote", "oracle")


def picks(f, cand, pos):
    """Rule picks at positions pos ([P]); also returns the ambiguity mask (>= 2 distinct candidates)."""
    K = n_slots(cand, 6)
    tok, ln, ds = f["ice_v_tok"][:K, pos], f["ice_v_len"][:K, pos], f["ice_v_dist"][:K, pos]
    valid = tok >= 0
    P = len(pos)
    out = {r: np.full(P, -1, dtype=np.int64) for r in RULES[:-1]}
    amb = np.zeros(P, dtype=bool)
    out["s_star"] = f["ice_s_star"][pos]
    for i in range(P):
        v = valid[:, i]
        if not v.any():
            continue
        t, l, d = tok[v, i], ln[v, i], ds[v, i]
        uniq = np.unique(t)
        amb[i] = len(uniq) >= 2
        order = np.lexsort((np.arange(len(t)), d, -l))      # longest, then nearest, then slot order
        out["longest"][i] = t[order[0]]
        counts = {u: int((t == u).sum()) for u in uniq}
        best = max(counts.values())
        tied = [u for u in uniq if counts[u] == best]
        out["vote"][i] = tied[0] if len(tied) == 1 else next(x for x in t[order] if x in tied)
        sums = {u: int(l[t == u].sum()) for u in uniq}
        out["lenvote"][i] = max(sums, key=sums.get)
    return out, amb, tok


def score(out, amb, tok, y):
    res = {"n": int(amb.sum())}
    for r in RULES[:-1]:
        res[r] = float(((out[r] == y) & amb).sum() / max(1, amb.sum()))
    res["oracle"] = float((((tok == y[None, :]).any(0)) & amb).sum() / max(1, amb.sum()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--windows", type=int, default=150)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--gaps", type=int, nargs="+", default=[256, 1024, 3072])
    ap.add_argument("--out", default="rule_oracle.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    meta = load_meta(a.data_dir)
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
    eos = meta["eos_id"]
    set_alpha_mask(alpha_mask(tok, max(meta["vocab_size"], len(tok))))
    res = {"code": {}, "coderecall": {}}

    data = open_tokens(a.data_dir, "val_code")
    n_win = min(a.windows, (len(data) - 1) // a.seq_len)
    for cand in ("v", "vr"):
        acc = {g: [] for g in ("all", "s_star_wrong")}
        for j in range(n_win):
            w = np.asarray(data[j * a.seq_len: j * a.seq_len + a.seq_len + 1]).astype(np.int64)
            x, y = w[:-1], w[1:]
            f = compute_ice_features(x, segments_from_eos(x, eos), cand=cand)
            pos = np.arange(len(x))
            out, amb, toks = picks(f, cand, pos)
            wrong = (f["ice_s_star"] >= 0) & (f["ice_s_star"] != y)
            acc["all"].append((out, amb, toks, y))
            acc["s_star_wrong"].append((out, amb & wrong, toks, y))
        for g, rows in acc.items():
            out = {r: np.concatenate([o[r] for o, _, _, _ in rows]) for r in RULES[:-1]}
            amb = np.concatenate([m for _, m, _, _ in rows])
            toks = np.concatenate([t for _, _, t, _ in rows], axis=1)
            y = np.concatenate([yy for _, _, _, yy in rows])
            res["code"][f"{cand}/{g}"] = score(out, amb, toks, y)
        print(cand, res["code"][f"{cand}/all"], res["code"][f"{cand}/s_star_wrong"], flush=True)

    for cat in CATEGORIES:
        for cand in ("v", "vr"):
            rows = []
            for gap in (a.gaps if cat not in ("increment", "track") else a.gaps[:1]):
                rng = np.random.default_rng([0, CATEGORIES.index(cat), gap])   # identical to run_coderecall
                seqs, spans, decide = build_items(tok, rng, cat, a.n, gap, open_tokens(a.data_dir, "val_code"), eos)
                for seq, t1 in zip(seqs, decide):
                    x = np.asarray(seq[:-1], dtype=np.int64)
                    f = compute_ice_features(x, np.zeros(len(x), dtype=np.int64), cand=cand)
                    pos = np.array([t1 - 1])
                    out, amb, toks = picks(f, cand, pos)
                    rows.append((out, amb, toks, np.array([seq[t1]])))
            out = {r: np.concatenate([o[r] for o, _, _, _ in rows]) for r in RULES[:-1]}
            amb = np.concatenate([m for _, m, _, _ in rows])
            toks = np.concatenate([t for _, _, t, _ in rows], axis=1)
            y = np.concatenate([yy for _, _, _, yy in rows])
            res["coderecall"][f"{cat}/{cand}"] = score(out, amb, toks, y)

    print(f"\n{'set':28s}{'n':>7s}" + "".join(f"{r:>9s}" for r in RULES))
    for sect in ("code", "coderecall"):
        for k, s in res[sect].items():
            print(f"{sect + ':' + k:28s}{s['n']:7d}" + "".join(f"{s[r]:9.3f}" for r in RULES))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
