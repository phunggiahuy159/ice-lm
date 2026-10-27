"""Inference helpers: an incremental ICE index for decoding, and memory accounting.

ICEIndex keeps, per sequence, the token store and one table per order n: n-gram -> (most recent end position,
the one before it). Feeding tokens one at a time yields exactly the features of
ice_features.compute_ice_features (tested in tests/test_ice_lm.py), including the true match length, the
agreement of the two most recent occurrences and the online reliability of each order.

With cand="v" / "vr" it also returns the ICE-V / VR candidate slots (ice_features.V_KEYS). V needs nothing new
(the second occurrence is already in the tables); VR also keeps, per order, the previous occurrence of the key
ending at each position (one int32 per order per token: the suffix chain and the anchor lookups).
With v2=True it also returns the source positions of the slots (ice_features.SRC_KEY) and the ICE-V2 slot features
(ice_features.V2_KEYS), computed from the current slots alone (no extra state, 0 bytes per token).

It runs on the CPU, next to the sampling loop; only ~10 small integers per order go to the GPU per step.
Two memory numbers are reported:
  * compact_bytes(): what a C / GPU implementation needs -- token ids (2 B/token for vocab <= 65536) +
    for orders n >= 2 an open-addressing table of 2 x int32 per distinct n-gram at load factor 0.5
    (16 B per distinct n-gram) + a direct int32 [vocab] x 2 table for order 1 + reliability bits.
  * the Python objects used here are much larger (dict + tuple overhead); that is an implementation artefact,
    not a property of the method, and is not what we report as ICE memory.
"""
from __future__ import annotations

from array import array
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .ice_features import (ANCHOR_ORDERS, ANCHOR_WINDOW, CHAIN_OCC, CNT_MAX, DEFAULT_ORDERS, DELTA_CLIP, DELTA_NONE,
                           DELTA_OFF, MAX_MATCH, N_ANCHORS, N_DIST_BUCKETS, REC_MAX, REL_EDGES, REL_WINDOW, SPAN,
                           SRC_KEY, V2_KEYS, V_KEYS, _dist_bucket, _len_bucket, compute_ice_features, get_alpha_mask,
                           n_occ_slots, n_slots, segments_from_eos)

FEATURE_KEYS_N = ("ice_succ", "ice_len", "ice_dist", "ice_agree", "ice_rel")      # [N] per position
FEATURE_KEYS_1 = ("ice_s_star", "ice_len_star", "ice_agree_star", "ice_rel_star", "ice_match_len",
                  "ice_dist_star")                                                 # scalar per position


def _dist_b(d: int) -> int:
    """_dist_bucket of one integer: floor(log2 d) + 1 = bit length, clipped (0 for no match)."""
    return min(d.bit_length(), N_DIST_BUCKETS - 1) if d > 0 else 0


def _rel_bucket(hist) -> int:
    if not hist:
        return 0
    return 1 + int(np.searchsorted(REL_EDGES, sum(hist) / len(hist), side="right"))


class ICEIndex:
    def __init__(self, orders: Sequence[int] = DEFAULT_ORDERS, eos_id: Optional[int] = None, vocab_size: int = 32000,
                 cand: str = "", v2: bool = False):
        self.orders = tuple(orders)
        self.cand = cand
        self.v2 = bool(cand) and v2
        self.K = n_slots(cand, len(self.orders)) if cand else 0
        self.alpha = get_alpha_mask() if cand == "vr" else None
        # VR: previous occurrence (same document) of the order-n key ending at each position, -1 none
        self.prev: Optional[List[array]] = [array("i") for _ in self.orders] if cand == "vr" else None
        self.N = len(self.orders)
        self.eos = eos_id
        self.vocab_size = vocab_size
        self.tokens = array("H" if vocab_size <= 65536 else "I")
        self.tables: List[dict] = [dict() for _ in self.orders]
        self.seg_start = 0
        self.hist = [deque(maxlen=REL_WINDOW) for _ in self.orders]
        self.hist_star = deque(maxlen=REL_WINDOW)
        self.pending = [-1] * self.N
        self.pending_star = -1

    def __len__(self):
        return len(self.tokens)

    def step(self, tok: int) -> Dict[str, object]:
        """Append one token (position t) and return the ICE features of position t (which predicts t+1)."""
        tok = int(tok)
        x = self.tokens
        t = len(x)
        if t > 0 and self.eos is not None and x[t - 1] == self.eos:      # a new document starts at t
            self.seg_start = t
            for h in self.hist:
                h.clear()
            self.hist_star.clear()
            self.pending = [-1] * self.N
            self.pending_star = -1
        # the predictions made at t-1 were about this token: update the reliability history
        for j, pr in enumerate(self.pending):
            if pr >= 0:
                self.hist[j].append(pr == tok)
        if self.pending_star >= 0:
            self.hist_star.append(self.pending_star == tok)
        x.append(tok)

        succ, length, dist, agree = [-1] * self.N, [0] * self.N, [0] * self.N, [0] * self.N
        lasts, seconds = [-1] * self.N, [-1] * self.N
        s0 = self.seg_start
        for j, n in enumerate(self.orders):
            if t - n + 1 < s0:
                continue
            key = tuple(x[t - n + 1: t + 1])
            table = self.tables[j]
            last, second = table.get(key, (-1, -1))
            if last < s0:                     # entries from earlier documents never match
                last = second = -1
            elif second < s0:
                second = -1
            table[key] = (t, last)
            if last < 0:
                continue
            lasts[j], seconds[j] = last, second
            succ[j] = x[last + 1]
            dist[j] = t - last
            k = n                              # the n tokens are equal by construction; extend backwards
            lim = min(MAX_MATCH, last - s0 + 1)
            while k < lim and x[t - k] == x[last - k]:
                k += 1
            length[j] = min(k, MAX_MATCH)
            agree[j] = 1 if second < 0 else (2 if x[second + 1] == x[last + 1] else 3)

        best, best_score = -1, -1
        for j in range(self.N):
            if length[j] > 0 and length[j] * (self.N + 1) + j > best_score:
                best, best_score = j, length[j] * (self.N + 1) + j
        out = dict(
            ice_succ=succ,
            ice_len=_len_bucket(np.array(length)).tolist(),
            ice_dist=_dist_bucket(np.array(dist)).tolist(),
            ice_agree=agree,
            ice_rel=[_rel_bucket(h) for h in self.hist],
            ice_s_star=succ[best] if best >= 0 else -1,
            ice_len_star=int(_len_bucket(np.array([length[best]]))[0]) if best >= 0 else 0,
            ice_agree_star=agree[best] if best >= 0 else 0,
            ice_rel_star=_rel_bucket(self.hist_star),
            ice_match_len=length[best] if best >= 0 else 0,
            ice_dist_star=dist[best] if best >= 0 else 0,
        )
        if self.prev is not None:
            for j in range(self.N):
                self.prev[j].append(lasts[j])
        if self.cand:
            out.update(self._candidates(t, lasts, seconds, length, dist, best))
        self.pending = succ
        self.pending_star = out["ice_s_star"]
        return out

    def _match(self, a: int, q: int) -> int:
        """Exact backward match length of x[..a] and x[..q] (q < a, same document), capped at MAX_MATCH."""
        x, s0, k = self.tokens, self.seg_start, 0
        while k < MAX_MATCH and q - k >= s0 and x[a - k] == x[q - k]:
            k += 1
        return k

    def _candidates(self, t, lasts, seconds, length, dist, best) -> Dict[str, List[int]]:
        """ICE-V / VR slots of position t, in the order of ice_features._candidates."""
        x = self.tokens
        tok, ln, db, src = [], [], [], []

        def occ(p, key=t, n=None):        # candidate x[p+1] of an occurrence p of the key ending at `key`
            if p < 0:
                tok.append(-1), ln.append(0), db.append(0), src.append(-1)
            else:
                tok.append(x[p + 1]), ln.append(self._match(key, p) if n is None else n), db.append(_dist_b(key - p))
                src.append(p + 1)

        for j in range(self.N):
            occ(lasts[j], n=length[j])
            occ(seconds[j])
        if self.cand == "vr":
            o = seconds[best] if best >= 0 else -1
            for _ in CHAIN_OCC:
                o = self.prev[best][o] if o >= 0 else -1
                occ(o)
            anchors = [a for a in range(t, max(self.seg_start, t - ANCHOR_WINDOW + 1) - 1, -1)
                       if self.alpha[x[a]]][:N_ANCHORS]
            anchors += [-1] * (N_ANCHORS - len(anchors))
            for a in anchors:
                for n in ANCHOR_ORDERS:
                    q = self.prev[self.orders.index(n)][a] if a >= 0 else -1
                    m = self._match(a, q) if q >= 0 else 0
                    for i in range(1, SPAN + 1):
                        if q >= 0 and q + i <= t:           # q + i > t would be the label (or later)
                            tok.append(x[q + i]), ln.append(m), db.append(_dist_b(a - q)), src.append(q + i)
                        else:
                            tok.append(-1), ln.append(0), db.append(0), src.append(-1)
        out = dict(ice_v_tok=tok, ice_v_len=ln, ice_v_dist=db)
        if self.v2:
            out[SRC_KEY] = src
            out.update(self._v2_features(tok, ln, src))
        return out

    def _v2_features(self, tok, ln, src) -> Dict[str, List[int]]:
        """ICE-V2 slot features of one position (ice_features.v2_features), O(K) per step."""
        n_occ = n_occ_slots(self.cand, self.N)
        cnt, newest, best = {}, {}, {}           # per distinct token: #slots, max source, best occurrence len
        for k, (c, n, i) in enumerate(zip(tok, ln, src)):
            if c < 0:
                continue
            cnt[c] = cnt.get(c, 0) + 1
            newest[c] = max(newest.get(c, -1), i)
            if k < n_occ:
                best[c] = max(best.get(c, -1), n)
        rank = {c: r for r, c in enumerate(sorted(newest, key=newest.get, reverse=True))}
        top2 = sorted(best.items(), key=lambda kv: -kv[1])[:2]     # the best competitor is one of these
        delta, n_k, rec = [], [], []
        for c, n in zip(tok, ln):
            if c < 0:
                delta.append(0), n_k.append(0), rec.append(0)
                continue
            comp = next((v for cc, v in top2 if cc != c), None)
            delta.append(DELTA_NONE if comp is None else DELTA_OFF + min(max(n - comp, -DELTA_CLIP), DELTA_CLIP))
            n_k.append(min(cnt[c], CNT_MAX)), rec.append(min(rank[c], REC_MAX))
        return dict(ice_v_delta=delta, ice_v_cnt=n_k, ice_v_rec=rec)

    def extend(self, tokens: Sequence[int]) -> Dict[str, np.ndarray]:
        """Feed several tokens; returns features stacked like compute_ice_features ([N,T] and [T])."""
        feats = [self.step(t) for t in tokens]
        keys_k = (V_KEYS + ((SRC_KEY,) + V2_KEYS if self.v2 else ())) if self.cand else ()
        if not feats:
            return {**{k: np.zeros((self.N, 0), dtype=np.int64) for k in FEATURE_KEYS_N},
                    **{k: np.zeros((self.K, 0), dtype=np.int64) for k in keys_k},
                    **{k: np.zeros(0, dtype=np.int64) for k in FEATURE_KEYS_1}}
        out = {k: np.array([f[k] for f in feats], dtype=np.int64).T for k in FEATURE_KEYS_N + keys_k}
        out.update({k: np.array([f[k] for f in feats], dtype=np.int64) for k in FEATURE_KEYS_1})
        return out

    def prefill(self, tokens: Sequence[int]) -> Dict[str, np.ndarray]:
        """Process a whole prompt at once (empty index only): features from the vectorized compute_ice_features,
        then the tables and reliability history that step() needs to continue. Same features, tables and state
        as extend() (tables keep earlier documents' n-grams too, as a simple implementation would)."""
        if len(self):
            raise ValueError("prefill() needs an empty index; use extend() to add tokens")
        x = np.asarray(tokens, dtype=np.int64)
        T = len(x)
        if T == 0:
            return self.extend([])
        seg = segments_from_eos(x, self.eos)
        f = compute_ice_features(x, seg, self.orders, self.cand, self.v2)
        xl = x.tolist()
        self.tokens.extend(xl)
        s0 = int(np.searchsorted(seg, seg[-1]))              # start of the current (last) document
        self.seg_start = s0
        segl = seg.tolist()
        for j, n in enumerate(self.orders):
            table = self.tables[j]
            ptr = [-1] * T
            for t in range(n - 1, T):
                if segl[t - n + 1] != segl[t]:                  # n-gram crosses a document boundary
                    continue
                key = tuple(xl[t - n + 1: t + 1])
                prev = table.get(key)
                ptr[t] = prev[0] if prev and segl[prev[0]] == segl[t] else -1
                table[key] = (t, ptr[t])
            if self.prev is not None:
                self.prev[j].extend(ptr)

        def history(pred):
            ev = np.nonzero(pred[s0: T - 1] >= 0)[0] + s0      # predictions whose outcome is known
            return deque((pred[ev] == x[ev + 1]).tolist()[-REL_WINDOW:], maxlen=REL_WINDOW)

        self.hist = [history(f["ice_succ"][j]) for j in range(self.N)]
        self.hist_star = history(f["ice_s_star"])
        self.pending = [int(v) for v in f["ice_succ"][:, T - 1]]
        self.pending_star = int(f["ice_s_star"][T - 1])
        return f

    # -- memory --------------------------------------------------------------
    def distinct_ngrams(self) -> List[int]:
        return [len(t) for t in self.tables]

    def compact_bytes(self) -> Dict[str, int]:
        tok_b = len(self.tokens) * (2 if self.vocab_size <= 65536 else 4)
        tables = 0
        for n, d in zip(self.orders, self.distinct_ngrams()):
            tables += self.vocab_size * 8 if n == 1 else d * 16     # 2 x int32 per entry; load factor 0.5
        rel = (self.N + 1) * REL_WINDOW // 8
        out = dict(tokens=tok_b, tables=tables, reliability=rel, total=tok_b + tables + rel)
        if self.prev is not None:                                    # VR: int32 per order per token
            out["prev"] = 4 * self.N * len(self.tokens)
            out["total"] += out["prev"]
        return out


def features_to_tensors(feats: List[Dict[str, np.ndarray]], device) -> Dict[str, torch.Tensor]:
    """Batch the per-row features of a chunk ([N,T] / [T] arrays) into model inputs [B,N,T] / [B,T]."""
    keys = [k for k in FEATURE_KEYS_N + V_KEYS + V2_KEYS + FEATURE_KEYS_1 if k in feats[0]]
    return {k: torch.from_numpy(np.stack([np.asarray(f[k], dtype=np.int64) for f in feats])).to(device)
            for k in keys}


def step_features_to_tensors(feats: List[Dict[str, object]], device) -> Dict[str, torch.Tensor]:
    """Batch single-position features (ICEIndex.step outputs) into [B,N,1] / [B,1] tensors."""
    out = {k: torch.tensor([f[k] for f in feats], dtype=torch.long)[:, :, None]
           for k in FEATURE_KEYS_N + V_KEYS + V2_KEYS if k in feats[0]}
    out.update({k: torch.tensor([f[k] for f in feats], dtype=torch.long)[:, None] for k in FEATURE_KEYS_1})
    return {k: v.to(device) for k, v in out.items()}
