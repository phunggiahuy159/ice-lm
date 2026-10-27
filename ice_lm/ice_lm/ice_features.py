"""In-Context Engram (ICE) features: an exact, non-learned n-gram lookup over the current context.

For every position t (which predicts token t+1) and every order n, find the most recent earlier
position p < t whose last n tokens equal the last n tokens at t, and report:

    succ[n, t]  = x[p + 1]          the successor token (-1 if no match)
    len_b[n, t] = bucket(true match length at p, extended backwards up to MAX_MATCH)  (0 = no match)
    dist_b[n, t]= bucket(log2(t - p))                                                (0 = no match)

plus the longest-match successor s_star[t] and its length bucket len_star[t].
p < t, so p + 1 <= t: ICE never sees the future.

Two signals tell the model when NOT to trust a lookup (both causal, both cheap):

    agree[n, t] = 0 no match | 1 only one earlier occurrence | 2 the two most recent earlier occurrences
                  had the same successor | 3 they had different successors (ambiguous key)
    rel[n, t]   = reliability of order n in this document so far: of its last REL_WINDOW predictions
                  (positions t' < t with a match), how many were right (succ[n, t'] == x[t'+1]).
                  0 no history | 1 <20% | 2 20-50% | 3 50-80% | 4 >=80%

and the same two for the longest-match prediction s* (agree_star, rel_star).

ICE-V / ICE-VR candidate sets (cand="v" / "vr", used by the pointer-sentinel arms, see model.ICEPointer): K slots
per position, each a token c_k = x[i_k] with i_k <= t, its exact match length (0..MAX_MATCH) and a distance bucket:

    V  (K = 2N)      order j x occurrence {last, second}: the successor x[p+1], match length of that occurrence,
                     bucket(t - p)
    VR (K = 2N + 22) V + chain: occurrences 3..8 of the longest-matching order (each with its own match length)
                     + anchors: the 2 most recent alphabetic tokens a in x[t-7..t] (vocabulary mask, see
                     set_alpha_mask); for each, the latest earlier occurrence q < a under the order-1 and the
                     order-2 key and the span x[q+1..q+4]: slots with q + i > t are empty (they would be the label)

Documents packed into one window are separated by `seg` (segment ids); matches never cross a
segment boundary. Everything is vectorized NumPy (sort-based), cheap enough for DataLoader workers.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

DEFAULT_ORDERS = (1, 2, 3, 4, 6, 8)
MAX_MATCH = 32
LEN_EDGES = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
N_LEN_BUCKETS = len(LEN_EDGES) + 1      # 0 = no match, 1..10
N_DIST_BUCKETS = 18                     # 0 = no match, 1..17 = floor(log2(dist)) + 1 (clipped)
N_AGREE_BUCKETS = 4                     # 0 none, 1 single occurrence, 2 agree, 3 disagree
N_REL_BUCKETS = 5                       # 0 no history, then <20%, 20-50%, 50-80%, >=80% right
REL_WINDOW = 32
REL_EDGES = np.array([0.2, 0.5, 0.8])
# ICE-V / VR candidate slots
V_KEYS = ("ice_v_tok", "ice_v_len", "ice_v_dist")     # [K, T] each
CHAIN_OCC = tuple(range(3, 9))                        # deeper occurrences of the longest-matching order
N_ANCHORS = 2
ANCHOR_WINDOW = 8                                     # anchors are searched in x[t-7..t]
ANCHOR_ORDERS = (1, 2)
SPAN = 4                                              # span tokens x[q+1..q+4] after an anchor occurrence

_P = np.uint64(0x9E3779B97F4A7C15)
_ALPHA: Optional[np.ndarray] = None


def n_slots(cand: str, n_orders: int) -> int:
    """Number of candidate slots K of a candidate set ("v" / "vr")."""
    if cand == "v":
        return 2 * n_orders
    if cand == "vr":
        return 2 * n_orders + len(CHAIN_OCC) + N_ANCHORS * len(ANCHOR_ORDERS) * SPAN
    raise ValueError(f"unknown candidate set {cand!r}")


def set_alpha_mask(mask: Optional[np.ndarray]) -> None:
    """Boolean mask over the vocabulary: which tokens can be identifier anchors (ICE-VR). Set it once per process
    (data.alpha_mask_for_data builds it from the tokenizer); None unsets it."""
    global _ALPHA
    _ALPHA = None if mask is None else np.asarray(mask, dtype=bool)


def get_alpha_mask() -> np.ndarray:
    if _ALPHA is None:
        raise RuntimeError("ICE-VR needs the alphabetic-token mask: call ice_features.set_alpha_mask first "
                           "(data.alpha_mask_for_data(data_dir) builds it from the tokenizer)")
    return _ALPHA


def _len_bucket(length: np.ndarray) -> np.ndarray:
    """0 for length 0, else 1 + index of the largest edge <= length."""
    b = np.searchsorted(LEN_EDGES, length, side="right")  # 0..10, and 0 only if length < 1
    return b.astype(np.int64)


def _dist_bucket(dist: np.ndarray) -> np.ndarray:
    out = np.zeros_like(dist, dtype=np.int64)
    pos = dist > 0
    out[pos] = np.minimum(np.floor(np.log2(dist[pos])).astype(np.int64) + 1, N_DIST_BUCKETS - 1)
    return out


def _reliability(pred: np.ndarray, x: np.ndarray, seg: np.ndarray, window: int = REL_WINDOW) -> np.ndarray:
    """pred [T] (-1 = no prediction). For each t: bucket of the hit rate of the last `window` predictions made
    at positions t' < t of the same segment (a prediction at t' is about x[t'+1], known at time t'+1 <= t)."""
    T = len(pred)
    ev = np.nonzero(pred[: T - 1] >= 0)[0]                       # predictions whose outcome is in the window
    out = np.zeros(T, dtype=np.int64)
    if len(ev) == 0:
        return out
    hits = np.concatenate([[0], np.cumsum(pred[ev] == x[ev + 1])])
    t_idx = np.arange(T)
    seg_start = np.searchsorted(seg, seg, side="left")           # seg is non-decreasing
    hi = np.searchsorted(ev, t_idx, side="left")                 # events with t' < t
    lo = np.maximum(hi - window, np.searchsorted(ev, seg_start, side="left"))
    cnt = hi - lo
    has = cnt > 0
    rate = np.zeros(T)
    rate[has] = (hits[hi[has]] - hits[lo[has]]) / cnt[has]
    out[has] = 1 + np.searchsorted(REL_EDGES, rate[has], side="right")
    return out


def _prev_occurrence(h: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """For each t: most recent p < t with valid[p] and h[p] == h[t] (and valid[t]); else -1."""
    T = len(h)
    prev = np.full(T, -1, dtype=np.int64)
    idx = np.nonzero(valid)[0]
    if len(idx) < 2:
        return prev
    hv = h[idx]
    order = np.argsort(hv, kind="stable")      # stable: equal hashes stay in position order
    hs = hv[order]
    same = hs[1:] == hs[:-1]
    prev[idx[order[1:][same]]] = idx[order[:-1][same]]
    return prev


def _match_len(x: np.ndarray, seg: np.ndarray, a: np.ndarray, p: np.ndarray) -> np.ndarray:
    """a, p [M]: key end positions with p < a in the same document (p = -1: none) -> exact backward match length
    of x[..a] and x[..p], capped at MAX_MATCH, staying inside the document (as ice_len, before bucketing)."""
    out = np.zeros(len(a), dtype=np.int64)
    idx = np.nonzero(p >= 0)[0]
    for k in range(MAX_MATCH):
        pk = p[idx] - k
        ok = pk >= 0
        ok[ok] = (x[a[idx[ok]] - k] == x[pk[ok]]) & (seg[pk[ok]] == seg[a[idx[ok]]])
        idx = idx[ok]
        if not len(idx):
            break
        out[idx] += 1
    return out


def _candidates(x, seg, cand, orders, prev_ok, len_raw, best, has_match) -> Dict[str, np.ndarray]:
    """ICE-V / VR slots (see the module docstring). prev_ok [N,T]: verified most recent earlier occurrence of
    the order-n key ending at each position (-1 none); len_raw [N,T]: match length of that occurrence;
    best / has_match [T]: the longest-matching order (as for s*)."""
    T = len(x)
    t_idx = np.arange(T)
    srcs, lens, dists = [], [], []       # per slot: candidate position i_k (-1 empty), match length, distance

    def occ_slot(key, p, length=None):  # candidate x[p+1] of an occurrence p of the key ending at `key`
        p = np.where(p >= 0, p, -1)
        srcs.append(np.where(p >= 0, p + 1, -1))
        lens.append(_match_len(x, seg, key, p) if length is None else length)
        dists.append(_dist_bucket(np.where(p >= 0, key - p, 0)))

    def at(j, p):                        # prev_ok[j, p] for p >= 0, else -1
        return np.where(p >= 0, prev_ok[j, np.maximum(p, 0)], -1)

    second = np.full((len(orders), T), -1, dtype=np.int64)
    for j in range(len(orders)):
        occ_slot(t_idx, prev_ok[j], len_raw[j])
        second[j] = at(j, prev_ok[j])
        occ_slot(t_idx, second[j])
    if cand == "vr":
        # deep suffix chain of the longest-matching order
        o = np.where(has_match, second[best, t_idx], -1)
        for _ in CHAIN_OCC:
            o = np.where(o >= 0, prev_ok[best, np.maximum(o, 0)], -1)
            occ_slot(t_idx, o)
        # identifier anchors: the N_ANCHORS most recent alphabetic tokens in x[t-7..t] (same document)
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
                    ok = (q >= 0) & (q + i <= t_idx)          # q + i > t would be the label (or later)
                    srcs.append(np.where(ok, q + i, -1))
                    lens.append(np.where(ok, length, 0))
                    dists.append(np.where(ok, dist, 0))
    src = np.stack(srcs)
    assert (src <= t_idx).all(), "ICE candidate from the future"
    has = src >= 0
    return dict(ice_v_tok=np.where(has, x[np.maximum(src, 0)], -1),
                ice_v_len=np.where(has, np.stack(lens), 0), ice_v_dist=np.where(has, np.stack(dists), 0))


def compute_ice_features(
    x: np.ndarray,
    seg: Optional[np.ndarray] = None,
    orders: Sequence[int] = DEFAULT_ORDERS,
    cand: str = "",
) -> Dict[str, np.ndarray]:
    """x: [T] token ids. seg: [T] segment (document) ids, non-decreasing; None = one document.
    cand: "" (ICE / C arms), "v" or "vr": also return the ICE-V / VR candidate slots (V_KEYS, [K,T])."""
    x = np.asarray(x, dtype=np.int64)
    T = len(x)
    if seg is None:
        seg = np.zeros(T, dtype=np.int64)
    seg = np.asarray(seg, dtype=np.int64)
    N = len(orders)
    t_idx = np.arange(T)
    xu = x.astype(np.uint64) + np.uint64(1)
    su = seg.astype(np.uint64) + np.uint64(1)

    succ = np.full((N, T), -1, dtype=np.int64)
    len_raw = np.zeros((N, T), dtype=np.int64)
    dist = np.zeros((N, T), dtype=np.int64)
    agree = np.zeros((N, T), dtype=np.int64)
    prev_ok = np.full((N, T), -1, dtype=np.int64)       # verified previous occurrence (ICE-V / VR candidates)

    for j, n in enumerate(orders):
        if n > T:
            continue
        # rolling polynomial hash of x[t-n+1 .. t], mixed with the segment id
        h = np.zeros(T, dtype=np.uint64)
        for i in range(n):
            sh = np.zeros(T, dtype=np.uint64)
            sh[i:] = xu[: T - i]
            h = h * _P + sh
        h = h * _P + su
        valid = np.zeros(T, dtype=bool)
        valid[n - 1:] = seg[n - 1:] == seg[: T - n + 1]   # window inside one document
        p = _prev_occurrence(h, valid)

        has = p >= 0
        if not has.any():
            continue
        # guard against hash collisions: verify the n tokens exactly
        tt, pp = t_idx[has], p[has]
        ok = np.ones(len(tt), dtype=bool)
        for i in range(n):
            ok &= x[tt - i] == x[pp - i]
        has_idx = tt[ok]
        p_ok = pp[ok]

        # true match length: extend backwards up to MAX_MATCH, staying inside the document
        length = np.zeros(len(has_idx), dtype=np.int64)
        alive = np.ones(len(has_idx), dtype=bool)
        for k in range(MAX_MATCH):
            pk = p_ok - k
            tk = has_idx - k
            chk = alive & (pk >= 0)
            m = np.zeros_like(chk)
            ci = np.nonzero(chk)[0]
            m[ci] = (x[tk[ci]] == x[pk[ci]]) & (seg[pk[ci]] == seg[has_idx[ci]])
            length += m
            alive = m
        succ[j, has_idx] = x[p_ok + 1]
        prev_ok[j, has_idx] = p_ok
        len_raw[j, has_idx] = length
        dist[j, has_idx] = has_idx - p_ok
        # agreement with the occurrence before that one (same n-gram => p's own previous occurrence)
        p2 = p[p_ok]
        ok2 = p2 >= 0
        for i in range(n):                                        # collision guard, as above
            ok2 &= x[np.where(ok2, p_ok - i, 0)] == x[np.where(ok2, p2 - i, 0)]
        a = np.ones(len(has_idx), dtype=np.int64)
        a[ok2] = np.where(x[p2[ok2] + 1] == x[p_ok[ok2] + 1], 2, 3)
        agree[j, has_idx] = a

    len_b = _len_bucket(len_raw)
    dist_b = _dist_bucket(dist)

    # longest match: max true length; ties -> higher order
    score = len_raw * (N + 1) + np.arange(N)[:, None]
    score = np.where(len_raw > 0, score, -1)
    best = np.argmax(score, axis=0)
    match_len = len_raw[best, t_idx]
    s_star = np.where(match_len > 0, succ[best, t_idx], -1)
    len_star = _len_bucket(match_len)
    agree_star = np.where(match_len > 0, agree[best, t_idx], 0)
    dist_star = np.where(match_len > 0, dist[best, t_idx], 0)

    rel = np.stack([_reliability(succ[j], x, seg) for j in range(N)]) if N else np.zeros((0, T), np.int64)
    rel_star = _reliability(s_star, x, seg)

    out = dict(
        ice_succ=succ, ice_len=len_b, ice_dist=dist_b, ice_agree=agree, ice_rel=rel,
        ice_s_star=s_star, ice_len_star=len_star, ice_agree_star=agree_star, ice_rel_star=rel_star,
        ice_match_len=match_len, ice_dist_star=dist_star,
    )
    if cand:
        out.update(_candidates(x, seg, cand, tuple(orders), prev_ok, len_raw, best, match_len > 0))
    return out


def segments_from_eos(x: np.ndarray, eos_id: Optional[int]) -> np.ndarray:
    """Segment ids for a packed window: a new document starts after each EOS token."""
    if eos_id is None:
        return np.zeros(len(x), dtype=np.int64)
    is_eos = (np.asarray(x) == eos_id).astype(np.int64)
    seg = np.concatenate([[0], np.cumsum(is_eos)[:-1]])
    return seg


def brute_force_successor(x: Sequence[int], n: int, seg: Optional[Sequence[int]] = None) -> np.ndarray:
    """Slow reference used by the tests."""
    x = list(x)
    T = len(x)
    seg = [0] * T if seg is None else list(seg)
    out = np.full(T, -1, dtype=np.int64)
    for t in range(n - 1, T):
        if seg[t - n + 1] != seg[t]:
            continue
        for p in range(t - 1, n - 2, -1):
            if seg[p - n + 1] == seg[p] == seg[t] and x[p - n + 1: p + 1] == x[t - n + 1: t + 1]:
                out[t] = x[p + 1]
                break
    return out
