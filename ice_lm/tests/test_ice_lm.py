"""Correctness tests. Run:  python -m pytest -q tests"""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ice_lm.config import ARMS, build_configs, ice_param_count  # noqa: E402
from ice_lm.data import TrainWindows, collate, make_example, make_synthetic_data, split_batch  # noqa: E402
from ice_lm.gdn import gated_delta_rule_chunked, gated_delta_rule_recurrent  # noqa: E402
from ice_lm.ice_features import (DEFAULT_ORDERS, V_KEYS, brute_force_successor, compute_ice_features,  # noqa: E402
                                 n_slots, segments_from_eos, set_alpha_mask)
from ice_lm.inference import ICEIndex, features_to_tensors, step_features_to_tensors  # noqa: E402
from ice_lm.model import ICELM, kv_bytes_per_token, layer_plan  # noqa: E402

# ICE-VR identifier anchors: odd token ids count as "alphabetic" in these tests
ALPHA_TEST = np.arange(1 << 16) % 2 == 1
set_alpha_mask(ALPHA_TEST)
V_ARMS = ["G_V", "G_VR", "G_V_in", "G_V_nover", "C_V", "C_VR"]


# ------------------------------------------------------------------ ICE features
@pytest.mark.parametrize("seed", range(20))
def test_successor_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 4, size=150)
    seg = np.sort(rng.integers(0, 3, size=150)) if seed % 2 else None
    f = compute_ice_features(x, seg)
    for j, n in enumerate(DEFAULT_ORDERS):
        assert (f["ice_succ"][j] == brute_force_successor(x, n, seg)).all(), (seed, n)


def test_feature_details():
    x = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8])
    f = compute_ice_features(x)
    t = 8                                  # last position: "5 6 7 8" seen at positions 0..3
    j4 = DEFAULT_ORDERS.index(4)
    assert f["ice_succ"][j4, t] == 9
    assert f["ice_match_len"][t] == 4      # cannot extend: x[4]=9 vs x[-?] differ
    assert f["ice_s_star"][t] == 9
    assert f["ice_dist"][j4, t] == int(np.floor(np.log2(5))) + 1
    assert (f["ice_succ"][:, :5] == -1).all()   # nothing repeats in the first 5 tokens


def test_segments_block_cross_document_matches():
    eos = 0
    x = np.array([3, 4, 5, eos, 3, 4, 5])
    f = compute_ice_features(x, segments_from_eos(x, eos))
    assert (f["ice_succ"][:, 4:] == -1).all()


def test_agree_and_reliability_details():
    # "a b X ... a b Y ... a b" -> two earlier occurrences of (a b) with different successors: agree = 3
    x = np.array([1, 2, 7, 5, 1, 2, 8, 6, 1, 2])
    f = compute_ice_features(x)
    j2 = DEFAULT_ORDERS.index(2)
    assert f["ice_succ"][j2, 9] == 8 and f["ice_agree"][j2, 9] == 3
    assert f["ice_agree"][j2, 5] == 1                 # only one earlier occurrence
    # order-2 history at t=9: one prediction (t=5 said 7, truth 8) -> 0% right -> bucket 1
    assert f["ice_rel"][j2, 9] == 1
    assert f["ice_rel"][j2, 5] == 0                   # no resolved prediction yet


def _planted_sequence(seed, T=220, vocab=6, eos=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(1, vocab, size=T)
    x[120:170] = x[20:70]                             # a repeat longer than MAX_MATCH
    x[rng.integers(0, T, size=3)] = eos               # document boundaries
    return x


@pytest.mark.parametrize("cand", ["", "v", "vr"])
@pytest.mark.parametrize("seed", range(6))
def test_incremental_index_matches_batch_features(seed, cand):
    x = _planted_sequence(seed)
    ref = compute_ice_features(x, segments_from_eos(x, 0), DEFAULT_ORDERS, cand)
    idx = ICEIndex(eos_id=0, vocab_size=6, cand=cand)
    got = idx.extend(x[:90].tolist())
    rest = [idx.step(int(t)) for t in x[90:]]
    for k, v in ref.items():
        inc = np.concatenate([got[k], np.array([r[k] for r in rest]).T], axis=-1)
        assert (inc == v).all(), k
    assert idx.compact_bytes()["total"] > 0
    # vectorized prompt processing gives the same features and the same state to continue from
    idx2 = ICEIndex(eos_id=0, vocab_size=6, cand=cand)
    got2 = idx2.prefill(x[:90])
    rest2 = [idx2.step(int(t)) for t in x[90:]]
    for k, v in ref.items():
        inc = np.concatenate([got2[k], np.array([r[k] for r in rest2]).T], axis=-1)
        assert (inc == v).all(), ("prefill", k)
    assert idx2.tables == idx.tables and idx2.compact_bytes() == idx.compact_bytes()   # same memory accounting
    assert idx2.prev == idx.prev


def _brute_candidates(x, seg, cand, orders=DEFAULT_ORDERS, alpha=ALPHA_TEST):
    """Slow reference for the ICE-V / VR slots, straight from the definition in ice_features."""
    x, seg, T = list(x), list(seg), len(x)

    def occs(t, n):                       # earlier occurrences (end positions) of the key x[t-n+1..t], newest first
        if t - n + 1 < 0 or seg[t - n + 1] != seg[t]:
            return []
        return [p for p in range(t - 1, n - 2, -1) if seg[p - n + 1] == seg[p] == seg[t]
                and x[p - n + 1: p + 1] == x[t - n + 1: t + 1]]

    def ml(a, q):
        k = 0
        while k < 32 and q - k >= 0 and seg[q - k] == seg[a] and x[a - k] == x[q - k]:
            k += 1
        return k

    def db(d):
        return min(int(np.floor(np.log2(d))) + 1, 17)

    K = n_slots(cand, len(orders))
    out = {k: np.full((K, T), -1 if k == "ice_v_tok" else 0) for k in V_KEYS}
    for t in range(T):
        slots, last_len = [], []
        for n in orders:
            o = occs(t, n)
            slots += [(o[m] + 1, ml(t, o[m]), db(t - o[m])) if len(o) > m else None for m in range(2)]
            last_len.append(ml(t, o[0]) if o else 0)
        if cand == "vr":
            score = [ln * (len(orders) + 1) + j if ln > 0 else -1 for j, ln in enumerate(last_len)]
            o = occs(t, orders[int(np.argmax(score))]) if max(score) >= 0 else []
            slots += [(o[m - 1] + 1, ml(t, o[m - 1]), db(t - o[m - 1])) if len(o) >= m else None for m in range(3, 9)]
            s0 = seg.index(seg[t])
            anchors = [a for a in range(t, max(s0, t - 7) - 1, -1) if alpha[x[a]]][:2]
            for a in anchors + [-1] * (2 - len(anchors)):
                for n in (1, 2):
                    q = (occs(a, n) or [-1])[0] if a >= 0 else -1
                    slots += [(q + i, ml(a, q), db(a - q)) if q >= 0 and q + i <= t else None for i in range(1, 5)]
        for k, sl in enumerate(slots):
            if sl is not None:
                assert sl[0] <= t
                out["ice_v_tok"][k, t], out["ice_v_len"][k, t], out["ice_v_dist"][k, t] = x[sl[0]], sl[1], sl[2]
    return out


@pytest.mark.parametrize("cand", ["v", "vr"])
@pytest.mark.parametrize("seed", range(4))
def test_candidates_match_brute_force(seed, cand):
    x = _planted_sequence(seed)
    seg = segments_from_eos(x, 0)
    f = compute_ice_features(x, seg, DEFAULT_ORDERS, cand)
    ref = _brute_candidates(x, seg, cand)
    for k in V_KEYS:
        assert f[k].shape == (n_slots(cand, len(DEFAULT_ORDERS)), len(x))
        assert (f[k] == ref[k]).all(), k
    base = compute_ice_features(x, seg)                    # the ICE features themselves are unchanged
    assert all((base[k] == f[k]).all() for k in base)


def test_anchor_mask_must_be_set():
    x = _planted_sequence(0)
    set_alpha_mask(None)
    try:
        with pytest.raises(RuntimeError):
            compute_ice_features(x, None, DEFAULT_ORDERS, "vr")
        with pytest.raises(RuntimeError):
            ICEIndex(cand="vr")
    finally:
        set_alpha_mask(ALPHA_TEST)


def test_causality_of_features():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, size=300)
    f1 = compute_ice_features(x)
    x2 = x.copy(); x2[200:] = rng.integers(0, 5, size=100)
    f2 = compute_ice_features(x2)
    for k in f1:
        assert (np.asarray(f1[k])[..., :200] == np.asarray(f2[k])[..., :200]).all(), k


# ------------------------------------------------------------------ GDN kernel
@pytest.mark.parametrize("T,chunk", [(1, 16), (37, 16), (64, 64), (130, 32)])
def test_chunked_matches_recurrent(T, chunk):
    torch.manual_seed(T)
    B, H, K, V = 2, 3, 8, 5
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1)
    v = torch.randn(B, T, H, V)
    g = -torch.rand(B, T, H) * 0.5
    beta = torch.rand(B, T, H)
    ref = gated_delta_rule_recurrent(q, k, v, g, beta, K ** -0.5)
    out = gated_delta_rule_chunked(q, k, v, g, beta, K ** -0.5, chunk)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4), (out - ref).abs().max()


def test_chunked_gradients():
    torch.manual_seed(0)
    B, T, H, K = 1, 50, 2, 8
    q, k, v = (torch.randn(B, T, H, K, dtype=torch.float64, requires_grad=True) for _ in range(3))
    g = (-torch.rand(B, T, H, dtype=torch.float64) * 0.3).requires_grad_()
    beta = torch.rand(B, T, H, dtype=torch.float64).requires_grad_()
    qn = torch.nn.functional.normalize(q, dim=-1)
    kn = torch.nn.functional.normalize(k, dim=-1)
    a = gated_delta_rule_chunked(qn, kn, v, g, beta, 0.3, 16)
    b = gated_delta_rule_recurrent(qn, kn, v, g, beta, 0.3)
    ga = torch.autograd.grad(a.square().sum(), (q, k, v, g, beta), retain_graph=True)   # qn, kn are shared
    gb = torch.autograd.grad(b.square().sum(), (q, k, v, g, beta))
    for x, y in zip(ga, gb):
        assert torch.allclose(x.float(), y.float(), atol=1e-4, rtol=1e-3)


# ------------------------------------------------------------------ model
def _batch(cfg, B=2, T=64, seed=0, hi=None):
    rng = np.random.default_rng(seed)
    exs = []
    for _ in range(B):
        w = rng.integers(1, hi or cfg.vocab_size, size=T + 1)
        w[T // 2: T // 2 + 10] = w[3:13]      # create a copy
        ex = make_example(w, None, cfg.ice_orders, cfg.ice, cfg.ice_v)
        ex["ice_keep"] = torch.tensor(1.0)
        exs.append(ex)
    return collate(exs)


@pytest.mark.parametrize("arm", list(ARMS))
def test_forward_backward_all_arms(arm):
    mcfg, _ = build_configs("smoke", arm, vocab_size=300)
    model = ICELM(mcfg)
    x, y, ice, keep, _ = split_batch(_batch(mcfg), "cpu")
    _, loss = model(x, y, ice=ice, ice_keep=keep)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


@pytest.mark.parametrize("arm", ["B", "C"])
def test_chunked_training_loss_matches_full(arm):
    """The training path (chunked + recomputed head/loss) must equal the plain full-logits loss and gradients."""
    mcfg, _ = build_configs("smoke", arm, vocab_size=300)
    model = ICELM(mcfg)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    x, y, ice, keep, _ = split_batch(_batch(mcfg, T=100), "cpu")
    model.LOSS_CHUNK = 32                      # 100 tokens -> 4 uneven chunks
    _, loss_c = model(x, y, ice=ice, ice_keep=keep)
    g_c = torch.autograd.grad(loss_c, list(model.parameters()), allow_unused=True)
    logits = model(x, ice=ice, ice_keep=keep)
    loss_f = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    g_f = torch.autograd.grad(loss_f, list(model.parameters()), allow_unused=True)
    assert torch.allclose(loss_c, loss_f, atol=1e-5)
    for a, b in zip(g_c, g_f):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("arm", ["C", "C_notrust", "C_in", "C_out", "E"] + V_ARMS)
def test_ice_param_count(arm):
    mcfg, _ = build_configs("smoke", arm, vocab_size=300)
    assert ice_param_count(mcfg) == ICELM(mcfg).ice_params()


def test_param_matched_B():
    c, _ = build_configs("smoke", "C", vocab_size=300)
    b, _ = build_configs("smoke", "B", vocab_size=300)
    bpm, _ = build_configs("smoke", "B_pm", vocab_size=300)
    n_c, n_b, n_pm = (ICELM(m).num_params() for m in (c, b, bpm))
    assert n_b < n_pm and abs(n_pm - n_c) <= 3 * bpm.d_model * bpm.n_layers


def test_kv_budget_head_layout_matches_B():
    for preset in ("smoke", "140m", "0.6b"):
        b, _ = build_configs(preset, "B")
        d, _ = build_configs(preset, "D")
        a, _ = build_configs(preset, "A")
        assert kv_bytes_per_token(b) == kv_bytes_per_token(d)
        if preset != "smoke":            # 4-layer smoke model: A and B both have one attention layer
            assert kv_bytes_per_token(a) == 2 * kv_bytes_per_token(b)
        assert all(p["gdn"] + p["attn"] == d.n_heads for p in layer_plan(d))


@pytest.mark.parametrize("arm", ["B", "C", "E"])
def test_model_is_causal(arm):
    """Changing future tokens (and recomputing ICE features) must not change past logits."""
    mcfg, _ = build_configs("smoke", arm, vocab_size=300)
    model = ICELM(mcfg).eval()
    with torch.no_grad():
        for p in model.parameters():          # make ICE non-trivial
            p.add_(torch.randn_like(p) * 0.01)
    rng = np.random.default_rng(1)
    w = rng.integers(1, 300, size=97)
    w2 = w.copy(); w2[60:] = rng.integers(1, 300, size=37)
    outs = []
    for seq in (w, w2):
        b = collate([dict(make_example(seq, None, mcfg.ice_orders, mcfg.ice), ice_keep=torch.tensor(1.0))])
        x, _, ice, keep, _ = split_batch(b, "cpu")
        outs.append(model(x, ice=ice, ice_keep=keep))
    assert torch.allclose(outs[0][:, :59], outs[1][:, :59], atol=1e-5)


def test_ice_dropout_disables_ice():
    """keep=0 must give exactly the same logits as the same weights with ICE modules removed."""
    mcfg, _ = build_configs("smoke", "C", vocab_size=300)
    model = ICELM(mcfg).eval()
    with torch.no_grad():
        for p in model.ice_in.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        model.ice_out.w.weight.add_(1.0)
    x, y, ice, _, _ = split_batch(_batch(mcfg), "cpu")
    off = model(x, ice=ice, ice_keep=torch.zeros(x.shape[0]))
    on = model(x, ice=ice, ice_keep=torch.ones(x.shape[0]))
    ice_in, ice_out = model.ice_in, model.ice_out
    model.ice_in = model.ice_out = None
    base = model(x)
    model.ice_in, model.ice_out = ice_in, ice_out
    assert torch.allclose(off, base, atol=1e-5)
    assert not torch.allclose(on, base, atol=1e-3)


def test_train_windows_deterministic():
    with tempfile.TemporaryDirectory() as d:
        make_synthetic_data(d, train_tokens=50_000, val_tokens=5_000)
        a = TrainWindows(d, 64, DEFAULT_ORDERS, True, 0.2, seed=7)
        b = TrainWindows(d, 64, DEFAULT_ORDERS, True, 0.2, seed=7)
        ia, ib = iter(a), iter(b)
        for _ in range(5):
            ea, eb = next(ia), next(ib)
            assert torch.equal(ea["input_ids"], eb["input_ids"])
            assert torch.equal(ea["ice_succ"], eb["ice_succ"])
            assert ea["ice_keep"] == eb["ice_keep"]


def test_train_windows_global_order_and_resume():
    """Rank r / micro-batch m holds samples (m*world + r)*mb + j; resuming at a sample count continues exactly."""
    with tempfile.TemporaryDirectory() as d:
        make_synthetic_data(d, train_tokens=20_000, val_tokens=2_000)
        ref = TrainWindows(d, 32, DEFAULT_ORDERS, False, 0.0, seed=1)
        data = np.memmap(os.path.join(d, "train.bin"), dtype=np.uint16, mode="r")
        want = lambda i: ref.sample(data, i, {})["input_ids"]
        for rank in (0, 1):
            ds = TrainWindows(d, 32, DEFAULT_ORDERS, False, 0.0, seed=1, rank=rank, world=2, micro_batch=3)
            it = iter(ds)
            got = [next(it)["input_ids"] for _ in range(9)]           # local micro-batches 0, 1, 2
            for m in range(3):
                for j in range(3):
                    assert torch.equal(got[m * 3 + j], want((m * 2 + rank) * 3 + j))
        # resume after 2 optimizer micro-batches of both ranks (12 samples): rank 0 continues at micro-batch 2
        ds = TrainWindows(d, 32, DEFAULT_ORDERS, False, 0.0, seed=1, rank=0, world=2, micro_batch=3, start_index=12)
        assert torch.equal(next(iter(ds))["input_ids"], want((2 * 2 + 0) * 3))


def test_train_windows_cover_data_once_per_pass():
    with tempfile.TemporaryDirectory() as d:
        make_synthetic_data(d, train_tokens=20_000, val_tokens=2_000)
        ds = TrainWindows(d, 64, DEFAULT_ORDERS, False, 0.0, seed=3)
        n_win = (len(np.memmap(os.path.join(d, "train.bin"), dtype=np.uint16, mode="r")) - 1) // 64 - 1
        it = iter(ds)
        first = [tuple(next(it)["input_ids"].tolist()) for _ in range(n_win)]
        assert len(set(first)) == n_win              # no window repeats within a pass


@pytest.mark.parametrize("arm", ["A", "B", "C", "D", "E", "F", "C_notrust"] + V_ARMS)
def test_cached_decoding_matches_full_forward(arm):
    """prefill (37) + chunked prefill (50) + single decode steps == one full forward pass, every arm."""
    eos = 0
    mcfg, _ = build_configs("smoke", arm, vocab_size=64)
    model = ICELM(mcfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    rng = np.random.default_rng(3)
    x = rng.integers(1, 64, size=150)
    x[90:140] = x[10:60]
    x[[30, 100]] = eos
    feats = compute_ice_features(x, segments_from_eos(x, eos), mcfg.ice_orders, mcfg.ice_v)
    ice = {k: torch.from_numpy(feats[k])[None] for k in feats} if mcfg.ice else None
    with torch.no_grad():
        full = model(torch.from_numpy(x)[None], ice=ice)                   # [1,T,V]
        cache = model.allocate_cache(1, 160, "cpu", torch.float32)
        idx = ICEIndex(mcfg.ice_orders, eos_id=eos, vocab_size=64, cand=mcfg.ice_v)
        outs = []
        for a, b in ((0, 37), (37, 87)):
            f = features_to_tensors([idx.extend(x[a:b].tolist())], "cpu") if mcfg.ice else None
            outs.append(model.forward_cached(torch.from_numpy(x[a:b])[None], cache, f, last_only=False))
        for t in range(87, 150):
            f = step_features_to_tensors([idx.step(int(x[t]))], "cpu") if mcfg.ice else None
            outs.append(model.forward_cached(torch.from_numpy(x[t:t + 1])[None], cache, f))
    inc = torch.cat(outs, dim=1)
    assert cache.pos == 150
    assert torch.allclose(inc, full, atol=1e-4, rtol=1e-4), (inc - full).abs().max()


@pytest.mark.parametrize("arm", ["B", "E", "G_VR"])
def test_cached_decoding_batch2(arm):
    """Two different sequences decoded together (vectorized prompt prefill + steps) == full forward per row."""
    eos = 0
    mcfg, _ = build_configs("smoke", arm, vocab_size=64)
    model = ICELM(mcfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    rng = np.random.default_rng(5)
    xs = rng.integers(1, 64, size=(2, 120))
    xs[:, 70:110] = xs[:, 5:45]
    xs[1, 50] = eos
    fulls = []
    with torch.no_grad():
        for x in xs:
            feats = compute_ice_features(x, segments_from_eos(x, eos), mcfg.ice_orders, mcfg.ice_v)
            ice = {k: torch.from_numpy(feats[k])[None] for k in feats} if mcfg.ice else None
            fulls.append(model(torch.from_numpy(x)[None], ice=ice))
        full = torch.cat(fulls)
        cache = model.allocate_cache(2, 120, "cpu", torch.float32)
        idx = [ICEIndex(mcfg.ice_orders, eos_id=eos, vocab_size=64, cand=mcfg.ice_v) for _ in range(2)]
        f = features_to_tensors([ix.prefill(x[:64]) for ix, x in zip(idx, xs)], "cpu") if mcfg.ice else None
        outs = [model.forward_cached(torch.from_numpy(xs[:, :64]), cache, f, last_only=False)]
        for t in range(64, 120):
            f = step_features_to_tensors([ix.step(int(x[t])) for ix, x in zip(idx, xs)], "cpu") if mcfg.ice else None
            outs.append(model.forward_cached(torch.from_numpy(xs[:, t:t + 1]), cache, f))
    inc = torch.cat(outs, dim=1)
    assert torch.allclose(inc, full, atol=1e-4, rtol=1e-4), (inc - full).abs().max()


# ------------------------------------------------------------------ ICE-V / VR
def _v_model(arm, vocab=300, scale=0.3, seed=0):
    """A V model with non-trivial pointer weights (so candidates carry real mass)."""
    torch.manual_seed(seed)
    mcfg, _ = build_configs("smoke", arm, vocab_size=vocab)
    model = ICELM(mcfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
        for p in model.ice_ptr.parameters():
            p.add_(torch.randn_like(p) * scale)
        model.ice_ptr.b.weight.add_(3.0)                 # candidates win often
    return mcfg, model


def test_v_arm_configs():
    """Arm parameter-count sanity: the V arms are G plus exactly the pointer (+ the input module for G_V_in)."""
    g, _ = build_configs("smoke", "G", vocab_size=300)
    n_g = ICELM(g).num_params()
    n = {}
    for arm in V_ARMS:
        m, t = build_configs("smoke", arm, vocab_size=300)
        model = ICELM(m)
        n[arm] = model.num_params()
        assert t.ice_dropout == 0.0 and model.ice_out is None and model.ice_ptr is not None
        assert (model.ice_in is not None) == (arm == "G_V_in")
        assert model.ice_ptr.K == n_slots(m.ice_v, len(m.ice_orders))
        if arm.startswith("G_"):
            assert n[arm] == n_g + ice_param_count(m)
    k_v, k_vr = n_slots("v", 6), n_slots("vr", 6)
    assert (k_v, k_vr) == (12, 34)
    assert n["G_V"] - n["G_V_nover"] == k_v + 1                    # gamma[src], beta
    assert n["G_VR"] - n["G_V"] == (k_vr - k_v) * (1 + 33 + 1)    # b, L[., 0..32], gamma per extra slot
    assert ice_param_count(build_configs("140m", "G_V")[0]) < 2000   # ~1.3K parameters at 140m


@pytest.mark.parametrize("arm", ["G_V", "G_VR"])
def test_v_no_leakage(arm):
    """Perturb x[200:] (features recomputed): log-probs at positions <= 199 must be bit-identical."""
    mcfg, model = _v_model(arm)
    rng = np.random.default_rng(2)
    w = rng.integers(1, 12, size=257)                  # small alphabet: many repeats, anchor spans reach t+1
    w2 = w.copy(); w2[200:] = rng.integers(1, 12, size=57)
    outs, offs = [], []
    with torch.no_grad():
        for seq in (w, w2):
            b = collate([dict(make_example(seq, None, mcfg.ice_orders, True, mcfg.ice_v), ice_keep=torch.tensor(1.0))])
            x, _, ice, keep, _ = split_batch(b, "cpu")
            outs.append(model(x, ice=ice, ice_keep=keep))
            offs.append(model(x, ice=ice, ice_keep=torch.zeros(1)))
    assert torch.equal(outs[0][:, :200], outs[1][:, :200])
    assert not torch.equal(outs[0][:, :200], offs[0][:, :200])      # the pointer is active


@pytest.mark.parametrize("arm", ["G_V", "G_VR", "G_V_in"])
def test_v_keep0_is_backbone(arm):
    """keep=0 masks every slot: the output is exactly log_softmax(z). keep=1: a normalized distribution."""
    mcfg, model = _v_model(arm)
    x, _, ice, _, _ = split_batch(_batch(mcfg, T=96, hi=20), "cpu")
    with torch.no_grad():
        off = model(x, ice=ice, ice_keep=torch.zeros(x.shape[0]))
        on = model(x, ice=ice, ice_keep=torch.ones(x.shape[0]))
        ptr, model.ice_ptr = model.ice_ptr, None
        z = model(x, ice=ice, ice_keep=torch.zeros(x.shape[0]))       # backbone logits (input side off as well)
        model.ice_ptr = ptr
    assert torch.equal(off, torch.log_softmax(z, -1))
    assert torch.allclose(on.logsumexp(-1), torch.zeros(()), atol=1e-5)
    assert not torch.allclose(on, off, atol=1e-3)


@pytest.mark.parametrize("arm", ["G_V", "G_VR"])
def test_v_mixture_matches_sequential_definition(arm):
    """The eval mixture equals out = log pi_0 + l, then out[c_k] = logaddexp(out[c_k], log pi_k) slot by slot
    (duplicate candidates included)."""
    mcfg, model = _v_model(arm)
    x, _, ice, keep, _ = split_batch(_batch(mcfg, T=96, hi=20), "cpu")
    hs = []
    hook = model.norm.register_forward_hook(lambda mod, inp, out: hs.append(out))
    with torch.no_grad():
        mix = model(x, ice=ice, ice_keep=keep)
        ls = model(x, ice=ice, ice_keep=torch.zeros(x.shape[0]))
        hook.remove()
        h = hs[0]                                       # h_last
        tok, ln, dist = (ice[k].transpose(1, 2) for k in V_KEYS)
        lp, _ = model.ice_ptr.log_pi(h, ls, keep, tok, ln, dist)
        ref = ls + lp[..., :1]
        for k in range(tok.shape[-1]):
            idx = tok[..., k:k + 1].clamp(min=0)
            ref = ref.scatter(-1, idx, torch.logaddexp(ref.gather(-1, idx), lp[..., 1 + k:2 + k]))
    dup = ((tok.unsqueeze(-1) == tok.unsqueeze(-2)) & (tok >= 0).unsqueeze(-1)).sum() > tok.ge(0).sum()
    assert dup                                          # the test covers duplicate candidates
    assert torch.allclose(mix, ref, atol=1e-5)


@pytest.mark.parametrize("arm", ["G_V", "G_VR", "G_V_nover", "G_V_in"])
def test_v_training_loss_is_mixture_nll(arm):
    """Chunked training loss == mean(-log p_mix(y)) + aux * CE(z, y) of the eval paths, values and gradients."""
    mcfg, model = _v_model(arm, scale=0.1)
    model.train()
    x, y, ice, keep, _ = split_batch(_batch(mcfg, T=100, hi=20), "cpu")
    model.LOSS_CHUNK = 32                      # 100 tokens -> 4 uneven chunks
    _, loss_c = model(x, y, ice=ice, ice_keep=keep)
    params = list(model.parameters())
    g_c = torch.autograd.grad(loss_c, params, allow_unused=True)
    mix = model(x, ice=ice, ice_keep=keep)
    ptr, model.ice_ptr = model.ice_ptr, None
    z = model(x, ice=ice, ice_keep=keep)                 # backbone logits (G_V_in: with its input side)
    model.ice_ptr = ptr
    loss_f = (-mix.gather(-1, y.unsqueeze(-1)).mean()
              + mcfg.ice_v_aux * torch.nn.functional.cross_entropy(z.reshape(-1, z.size(-1)), y.reshape(-1)))
    g_f = torch.autograd.grad(loss_f, params, allow_unused=True)
    assert torch.allclose(loss_c, loss_f, atol=1e-5)
    for a, b in zip(g_c, g_f):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.isfinite(a).all()
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
