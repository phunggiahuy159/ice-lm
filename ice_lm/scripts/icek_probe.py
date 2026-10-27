#!/usr/bin/env python
"""Run 6 of decision2.md: the ICE-K probe on a FROZEN G_VR (backbone and trained pointer frozen as the start).

    python scripts/icek_probe.py --base_run /vol/runs/t200_G_VR --data_dir /vol/data \
        --out /vol/diag/icek_probe.json --tokens 1.2e7

Two scorers are refitted on the same batches (same windows, same loss, same steps), both starting from G_VR's
trained ICEPointer (b, L, D, gamma, w, c0, beta):
  ctrl   the pointer refitted without keys (the control the go rule is measured against)
  keys   the pointer + a 16-byte write-time key:  s_k += tau * (q_t . k_{i_k - 1}) / sqrt(16)
         q_t = W_q RMSNorm(h_t),  k_i = Q8(tanh(W_k RMSNorm(h_i))) (int8 fake-quant, straight-through, value/127)
         h = G_VR's final normed hidden state; i_k = the candidate's source position (c_k = x[i_k]), so the key is
         the state that was about to emit c_k (i_k - 1 <= t - 1 < t: causal; asserted). W_q = 0 at init, so the
         arm starts equal to ctrl. 2 x 768 x 16 + 1 parameters.
Loss (both): mixture NLL -log(pi_0 p_bb(y) + sum_{c_k = y} pi_k) + lambda * listwise slot CE (positions with >= 2
distinct candidate tokens and y in the set), over code positions of TRAIN code windows (rule_oracle2.code_doc_mask,
training-stream order of data.TrainWindows with --seed). Only the pointer / key parameters train.
Held-out (first --val_windows val_code windows, all positions): mixture NLL of G_VR as is (ref; must match its code
loss), ctrl and keys, paired bootstrap over windows; token-level slot rank-1 (largest summed slot probability = y)
on ambiguous s*-wrong positions (unconditional, oracle ceiling ~0.34; and with y in the set). CodeRecall (focus 4
categories, --n_cr items per gap, never trained on): slot rank-1 and deciding-token accuracy.
Go rule (decision2 run 6): keys - ctrl >= +0.10 rank-1 on s*-wrong code AND >= 0.005 nats paired held-out loss.

--smoke: random tiny model with G_VR's layout (plumbing check on CPU; numbers meaningless).
"""
import argparse
import copy
import dataclasses
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, IterableDataset  # noqa: E402

from ice_lm.config import model_config_from_dict  # noqa: E402
from ice_lm.data import V_KEYS, alpha_mask_for_data, load_meta, open_tokens  # noqa: E402
from ice_lm.evals import _autocast  # noqa: E402
from ice_lm.ice_features import MAX_MATCH, segments_from_eos, set_alpha_mask  # noqa: E402
from ice_lm.model import ICELM, V_DIST_MAX, _NEG  # noqa: E402
from ice_lm.utils import pick_precision  # noqa: E402
from rule_oracle2 import K, code_doc_mask, codeish_tokens, vr_features  # noqa: E402

FOCUS = ("override_diff", "conflict_g2", "conflict_g5", "copy_diff")


def example(w, cm, eos):
    x, y = w[:-1], w[1:]
    f = vr_features(x, segments_from_eos(x, eos))
    d = {k: torch.from_numpy(np.ascontiguousarray(f[k].T)) for k in V_KEYS + ("ice_v_src",)}   # [T,K]
    d.update(x=torch.from_numpy(x), y=torch.from_numpy(y), cm=torch.from_numpy(cm),
             s_star=torch.from_numpy(f["ice_s_star"]))
    return d


class CodeWindows(IterableDataset):
    """Training-stream code windows (data.TrainWindows order, seed), sharded over DataLoader workers; or, with
    split="val_code", the val_code windows after the held-out ones (fallback when train.bin is absent)."""

    def __init__(self, data_dir, split, seq_len, seed, alpha, codeish, skip):
        self.data_dir, self.split, self.L, self.seed = data_dir, split, seq_len, seed
        self.alpha, self.codeish, self.skip = alpha, codeish, skip
        self.eos = load_meta(data_dir)["eos_id"]

    def __iter__(self):
        set_alpha_mask(self.alpha)
        data = open_tokens(self.data_dir, self.split)
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        L = self.L
        n_win = max(1, (len(data) - 1) // L - 1)
        if self.split == "val_code":
            order = lambda i: (self.skip + i % (n_win - self.skip)) * L          # noqa: E731
        else:
            cache = {}

            def order(i):
                epoch, k = divmod(i, n_win)
                if cache.get("epoch") != epoch:
                    prng = np.random.default_rng([self.seed, 7919, epoch])
                    cache.update(epoch=epoch, perm=prng.permutation(n_win),
                                 shift=0 if epoch == 0 else int(prng.integers(0, L)))
                return int(cache["perm"][k]) * L + cache["shift"]
        i = wid
        while True:
            off = order(i)
            w = np.asarray(data[off: off + L + 1]).astype(np.int64)
            cm = code_doc_mask(w[:-1], self.eos, self.codeish)
            if cm.mean() >= 0.5:
                yield example(w, cm, self.eos)
            i += nw


def collate(ex):
    return {k: torch.stack([e[k] for e in ex]) for k in ex[0]}


class Keys(nn.Module):
    def __init__(self, d, dk):
        super().__init__()
        self.d, self.dk = d, dk
        self.q = nn.Linear(d, dk, bias=False)
        self.k = nn.Linear(d, dk, bias=False)
        self.tau = nn.Parameter(torch.ones(1))
        nn.init.zeros_(self.q.weight)
        nn.init.normal_(self.k.weight, 0.0, 1.0 / math.sqrt(d))

    def forward(self, h, src):
        """h [B,T,d] (fp32), src [B,T,K] (-1 empty) -> additive slot scores [B,T,K]."""
        B, T, _ = h.shape
        hn = F.rms_norm(h, (self.d,))
        k = torch.tanh(self.k(hn))
        k = k + (torch.round(k * 127) / 127 - k).detach()                       # int8 fake-quant (STE)
        idx = (src - 1).clamp(min=0)
        t = torch.arange(T, device=src.device)[None, :, None]
        assert ((src - 1 < t) | (src < 0)).all(), "key from the future"
        kg = k.gather(1, idx.reshape(B, T * K, 1).expand(-1, -1, self.dk)).view(B, T, K, self.dk)
        return self.tau * (self.q(hn)[:, :, None] * kg).sum(-1) / math.sqrt(self.dk)


def slot_scores(P, h, ell, lmax, tok, ln, dist):
    """ICEPointer.log_pi's scores from precomputed ell = l[c_k] (clamped) and lmax = max l: s [B,T,K], s0 [B,T]."""
    sid = torch.arange(P.K, device=tok.device)
    s = (P.b.weight[:, 0] + P.L(sid * (MAX_MATCH + 1) + ln.clamp(0, MAX_MATCH)).squeeze(-1)
         + P.D(dist.clamp(max=V_DIST_MAX)).squeeze(-1)).float()
    s0 = P.w(P.norm(h)).float().squeeze(-1) + P.c0.float()
    if P.verify:
        s = s + P.gamma.float() * ell
        s0 = s0 + P.beta.float() * lmax
    return s, s0


@torch.no_grad()
def backbone(model, x, tok, y, amp, dev, chunk=1024):
    """Frozen G_VR: h (fp32 [B,T,d]), ell [B,T,K], ly [B,T], lmax [B,T]."""
    assert model.ice_in is None and model.ice_out is None
    with _autocast(dev, amp):
        h = model.embed(x)
        for blk in model.blocks:
            h = blk(h)
        h = model.norm(h)
    h = h.float()
    W = model.embed.weight if model.lm_head is None else model.lm_head.weight
    ell, ly, lmax = [], [], []
    for a in range(0, x.shape[1], chunk):
        with _autocast(dev, amp):
            z = F.linear(h[:, a:a + chunk].to(W.dtype if amp is None else amp), W).float()
        ls = z.log_softmax(-1)
        ell.append(ls.gather(-1, tok[:, a:a + chunk].clamp(min=0)).clamp(min=-20.0))
        ly.append(ls.gather(-1, y[:, a:a + chunk, None]).squeeze(-1))
        lmax.append(ls.amax(-1))
    return h, torch.cat(ell, 1), torch.cat(ly, 1), torch.cat(lmax, 1)


def heads(P, keys, h, ell, lmax, tok, ln, dist, src):
    s, s0 = slot_scores(P, h, ell, lmax, tok, ln, dist)
    if keys is not None:
        s = s + keys(h, src)
    return s.masked_fill(tok < 0, float("-inf")), s0


def losses(s, s0, ly, tok, y):
    """Per position: mixture NLL [B,T], listwise slot CE [B,T] and its mask (>= 2 distinct, y in the set)."""
    valid = tok >= 0
    lp = torch.cat([s0[..., None], s], -1).log_softmax(-1)
    hit = valid & (tok == y[..., None])
    lhit = torch.where(hit, lp[..., 1:].clamp(min=_NEG), torch.full_like(s, _NEG)).logsumexp(-1)
    nll = -torch.logaddexp(lp[..., 0] + ly, lhit)
    first = tok.gather(-1, valid.float().argmax(-1, keepdim=True))
    amb = (valid & (tok != first)).any(-1)
    lw_mask = amb & hit.any(-1)
    sm = s.masked_fill(~valid, _NEG)
    lw = sm.logsumexp(-1) - sm.masked_fill(~hit, _NEG).logsumexp(-1)
    return nll, torch.where(lw_mask, lw, torch.zeros_like(lw)), lw_mask, amb, hit


def rank1(s, tok, hit):
    """Token-level top-1 over the slots: summed slot probability of y > that of any other token."""
    valid = tok >= 0
    p = s.masked_fill(~valid, float("-inf")).softmax(-1).nan_to_num(0.0)
    same = (tok[..., :, None] == tok[..., None, :]) & valid[..., :, None] & valid[..., None, :]
    ptok = (same.float() @ p[..., None]).squeeze(-1)
    py = (p * hit).sum(-1)
    return py > ptok.masked_fill(hit | ~valid, -1.0).amax(-1)


def boot(num, den, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(den), (n, len(den)))
    est = num[idx].sum(1) / np.maximum(1, den[idx].sum(1))
    return float(num.sum() / max(1, den.sum())), [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_run", default="/vol/runs/t200_G_VR")
    ap.add_argument("--data_dir", default="/vol/data")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--out", default="/vol/diag/icek_probe.json")
    ap.add_argument("--tokens", type=float, default=1.2e7, help="training window tokens (code windows)")
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=0, help="0 = the base run's")
    ap.add_argument("--dk", type=int, default=16)
    ap.add_argument("--lr_ptr", type=float, default=3e-3)
    ap.add_argument("--lr_key", type=float, default=5e-3)
    ap.add_argument("--listwise", type=float, default=0.25)
    ap.add_argument("--val_windows", type=int, default=200)
    ap.add_argument("--n_cr", type=int, default=50, help="CodeRecall items per (category, gap); 0 = skip")
    ap.add_argument("--gaps", type=int, nargs="+", default=[256, 1024, 3072])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num_workers", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    def log(*s):
        print(f"[{time.time() - t0:6.0f}s]", *s, flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp, _ = pick_precision("auto", dev)
    torch.manual_seed(0)
    saved = json.load(open(os.path.join(a.base_run, "config.json")))
    cfg = model_config_from_dict(saved["model"])
    if a.smoke:
        cfg = dataclasses.replace(cfg, d_model=64, n_layers=2, n_heads=2, head_dim=32, n_kv_heads=1, mlp_hidden=128)
    model = ICELM(cfg).to(dev)
    model.prepare_backend(dev, amp or torch.float32)
    if not a.smoke:
        ck = torch.load(os.path.join(a.base_run, "model_final.pt"), map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert cfg.ice_v == "vr" and model.ice_ptr.K == K
    seq_len = a.seq_len or saved["train"]["seq_len"]
    meta = load_meta(a.data_dir)
    eos = meta["eos_id"]
    from transformers import AutoTokenizer
    tokzr = AutoTokenizer.from_pretrained(meta["tokenizer"])
    codeish = codeish_tokens(tokzr, meta["vocab_size"])
    alpha = alpha_mask_for_data(a.data_dir)
    set_alpha_mask(alpha)

    ref = model.ice_ptr
    arms = {"ctrl": [copy.deepcopy(ref), None], "keys": [copy.deepcopy(ref), Keys(cfg.d_model, a.dk).to(dev)]}
    opts = {}
    for name, (P, kk) in arms.items():
        for p in P.parameters():
            p.requires_grad_(True)
        groups = [dict(params=list(P.parameters()), lr=a.lr_ptr, base=a.lr_ptr)]
        if kk is not None:
            groups.append(dict(params=list(kk.parameters()), lr=a.lr_key, base=a.lr_key))
        opts[name] = torch.optim.Adam(groups)
    log(f"device {dev} amp {amp} seq_len {seq_len} key params {sum(p.numel() for p in arms['keys'][1].parameters())}")

    ds = CodeWindows(a.data_dir, a.train_split, seq_len, a.seed, alpha, codeish, a.val_windows)
    loader = DataLoader(ds, batch_size=a.micro_batch, collate_fn=collate, num_workers=a.num_workers,
                        prefetch_factor=4 if a.num_workers else None, persistent_workers=False)
    steps = math.ceil(a.tokens / (a.micro_batch * seq_len))
    it = iter(loader)
    n_code, hist = 0, []
    for step in range(1, steps + 1):
        frac = min(1.0, step / 20) * 0.5 * (1 + math.cos(math.pi * step / steps))
        b = {k: v.to(dev, non_blocking=True) for k, v in next(it).items()}
        tok, ln, dist, src = b["ice_v_tok"], b["ice_v_len"], b["ice_v_dist"], b["ice_v_src"]
        h, ell, ly, lmax = backbone(model, b["x"], tok, b["y"], amp, dev)
        cm = b["cm"].float()
        n_code += int(cm.sum())
        rec = {}
        for name, (P, kk) in arms.items():
            for g in opts[name].param_groups:
                g["lr"] = g["base"] * frac
            s, s0 = heads(P, kk, h, ell, lmax, tok, ln, dist, src)
            nll, lw, lw_mask, _, _ = losses(s, s0, ly, tok, b["y"])
            loss = (nll * cm).sum() / cm.sum() + a.listwise * (lw * cm).sum() / cm.sum()
            loss.backward()
            opts[name].step()
            opts[name].zero_grad(set_to_none=True)
            rec[name] = float((nll.detach() * cm).sum() / cm.sum())
        if step % 25 == 0 or step == 1 or step == steps:
            kk = arms["keys"][1]
            log(f"step {step}/{steps} nll ctrl {rec['ctrl']:.4f} keys {rec['keys']:.4f} "
                f"(keys - ctrl {rec['keys'] - rec['ctrl']:+.4f}) tau {kk.tau.item():.3f} "
                f"|Wq| {kk.q.weight.norm().item():.3f} code tokens {n_code:,}")
            hist.append(dict(step=step, **rec))
    del it, loader

    # ------------------------------------------------------------------ held-out val_code
    views = {"ref": (ref, None), "ctrl": tuple(arms["ctrl"]), "keys": tuple(arms["keys"])}
    data = open_tokens(a.data_dir, "val_code")
    n_val = min(a.val_windows, (len(data) - 1) // seq_len)
    W = n_val
    per = {v: dict(nll=np.zeros(W), r1=np.zeros(W), r1y=np.zeros(W)) for v in views}
    cnt = dict(tok=np.zeros(W), sw=np.zeros(W), swy=np.zeros(W))
    qk_abs = []
    check = None
    with torch.no_grad():
        for j in range(n_val):
            w = np.asarray(data[j * seq_len: j * seq_len + seq_len + 1]).astype(np.int64)
            b = {k: v[None].to(dev) for k, v in example(w, np.ones(seq_len, bool), eos).items()}
            tok, ln, dist, src, y = b["ice_v_tok"], b["ice_v_len"], b["ice_v_dist"], b["ice_v_src"], b["y"]
            h, ell, ly, lmax = backbone(model, b["x"], tok, y, amp, dev)
            sw = None
            for v, (P, kk) in views.items():
                s, s0 = heads(P, kk, h, ell, lmax, tok, ln, dist, src)
                nll, _, _, amb, hit = losses(s, s0, ly, tok, y)
                if sw is None:
                    sw = (amb & (b["s_star"] >= 0) & (b["s_star"] != y))[0]
                    swy = sw & hit[0].any(-1)
                    cnt["tok"][j], cnt["sw"][j], cnt["swy"][j] = seq_len, sw.sum().item(), swy.sum().item()
                r = rank1(s, tok, hit)[0]
                per[v]["nll"][j] = nll.sum().item()
                per[v]["r1"][j] = (r & sw).sum().item()
                per[v]["r1y"][j] = (r & swy).sum().item()
                if v == "keys":
                    qk_abs.append(kk(h, src).abs()[tok >= 0].mean().item())
                if v == "ref" and j == 0:              # the probe's mixture must equal the model's own forward
                    ice = {k: b[k].transpose(1, 2) for k in V_KEYS}
                    ice.update({k: torch.zeros(1, seq_len, dtype=torch.long, device=dev) for k in
                                ("ice_succ", "ice_len", "ice_dist", "ice_agree", "ice_rel")})
                    with _autocast(dev, amp):
                        _, lm = model(b["x"], y, ice=ice, reduction="none")
                    check = (float(lm.float().mean()), float(nll.mean()))
                    assert abs(check[0] - check[1]) < 2e-3, f"probe mixture != model forward: {check}"
            if (j + 1) % 50 == 0:
                log(f"val {j + 1}/{n_val}")
    res = dict(config=vars(a), steps=steps, code_tokens_trained=n_code, train_log=hist, check_model_vs_probe=check,
               mean_abs_key_score=float(np.mean(qk_abs)), tau=arms["keys"][1].tau.item(), val={})
    for v in views:
        res["val"][v] = dict(nll=per[v]["nll"].sum() / cnt["tok"].sum(), rank1_s_wrong=per[v]["r1"].sum() / cnt["sw"].sum(),
                             rank1_s_wrong_y_in_set=per[v]["r1y"].sum() / cnt["swy"].sum())
    res["val"]["n_tokens"] = int(cnt["tok"].sum())
    res["val"]["n_s_wrong_ambiguous"] = int(cnt["sw"].sum())
    g_loss, ci_loss = boot(per["ctrl"]["nll"] - per["keys"]["nll"], cnt["tok"])
    g_r1, ci_r1 = boot(per["keys"]["r1"] - per["ctrl"]["r1"], cnt["sw"])
    g_r1y, ci_r1y = boot(per["keys"]["r1y"] - per["ctrl"]["r1y"], cnt["swy"])
    g_ctrl, ci_ctrl = boot(per["ref"]["nll"] - per["ctrl"]["nll"], cnt["tok"])
    res["paired"] = dict(loss_gain_keys_vs_ctrl=g_loss, loss_gain_ci=ci_loss, rank1_gain=g_r1, rank1_gain_ci=ci_r1,
                         rank1_gain_y_in_set=g_r1y, rank1_gain_y_in_set_ci=ci_r1y,
                         loss_gain_ctrl_vs_ref=g_ctrl, loss_gain_ctrl_vs_ref_ci=ci_ctrl)
    res["go"] = dict(rule="keys - ctrl >= +0.10 rank-1 on s*-wrong code AND >= 0.005 paired loss",
                     rank1_ok=g_r1 >= 0.10, loss_ok=g_loss >= 0.005, go=bool(g_r1 >= 0.10 and g_loss >= 0.005))
    log("val", json.dumps(res["val"]), json.dumps(res["paired"]), res["go"], "check", check)

    # ------------------------------------------------------------------ CodeRecall (never trained on)
    if a.n_cr:
        from ice_lm.coderecall import CATEGORIES, build_items
        filler = open_tokens(a.data_dir, "val_code")
        res["coderecall"] = {}
        with torch.no_grad():
            for cat in FOCUS:
                acc = {v: dict(r1=0, dec=0) for v in views}
                n = 0
                for gap in a.gaps:
                    rng = np.random.default_rng([0, CATEGORIES.index(cat), gap])   # run_coderecall's items
                    seqs, _, decide = build_items(tokzr, rng, cat, a.n_cr, gap, filler, eos)
                    for seq, t1 in zip(seqs, decide):
                        x = np.asarray(seq[:-1], dtype=np.int64)
                        f = vr_features(x, np.zeros(len(x), dtype=np.int64))
                        r = t1 - 1
                        tt = {k: torch.from_numpy(np.ascontiguousarray(f[k].T)).to(dev)[None] for k in
                              V_KEYS + ("ice_v_src",)}
                        xt = torch.from_numpy(x).to(dev)[None]
                        yv = torch.full_like(xt, int(seq[t1]))
                        h, ell, ly, lmax = backbone(model, xt, tt["ice_v_tok"], yv, amp, dev)
                        Wt = model.embed.weight if model.lm_head is None else model.lm_head.weight
                        with _autocast(dev, amp):
                            lsr = F.linear(h[0, r].to(Wt.dtype if amp is None else amp), Wt).float().log_softmax(-1)
                        tk = tt["ice_v_tok"][:, r:r + 1]
                        for v, (P, kk) in views.items():
                            s, s0 = heads(P, kk, h, ell, lmax, tt["ice_v_tok"], tt["ice_v_len"], tt["ice_v_dist"],
                                          tt["ice_v_src"])
                            s, s0 = s[:, r:r + 1], s0[:, r:r + 1]
                            hit = (tk >= 0) & (tk == int(seq[t1]))
                            acc[v]["r1"] += int(rank1(s, tk, hit)[0, 0])
                            lp = torch.cat([s0[..., None], s], -1).log_softmax(-1)[0, 0]
                            mix = lsr + lp[0]
                            for kslot in torch.nonzero(tk[0, 0] >= 0)[:, 0].tolist():
                                c = tk[0, 0, kslot]
                                mix[c] = torch.logaddexp(mix[c], lp[1 + kslot])
                            acc[v]["dec"] += int(mix.argmax().item() == int(seq[t1]))
                        n += 1
                res["coderecall"][cat] = {v: dict(rank1=acc[v]["r1"] / n, dec_acc=acc[v]["dec"] / n) for v in views}
                res["coderecall"][cat]["n"] = n
                log(cat, json.dumps(res["coderecall"][cat]))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    log("saved", a.out)


if __name__ == "__main__":
    main()
