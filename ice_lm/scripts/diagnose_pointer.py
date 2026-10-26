#!/usr/bin/env python
"""Diagnostics for the ICE pointer arms (t200 runs): why the pointer does / does not pick the right candidate.

    python scripts/diagnose_pointer.py --runs_dir /vol/runs --prefix t200_ --data_dir /vol/data \
        --out /vol/diag/diag.json [--parts cr,incr,heldout] [--n_cr 100] [--windows 200]

Read-only on the runs. Parts:
  cr       CodeRecall items (exactly the items run_coderecall builds: same seeds, the first n_cr per gap), scored
           at the deciding token: pointer mass on sentinel / correct / wrong slots, rank of the correct slot, which
           slot sources hold the correct and the lookup's (wrong) value, backbone log p of both, score decomposition
           (bias b+L+D vs verifier gamma*l), G_C with its input / output side switched off separately.
  incr     `increment` and control variants (step 2, decreasing, constant id, random ids): does G_C compute s*+1?
  heldout  val_code windows (as eval_loss_splits): p(wrong lookup) / p(target) on lk_wrong, calibration of the copy
           mass 1-pi0 vs "some candidate == target" (AUROC, reliability bins), backbone-only loss of G_V/G_VR vs G
           by token group and copy distance (laziness), per-slot fill / precision / mass, AUROC of slot score terms.
  params   pointer parameters (gamma, beta, c0, b, L, D).
Views: B, G, G_C, G_C_off (keep 0), G_C_in (input side only), G_C_out (output side only), G_V, G_V_bb (its
backbone log_softmax(z) = G_V_off), G_VR, G_VR_bb.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ice_lm import coderecall  # noqa: E402
from ice_lm.coderecall import CATEGORIES, build_items  # noqa: E402
from ice_lm.config import model_config_from_dict  # noqa: E402
from ice_lm.data import alpha_mask_for_data, load_meta, model_ice_keys, open_tokens  # noqa: E402
from ice_lm.evals import _autocast  # noqa: E402
from ice_lm.ice_features import (CHAIN_OCC, V_KEYS, _prev_occurrence, compute_ice_features,  # noqa: E402
                                 segments_from_eos, set_alpha_mask)
from ice_lm.model import ICELM, MAX_MATCH, V_DIST_MAX  # noqa: E402
from ice_lm.utils import pick_precision  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs_dir", default="/vol/runs")
ap.add_argument("--prefix", default="t200_")
ap.add_argument("--arms", default="B,G,G_C,G_V,G_VR")
ap.add_argument("--data_dir", default="/vol/data")
ap.add_argument("--out", default="/vol/diag/diag.json")
ap.add_argument("--parts", default="params,cr,incr,heldout")
ap.add_argument("--cr_cats", default="override_diff,conflict_g2,conflict_g5,copy_diff,copy_same,conflict_g1,"
                                    "override_same,increment")
ap.add_argument("--n_cr", type=int, default=100, help="items per (category, gap) for the 4 focus categories")
ap.add_argument("--n_cr_contrast", type=int, default=40, help="items per (category, gap) for the others")
ap.add_argument("--n_incr", type=int, default=100)
ap.add_argument("--windows", type=int, default=200)
ap.add_argument("--seq_len", type=int, default=4096)
ap.add_argument("--n_examples", type=int, default=2, help="printed example items per category")
a = ap.parse_args()

FOCUS = ("override_diff", "conflict_g2", "conflict_g5", "copy_diff")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp, _ = pick_precision("auto", dev)
meta = load_meta(a.data_dir)
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
set_alpha_mask(alpha_mask_for_data(a.data_dir))
EOS = meta["eos_id"]
VOCAB = meta["vocab_size"]
DIG = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
NEXT = np.full(VOCAB + 1, -1, dtype=np.int64)      # digit token -> token of digit + 1 (index -1 -> -1)
PREV = np.full(VOCAB + 1, -1, dtype=np.int64)
ISDIG = np.zeros(VOCAB + 1, dtype=bool)
for d in range(10):
    ISDIG[DIG[d]] = True
    if d < 9:
        NEXT[DIG[d]] = DIG[d + 1]
    if d > 0:
        PREV[DIG[d]] = DIG[d - 1]
OUT = {}
T0 = time.time()


def log(*s):
    print(f"[{time.time() - T0:6.0f}s]", *s, flush=True)


# ---------------------------------------------------------------------------- models
def load(arm):
    run = os.path.join(a.runs_dir, a.prefix + arm)
    cfg = json.load(open(os.path.join(run, "config.json")))
    mcfg = model_config_from_dict(cfg["model"])
    m = ICELM(mcfg).to(dev)
    ck = torch.load(os.path.join(run, "model_final.pt"), map_location=dev, weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    m.prepare_backend(dev, amp or torch.float32)
    return m


ARMS = a.arms.split(",")
MODELS = {arm: load(arm) for arm in ARMS}
log("loaded", ARMS)

K_VR = 34
GROUPS = ("last", "second", "chain", "anchor")


def slot_meta(K, orders=(1, 2, 3, 4, 6, 8)):
    labels, gid = [], []
    for n in orders:
        labels += [f"o{n}_last", f"o{n}_2nd"]
        gid += [0, 1]
    if K > 2 * len(orders):
        for o in CHAIN_OCC:
            labels.append(f"chain{o}")
            gid.append(2)
        for r in range(2):
            for n in (1, 2):
                for i in range(1, 5):
                    labels.append(f"anc{r}_k{n}_+{i}")
                    gid.append(3)
    return labels, np.array(gid)


def ice_tensors(f, model):
    cand = model.cfg.ice_v
    keys = model_ice_keys(cand)
    K = model.ice_ptr.K if model.ice_ptr is not None else None
    d = {}
    for k in keys:
        v = f[k]
        if k in V_KEYS:
            v = v[:K]
        d[k] = torch.from_numpy(np.ascontiguousarray(v)).to(dev)[None]
    return d


@torch.no_grad()
def hidden(model, x, ice, keep_in=1.0):
    with _autocast(dev, amp):
        h = model.embed(x)
        for i, blk in enumerate(model.blocks):
            h = blk(h)
            if model.ice_in is not None and i + 1 == model.cfg.ice_inject_after:
                h = model.ice_in(h, model.embed, ice["ice_succ"], ice["ice_len"], ice["ice_dist"],
                                 torch.full((1,), keep_in, device=dev), ice.get("ice_agree"), ice.get("ice_rel"))
        return model.norm(h)


@torch.no_grad()
def views(x_np, f, rows):
    """All views at the given rows -> dict name -> dict (lf [R,V] final log-probs, + pointer tensors)."""
    x = torch.from_numpy(np.asarray(x_np, dtype=np.int64)).to(dev)[None]
    rows_t = torch.as_tensor(rows, device=dev)
    one = torch.ones(1, device=dev)
    zero = torch.zeros(1, device=dev)
    out = {}
    for arm, m in MODELS.items():
        W = m.embed.weight
        ice = ice_tensors(f, m) if m.uses_ice() else None
        if m.ice_ptr is None and m.ice_out is None:
            h = hidden(m, x, ice)[:, rows_t]
            with _autocast(dev, amp):
                out[arm] = dict(lf=m._logits(h, None).float().log_softmax(-1)[0])
            continue
        if m.ice_out is not None:                 # G_C: input / output side separately
            star = tuple(t[:, rows_t] for t in m._out_feats(ice))
            for kin, tag_on, tag_off in ((1.0, "", "_in"), (0.0, "_out", "_off")):
                h = hidden(m, x, ice, kin)[:, rows_t]
                with _autocast(dev, amp):
                    z = F.linear(h, W).float()
                    zo = m.ice_out(z, h, star[0], star[1], one, star[2], star[3])
                out[arm + tag_on] = dict(lf=zo.log_softmax(-1)[0], lam=(zo - z)[0].gather(
                    -1, star[0][0].clamp(min=0)[:, None])[:, 0])
                out[arm + tag_off] = dict(lf=z.log_softmax(-1)[0])
            continue
        P = m.ice_ptr
        h = hidden(m, x, ice)[:, rows_t]
        tk, ln, ds = (t.transpose(1, 2)[:, rows_t] for t in (ice[k] for k in V_KEYS))
        with _autocast(dev, amp):
            z = F.linear(h, W).float()
            ls = z.log_softmax(-1)
            lp, valid = P.log_pi(h, ls, one, tk, ln, ds)
            lf = P.mix(z, h, one, tk, ln, ds)
            src = torch.arange(P.K, device=dev)
            bias = (P.b.weight[:, 0] + P.L(src * (MAX_MATCH + 1) + ln.clamp(0, MAX_MATCH)).squeeze(-1)
                    + P.D(ds.clamp(max=V_DIST_MAX)).squeeze(-1)).float()
            ell = ls.gather(-1, tk.clamp(min=0)).clamp(min=-20.0)
            ver = P.gamma.float() * ell
            s0a = P.w(P.norm(h)).float().squeeze(-1) + P.c0.float()
            s0b = P.beta.float() * ls.amax(-1)
        out[arm] = dict(lf=lf[0], lp=lp[0], valid=valid[0], tok=tk[0], ln=ln[0], dist=ds[0], bias=bias[0],
                        ell=ell[0], ver=ver[0], s0a=s0a[0], s0b=s0b[0])
        out[arm + "_bb"] = dict(lf=ls[0])
    return out


# ---------------------------------------------------------------------------- params
if "params" in a.parts:
    res = {}
    for arm, m in MODELS.items():
        if m.ice_ptr is None:
            continue
        P = m.ice_ptr
        labels, _ = slot_meta(P.K)
        Lt = P.L.weight.detach().float().view(P.K, MAX_MATCH + 1).cpu().numpy()
        r = dict(gamma=dict(zip(labels, P.gamma.detach().float().cpu().tolist())),
                 beta=float(P.beta), c0=float(P.c0), b=dict(zip(labels, P.b.weight[:, 0].detach().float().cpu().tolist())),
                 L={lab: {str(l): float(Lt[k, l]) for l in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)}
                    for k, lab in enumerate(labels)},
                 D=P.D.weight[:, 0].detach().float().cpu().tolist(), w_norm=float(P.w.weight.norm()))
        res[arm] = r
        log(f"== params {arm}: beta={r['beta']:.3f} c0={r['c0']:.3f} |w|={r['w_norm']:.3f}")
        log("  gamma:", " ".join(f"{k}={v:.2f}" for k, v in r["gamma"].items()))
        log("  b:", " ".join(f"{k}={v:.2f}" for k, v in r["b"].items()))
        log("  b+L[len] for len 1/4/8/16/32:")
        for k, lab in enumerate(labels):
            log(f"    {lab:12s}", " ".join(f"{r['b'][lab] + Lt[k, l]:6.2f}" for l in (1, 4, 8, 16, 32)))
        log("  D:", " ".join(f"{v:.2f}" for v in r["D"]))
    OUT["params"] = res


# ---------------------------------------------------------------------------- CodeRecall
def item_record(v, y, sstar, labels, gid):
    """Per-item stats at one row (index 0 of every view)."""
    rec = {}
    for name, d in v.items():
        lf = d["lf"][0]
        r = dict(pred=int(lf.argmax()), lpy=float(lf[y]), lps=float(lf[sstar]) if sstar >= 0 else float("nan"),
                 rank_y=int((lf > lf[y]).sum()) + 1)
        if "lam" in d:
            r["lam"] = float(d["lam"][0])
        if "lp" in d:
            lp, valid, tk = d["lp"][0], d["valid"][0], d["tok"][0]
            pis = lp[1:].exp()
            K = len(tk)
            hit = valid & (tk == y)
            st = valid & (tk == sstar) if sstar >= 0 else torch.zeros_like(hit)
            r.update(pi0=float(lp[0].exp()), m_y=float(pis[hit].sum()), m_star=float(pis[st].sum()),
                     m_wrong=float(pis[valid & ~hit].sum()), in_set=bool(hit.any()), n_valid=int(valid.sum()))
            g = gid[:K]
            r["grp_y"] = [GROUPS[i] for i in sorted(set(g[hit.cpu().numpy()].tolist()))]
            r["grp_s"] = [GROUPS[i] for i in sorted(set(g[st.cpu().numpy()].tolist()))]
            lpk = lp[1:].masked_fill(~valid, float("-inf"))
            top = int(lpk.argmax())
            r.update(top_lab=labels[top], top_hit=bool(hit[top]), top_star=bool(st[top]))
            if hit.any():
                best = int(lpk.masked_fill(~hit, float("-inf")).argmax())
                r["rank_slot"] = int((lpk > lpk[best]).sum()) + 1
                r["best_lab"] = labels[best]
                r["best_bias"], r["best_ver"], r["best_len"] = float(d["bias"][0, best]), float(d["ver"][0, best]), \
                    int(d["ln"][0, best])
                r["best_ell"], r["best_dist"] = float(d["ell"][0, best]), int(d["dist"][0, best])
                wr = valid & ~hit
                if wr.any():
                    tw = int(lpk.masked_fill(~wr, float("-inf")).argmax())
                    r["tw_lab"], r["tw_bias"], r["tw_ver"], r["tw_len"] = labels[tw], float(d["bias"][0, tw]), \
                        float(d["ver"][0, tw]), int(d["ln"][0, tw])
                    r["tw_ell"], r["tw_dist"] = float(d["ell"][0, tw]), int(d["dist"][0, tw])
                # rule: the slot with the longest exact match (ties: lowest slot index)
                lnv = d["ln"][0].masked_fill(~valid, -1)
                r["maxlen_ok"] = bool(tk[int(lnv.argmax())] == y)
                # backbone rerank: among candidate tokens, which does the backbone like most?
                ell = d["ell"][0].masked_fill(~valid, float("-inf"))
                r["rerank_ok"] = bool(tk[int(ell.argmax())] == y)
            r["s0a"], r["s0b"] = float(d["s0a"][0]), float(d["s0b"][0])
        rec[name] = r
    return rec


def print_example(seq, t, y, sstar, v, labels):
    tail = tok.convert_ids_to_tokens(seq[max(0, t - 14): t])
    log(f"   ...{tail} -> y={tok.convert_ids_to_tokens([y])} s*={tok.convert_ids_to_tokens([sstar]) if sstar >= 0 else None}")
    for name in ("G_C", "G_C_in", "G_V", "G_VR"):
        if name not in v:
            continue
        lf = v[name]["lf"][0]
        top = torch.topk(lf, 3)
        log(f"     {name:7s} top3", [(tok.convert_ids_to_tokens([int(i)])[0], round(float(p.exp()), 3))
                                      for p, i in zip(top.values, top.indices)])
        d = v[name]
        if "lp" in d:
            lp, valid = d["lp"][0], d["valid"][0]
            o = torch.argsort(lp[1:].masked_fill(~valid, float("-inf")), descending=True)[:6]
            log(f"       pi0={float(lp[0].exp()):.3f} s0a={float(d['s0a'][0]):.2f} s0b={float(d['s0b'][0]):.2f}; top slots:")
            for k in o.tolist():
                if not bool(valid[k]):
                    break
                log(f"         {labels[k]:12s} tok={tok.convert_ids_to_tokens([int(d['tok'][0, k])])[0]!s:8s} "
                    f"pi={float(lp[1 + k].exp()):.3f} bias={float(d['bias'][0, k]):6.2f} ver={float(d['ver'][0, k]):6.2f} "
                    f"ell={float(d['ell'][0, k]):6.2f} len={int(d['ln'][0, k])} dist_b={int(d['dist'][0, k])}")


def summarize_cat(recs, names):
    out = {}
    for n in names:
        rs = [r[n] for r in recs if n in r]
        if not rs:
            continue
        s = dict(n=len(rs), acc=np.mean([r["pred"] == r["y"] for r in rs]),
                 p_y=np.mean([np.exp(r["lpy"]) for r in rs]),
                 p_star=np.nanmean([np.exp(r["lps"]) for r in rs]),
                 follow=np.mean([r["pred"] == r["sstar"] for r in rs]),
                 pred_plus1=np.mean([r["sstar"] >= 0 and r["pred"] == NEXT[r["sstar"]] for r in rs]),
                 lpy=np.mean([r["lpy"] for r in rs]), lps=np.nanmean([r["lps"] for r in rs]),
                 bb_y_gt_star=np.nanmean([r["lpy"] > r["lps"] for r in rs if r["sstar"] >= 0 and r["sstar"] != r["y"]]
                                         or [np.nan]),
                 med_rank_y=float(np.median([r["rank_y"] for r in rs])))
        if "pi0" in rs[0]:
            ins = [r for r in rs if r["in_set"]]
            s.update(pi0=np.mean([r["pi0"] for r in rs]), m_y=np.mean([r["m_y"] for r in rs]),
                     m_wrong=np.mean([r["m_wrong"] for r in rs]), m_star=np.mean([r["m_star"] for r in rs]),
                     in_set=np.mean([r["in_set"] for r in rs]), n_valid=np.mean([r["n_valid"] for r in rs]),
                     top_hit=np.mean([r["top_hit"] for r in rs]), top_star=np.mean([r["top_star"] for r in rs]),
                     s0a=np.mean([r["s0a"] for r in rs]), s0b=np.mean([r["s0b"] for r in rs]))
            if ins:
                s.update(rank1=np.mean([r["rank_slot"] == 1 for r in ins]),
                         med_rank=float(np.median([r["rank_slot"] for r in ins])),
                         rerank_ok=np.mean([r["rerank_ok"] for r in ins]),
                         best_bias=np.mean([r["best_bias"] for r in ins]), best_ver=np.mean([r["best_ver"] for r in ins]),
                         best_len=np.mean([r["best_len"] for r in ins]), best_ell=np.mean([r["best_ell"] for r in ins]),
                         best_dist=np.mean([r["best_dist"] for r in ins]),
                         maxlen_ok=np.mean([r["maxlen_ok"] for r in ins]))
                tw = [r for r in ins if "tw_bias" in r]
                if tw:
                    s.update(tw_bias=np.mean([r["tw_bias"] for r in tw]), tw_ver=np.mean([r["tw_ver"] for r in tw]),
                             tw_len=np.mean([r["tw_len"] for r in tw]), tw_ell=np.mean([r["tw_ell"] for r in tw]),
                             tw_dist=np.mean([r["tw_dist"] for r in tw]))
                    labs = {}
                    for r in tw:
                        labs[r["tw_lab"]] = labs.get(r["tw_lab"], 0) + 1
                    s["tw_labs"] = dict(sorted(labs.items(), key=lambda kv: -kv[1])[:5])
                labs = {}
                for r in ins:
                    labs[r["best_lab"]] = labs.get(r["best_lab"], 0) + 1
                s["best_labs"] = dict(sorted(labs.items(), key=lambda kv: -kv[1])[:5])
            for g in GROUPS:
                s[f"y_in_{g}"] = np.mean([g in r["grp_y"] for r in rs])
                s[f"s_in_{g}"] = np.mean([g in r["grp_s"] for r in rs])
        out[n] = {k: (float(v) if isinstance(v, (float, np.floating, np.bool_, bool, int, np.integer)) else v)
                  for k, v in s.items()}
    return out


def run_items(cat, seqs, decide, labels, gid, n_ex):
    recs = []
    for i, (seq, t) in enumerate(zip(seqs, decide)):
        x = np.asarray(seq[:-1], dtype=np.int64)
        if len(x) < a.seq_len:      # right-pad to one fixed shape (no kernel re-tuning per length); all causal
            x = np.concatenate([x, np.full(a.seq_len - len(x), EOS, dtype=np.int64)])
        f = compute_ice_features(x, None, (1, 2, 3, 4, 6, 8), "vr")
        y, sstar = int(seq[t]), int(f["ice_s_star"][t - 1])
        v = views(x, f, [t - 1])
        rec = item_record(v, y, sstar, labels, gid)
        for r in rec.values():
            r["y"], r["sstar"] = y, sstar
        recs.append(rec)
        if i < n_ex:
            log(f"  example {cat} #{i}")
            print_example(seq, t, y, sstar, v, labels)
    return recs


VIEW_ORDER = ("B", "G", "G_C", "G_C_off", "G_C_in", "G_C_out", "G_V", "G_V_bb", "G_VR", "G_VR_bb")
LABELS, GID = slot_meta(K_VR)


def print_cat(cat, s):
    log(f"== CR {cat}")
    log("   view       n   acc  p(y) p(s*) follow pred=s*+1  lp(y)  lp(s*) P[lp(y)>lp(s*)]")
    for n in VIEW_ORDER:
        if n in s:
            r = s[n]
            log(f"   {n:8s} {int(r['n']):4d} {r['acc']:.3f} {r['p_y']:.3f} {r['p_star']:.3f} {r['follow']:.3f}  "
                f"{r['pred_plus1']:.3f}   {r['lpy']:6.2f} {r['lps']:6.2f}  {r['bb_y_gt_star']:.3f}")
    for n in ("G_V", "G_VR"):
        if n in s:
            r = s[n]
            log(f"   {n} pointer: pi0={r['pi0']:.3f} m_y={r['m_y']:.3f} m_wrong={r['m_wrong']:.3f} m_s*={r['m_star']:.3f} "
                f"in_set={r['in_set']:.2f} n_valid={r['n_valid']:.1f} top_hit={r['top_hit']:.2f} top_s*={r['top_star']:.2f} "
                f"s0a={r['s0a']:.2f} s0b={r['s0b']:.2f}")
            if "rank1" in r:
                log(f"      | in-set items: rank1={r['rank1']:.2f} med_rank={r['med_rank']:.1f} "
                    f"backbone-rerank-ok={r['rerank_ok']:.2f}  best-correct bias={r['best_bias']:.2f} ver={r['best_ver']:.2f} "
                    f"len={r['best_len']:.1f} | top-wrong bias={r.get('tw_bias', float('nan')):.2f} "
                    f"ver={r.get('tw_ver', float('nan')):.2f} len={r.get('tw_len', float('nan')):.1f}")
                log(f"      | best-correct slots {r['best_labs']}  top-wrong slots {r.get('tw_labs')}")
                log(f"      | ell (backbone log p) best-correct={r['best_ell']:.2f} top-wrong={r.get('tw_ell', float('nan')):.2f}; "
                    f"dist bucket best-correct={r['best_dist']:.1f} top-wrong={r.get('tw_dist', float('nan')):.1f}; "
                    f"longest-match rule ok={r['maxlen_ok']:.2f}")
            log("      | y in groups: " + " ".join(f"{g}={r[f'y_in_{g}']:.2f}" for g in GROUPS)
                + " ; s* in groups: " + " ".join(f"{g}={r[f's_in_{g}']:.2f}" for g in GROUPS))


if "cr" in a.parts:
    filler = open_tokens(a.data_dir, "val_code")
    OUT["cr"] = {}
    for cat in a.cr_cats.split(","):
        n = a.n_cr if (cat in FOCUS or cat == "increment") else a.n_cr_contrast
        gaps = (256, 1024, 3072) if cat not in ("increment", "track") else (256,)
        recs = []
        for gap in gaps:
            rng = np.random.default_rng([0, CATEGORIES.index(cat), gap])       # = run_coderecall's items
            seqs, spans, decide = build_items(tok, rng, cat, n, gap, filler, EOS, 8)
            recs += run_items(f"{cat}@{gap}", seqs, decide, LABELS, GID, a.n_examples if gap == gaps[-1] else 0)
        s = summarize_cat(recs, VIEW_ORDER)
        OUT["cr"][cat] = s
        print_cat(cat, s)


# ---------------------------------------------------------------------------- increment variants
_orig_make_item = coderecall.make_item


def make_item_ext(category, rng, n_distract):
    if not category.startswith("incr_"):
        return _orig_make_item(category, rng, n_distract)
    kind, n_items = category[5:], 4
    names = list(rng.choice(coderecall.MIGRATIONS, size=n_items + 1, replace=False))
    if kind == "step2":
        base = int(rng.integers(100, 900)) * 10 + int(rng.integers(0, 2))
        ids = [base + 2 * i for i in range(n_items + 1)]
    elif kind == "dec":
        base = int(rng.integers(100, 900)) * 10 + int(rng.integers(4, 10))
        ids = [base - i for i in range(n_items + 1)]
    elif kind == "const":
        base = int(rng.integers(100, 900)) * 10 + int(rng.integers(0, 10))
        ids = [base] * (n_items + 1)
    elif kind == "rand":          # same leading digits, random distinct last digits (no rule)
        base = int(rng.integers(100, 900)) * 10
        ids = [base + int(d) for d in rng.choice(10, size=n_items + 1, replace=False)]
    else:
        raise ValueError(category)
    lines = "".join(f'    Migration(id={ids[i]}, name="{names[i]}"),\n' for i in range(n_items))
    return [None, "\nMIGRATIONS = [\n" + lines + "    Migration(id="], f"{ids[-1]}"


if "incr" in a.parts:
    coderecall.make_item = make_item_ext
    filler = open_tokens(a.data_dir, "val_code")
    OUT["incr"] = {}
    for cat in ("increment", "incr_step2", "incr_dec", "incr_const", "incr_rand"):
        seed_cat = CATEGORIES.index("increment")
        rng = np.random.default_rng([0, seed_cat, 256] if cat == "increment" else [1, len(cat), 256])
        seqs, spans, _ = build_items(tok, rng, cat, a.n_incr, 256, filler, EOS, 8)
        decide = [b - 1 for (_, b) in spans]                    # last digit
        recs = run_items(cat, seqs, decide, LABELS, GID, 1)
        s = summarize_cat(recs, VIEW_ORDER)
        OUT["incr"][cat] = s
        log(f"== INCR {cat}: acc / follow s* / pred = s*+1 / p(y)")
        for n in VIEW_ORDER:
            if n in s:
                r = s[n]
                log(f"   {n:8s} acc={r['acc']:.3f} follow={r['follow']:.3f} pred=s*+1:{r['pred_plus1']:.3f} p(y)={r['p_y']:.3f}"
                    + (f" pi0={r['pi0']:.3f} in_set={r['in_set']:.2f}" if "pi0" in r else ""))
    coderecall.make_item = _orig_make_item


# ---------------------------------------------------------------------------- held-out code
def auroc(score, label):
    score, label = np.asarray(score, np.float64), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    r = ((cum - cnt + 1 + cum) / 2.0)[inv]
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


if "heldout" in a.parts:
    data = open_tokens(a.data_dir, "val_code")
    L = a.seq_len
    acc = {}           # name -> list of dicts of per-position numpy arrays
    shared = []
    slot_acc = {}      # arm -> dict of per-slot counters
    sub = {}           # arm -> per-filled-slot samples (every 4th position)
    checked = False
    for j in range(a.windows):
        w = np.asarray(data[j * L: j * L + L + 1]).astype(np.int64)
        x, y = w[:-1], w[1:]
        seg = segments_from_eos(x, EOS)
        f = compute_ice_features(x, seg, (1, 2, 3, 4, 6, 8), "vr")
        if not checked:
            fv = compute_ice_features(x, seg, (1, 2, 3, 4, 6, 8), "v")
            for k in V_KEYS:
                assert (fv[k] == f[k][:12]).all(), f"V slots != first 12 VR slots ({k})"
            checked = True
        # y seen earlier in the same document (any position <= t)
        prev_same = _prev_occurrence(w.astype(np.uint64), np.ones(len(w), bool))
        wseg = np.concatenate([seg, [seg[-1] + (x[-1] == EOS)]])
        seen = (prev_same[1:] >= 0) & (wseg[np.maximum(prev_same[1:], 0)] == wseg[1:])
        sst = f["ice_s_star"]
        shared.append(dict(y=y, sstar=sst, mlen=f["ice_match_len"], dist=f["ice_dist_star"], seen=seen,
                           agree=f["ice_agree_star"]))
        rows = np.arange(L)
        v = views(x, f, rows)
        yt = torch.from_numpy(y).to(dev)
        st = torch.from_numpy(sst).to(dev)
        stc = st.clamp(min=0)
        nx = torch.from_numpy(NEXT[sst]).to(dev)
        for name, d in v.items():
            lf = d["lf"]
            r = dict(nll=(-lf.gather(-1, yt[:, None])[:, 0]).cpu().numpy(),
                     p_star=lf.gather(-1, stc[:, None])[:, 0].exp().cpu().numpy(),
                     top_star=(lf.argmax(-1) == st).cpu().numpy(),
                     p_plus1=torch.where(nx >= 0, lf.gather(-1, nx.clamp(min=0)[:, None])[:, 0].exp(),
                                         torch.zeros_like(lf[:, 0])).cpu().numpy(),
                     top_plus1=((lf.argmax(-1) == nx) & (nx >= 0)).cpu().numpy())
            if "lp" in d:
                lp, valid, tk = d["lp"], d["valid"], d["tok"]
                K = tk.shape[-1]
                pis = lp[:, 1:].exp() * valid
                hit = valid & (tk == yt[:, None])
                g = torch.from_numpy(GID[:K]).to(dev)
                r.update(pi0=lp[:, 0].exp().cpu().numpy(), m_y=(pis * hit).sum(-1).cpu().numpy(),
                         any_y=hit.any(-1).cpu().numpy(), n_valid=valid.sum(-1).cpu().numpy(),
                         s0a=d["s0a"].cpu().numpy(), s0b=d["s0b"].cpu().numpy())
                lpk = lp[:, 1:].masked_fill(~valid, float("-inf"))
                top = lpk.argmax(-1)
                r["top_hit"] = (hit.gather(-1, top[:, None])[:, 0] & valid.any(-1)).cpu().numpy()
                for gi, gn in enumerate(GROUPS):
                    gm = (g == gi)[None]
                    if not bool(gm.any()):
                        continue
                    r[f"gm_{gn}"] = (pis * gm).sum(-1).cpu().numpy()
                    r[f"gh_{gn}"] = (hit & gm).any(-1).cpu().numpy()
                sa = slot_acc.setdefault(name, dict(fill=np.zeros(K), hit=np.zeros(K), pi=np.zeros(K),
                                                    pi_hit=np.zeros(K), n=0))
                sa["fill"] += valid.sum(0).cpu().numpy()
                sa["hit"] += hit.sum(0).cpu().numpy()
                sa["pi"] += pis.sum(0).cpu().numpy()
                sa["pi_hit"] += (pis * hit).sum(0).cpu().numpy()
                sa["n"] += L
                sel = valid[::4]
                sb = sub.setdefault(name, dict(gid=[], hit=[], ell=[], bias=[], ver=[], score=[], ln=[]))
                gg = g[None].expand_as(valid)[::4][sel]
                sb["gid"].append(gg.cpu().numpy())
                sb["hit"].append(hit[::4][sel].cpu().numpy())
                sb["ell"].append(d["ell"][::4][sel].cpu().numpy())
                sb["bias"].append(d["bias"][::4][sel].cpu().numpy())
                sb["ver"].append(d["ver"][::4][sel].cpu().numpy())
                sb["score"].append((d["bias"] + d["ver"])[::4][sel].cpu().numpy())
                sb["ln"].append(d["ln"][::4][sel].cpu().numpy())
            acc.setdefault(name, []).append(r)
        del v
        if j % 25 == 0:
            log(f"heldout window {j}/{a.windows}")
    S = {k: np.concatenate([s[k] for s in shared]) for k in shared[0]}
    A = {n: {k: np.concatenate([r[k] for r in lst]) for k in lst[0]} for n, lst in acc.items()}
    y, sst, mlen, dist, seen = S["y"], S["sstar"], S["mlen"], S["dist"], S["seen"]
    lk_right, lk_wrong, lk_none = sst == y, (sst >= 0) & (sst != y), sst < 0
    cp = mlen >= 4
    masks = dict(all=np.ones_like(cp), copyable=cp, lk_right=lk_right, lk_wrong=lk_wrong, lk_none=lk_none,
                 cp_lk_right=cp & lk_right, cp_lk_wrong=cp & lk_wrong,
                 cp_d_lt64=cp & (dist < 64), cp_d_64_512=cp & (dist >= 64) & (dist < 512),
                 cp_d_512_2k=cp & (dist >= 512) & (dist < 2048), cp_d_ge2k=cp & (dist >= 2048),
                 noncp_match=(~cp) & (sst >= 0), noncp_match_right=(~cp) & lk_right,
                 noncp_match_wrong=(~cp) & lk_wrong,
                 lk_none_seen=lk_none & seen, lk_none_unseen=lk_none & ~seen,
                 notcopy_unseen=(~seen))
    H = {"frac": {k: float(m.mean()) for k, m in masks.items()}}
    names = [n for n in VIEW_ORDER if n in A]
    H["nll"] = {n: {k: float(A[n]["nll"][m].mean()) for k, m in masks.items()} for n in names}
    log("== held-out code: mean NLL per token group (backbone views: *_bb = ICE off)")
    log("   group              frac  " + " ".join(f"{n:>8s}" for n in names))
    for k, m in masks.items():
        log(f"   {k:18s} {m.mean():.3f} " + " ".join(f"{H['nll'][n][k]:8.3f}" for n in names))
    # Q2: lookup-wrong positions
    H["lk_wrong"] = {}
    log("== held-out code, lookup wrong: mean p(s*) (wrong), mean p(y), P(top1=s*), P(p(s*)>0.5), nll")
    for n in names:
        r = A[n]
        m = lk_wrong
        e = dict(p_wrong=float(r["p_star"][m].mean()), p_y=float(np.exp(-r["nll"][m]).mean()),
                 follow=float(r["top_star"][m].mean()), p_wrong_gt_half=float((r["p_star"][m] > 0.5).mean()),
                 nll=float(r["nll"][m].mean()))
        if "pi0" in r:
            e.update(pi0=float(r["pi0"][m].mean()), neg_log_pi0=float(-np.log(np.maximum(r["pi0"][m], 1e-30)).mean()),
                     any_y=float(r["any_y"][m].mean()),
                     m_y=float(r["m_y"][m].mean()),
                     nll_y_in_set=float(r["nll"][m & r["any_y"]].mean()),
                     nll_y_not_in_set=float(r["nll"][m & ~r["any_y"]].mean()),
                     frac_y_in_set=float(r["any_y"][m].mean()))
        H["lk_wrong"][n] = e
        log(f"   {n:8s} " + " ".join(f"{k}={v:.3f}" for k, v in e.items()))
    # follow decomposition: when top1 = wrong s*, how confident?
    for n in names:
        r = A[n]
        m = lk_wrong & r["top_star"]
        H["lk_wrong"][n]["p_wrong_when_followed"] = float(r["p_star"][m].mean()) if m.any() else float("nan")
        H["lk_wrong"][n]["p_wrong_when_followed_median"] = float(np.median(r["p_star"][m])) if m.any() else float("nan")
    log("   p(s*) when top1 = wrong s*: " + " ".join(f"{n}={H['lk_wrong'][n]['p_wrong_when_followed']:.3f}" for n in names))
    # calibration of the copy mass
    H["calib"] = {}
    for n in ("G_V", "G_VR"):
        if n not in A:
            continue
        r = A[n]
        has = r["n_valid"] > 0
        cm = 1.0 - r["pi0"]
        c = dict(auroc_copymass_vs_anyhit=auroc(cm[has], r["any_y"][has]),
                 auroc_my_vs_anyhit=auroc(r["m_y"][has], r["any_y"][has]),
                 auroc_copymass_vs_lkright=auroc(cm[has & (sst >= 0)], lk_right[has & (sst >= 0)]),
                 mean_copymass=float(cm[has].mean()), mean_anyhit=float(r["any_y"][has].mean()),
                 mean_my=float(r["m_y"][has].mean()),
                 sentinel_sat_gt_0_99=float((r["pi0"][has] > 0.99).mean()),
                 sentinel_lt_0_01=float((r["pi0"][has] < 0.01).mean()),
                 top_slot_hit=float(r["top_hit"][has].mean()))
        bins = np.linspace(0, 1, 11)
        bi = np.clip(np.digitize(cm[has], bins) - 1, 0, 9)
        rel = []
        ece = 0.0
        for b in range(10):
            mb = bi == b
            if mb.sum() == 0:
                continue
            row = dict(bin=f"{bins[b]:.1f}-{bins[b + 1]:.1f}", frac=float(mb.mean()), copymass=float(cm[has][mb].mean()),
                       anyhit=float(r["any_y"][has][mb].mean()), m_y=float(r["m_y"][has][mb].mean()),
                       precision=float((r["m_y"][has][mb] / np.maximum(cm[has][mb], 1e-9)).mean()))
            ece += row["frac"] * abs(row["copymass"] - row["m_y"])
            rel.append(row)
        c["ece_copymass_vs_mass_on_target"] = float(ece)
        c["bins"] = rel
        # group masses
        c["group_mass"] = {g: float(r[f"gm_{g}"][has].mean()) for g in GROUPS if f"gm_{g}" in r}
        c["group_hit"] = {g: float(r[f"gh_{g}"][has].mean()) for g in GROUPS if f"gh_{g}" in r}
        c["group_hit_only"] = {}
        for g in GROUPS:
            if f"gh_{g}" not in r:
                continue
            others = np.zeros_like(has)
            for g2 in GROUPS:
                if g2 != g and f"gh_{g2}" in r:
                    others |= r[f"gh_{g2}"]
            only = r[f"gh_{g}"] & ~others
            c["group_hit_only"][g] = float(only[has].mean())
            c[f"mass_y_when_only_{g}"] = float(r["m_y"][has & only].mean()) if (has & only).any() else float("nan")
        c["sentinel_terms"] = dict(s0a_mean=float(r["s0a"].mean()), s0a_std=float(r["s0a"].std()),
                                   s0b_mean=float(r["s0b"].mean()), s0b_std=float(r["s0b"].std()))
        H["calib"][n] = c
        log(f"== calibration {n}: " + " ".join(f"{k}={v:.3f}" for k, v in c.items() if isinstance(v, float)))
        log("   reliability (copy mass bin: frac, mean copy mass, P(any slot = y), mass on y, precision m_y/(1-pi0))")
        for row in rel:
            log(f"     {row['bin']}  {row['frac']:.3f}  {row['copymass']:.3f}  {row['anyhit']:.3f}  {row['m_y']:.3f}  {row['precision']:.3f}")
        log(f"   group mass {c['group_mass']} group hit {c['group_hit']} only-this-group hit {c['group_hit_only']}")
        log("   mass on y when only this group has y: " + " ".join(f"{g}={c.get(f'mass_y_when_only_{g}', float('nan')):.3f}" for g in GROUPS))
        log(f"   sentinel terms {c['sentinel_terms']}")
    # per slot
    H["slots"] = {}
    for n, sa in slot_acc.items():
        K = len(sa["fill"])
        tab = {}
        for k in range(K):
            tab[LABELS[k]] = dict(fill=sa["fill"][k] / sa["n"], prec=sa["hit"][k] / max(1, sa["fill"][k]),
                                  mean_pi=sa["pi"][k] / sa["n"], pi_on_hit_share=sa["pi_hit"][k] / max(1e-9, sa["pi"][k]))
        H["slots"][n] = tab
        log(f"== slots {n}: fill / precision P(c_k=y|filled) / mean pi / share of pi on hits")
        for lab, e in tab.items():
            log(f"   {lab:12s} {e['fill']:.3f} {e['prec']:.3f} {e['mean_pi']:.4f} {e['pi_on_hit_share']:.3f}")
    # AUROC of score terms per group, over filled slots
    H["slot_auroc"] = {}
    for n, sb in sub.items():
        D = {k: np.concatenate(v) for k, v in sb.items()}
        res = {}
        for gi, gn in enumerate(GROUPS):
            m = D["gid"] == gi
            if not m.any():
                continue
            res[gn] = dict(n=int(m.sum()), prec=float(D["hit"][m].mean()), auroc_ell=auroc(D["ell"][m], D["hit"][m]),
                           auroc_bias=auroc(D["bias"][m], D["hit"][m]), auroc_ver=auroc(D["ver"][m], D["hit"][m]),
                           auroc_score=auroc(D["score"][m], D["hit"][m]), auroc_len=auroc(D["ln"][m], D["hit"][m]))
        H["slot_auroc"][n] = res
        log(f"== slot AUROC (hit vs miss, filled slots) {n}")
        for gn, e in res.items():
            log(f"   {gn:7s} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()))
    # digit +1 on real code
    dmask = ISDIG[sst] & ISDIG[np.where(y >= 0, y, VOCAB)] & lk_wrong
    plus = dmask & (y == NEXT[sst])
    H["digit_plus1"] = dict(n_digit_lk_wrong=int(dmask.sum()), frac_y_eq_sstar_plus1=float(plus.sum() / max(1, dmask.sum())),
                            **{n: dict(top_plus1=float(A[n]["top_plus1"][dmask].mean()),
                                       p_plus1=float(A[n]["p_plus1"][dmask].mean()),
                                       p_y_when_plus1=float(np.exp(-A[n]["nll"][plus]).mean()) if plus.any() else float("nan"))
                               for n in names})
    log(f"== digits, lookup wrong: n={dmask.sum()} P(y = s*+1)={H['digit_plus1']['frac_y_eq_sstar_plus1']:.3f}")
    for n in names:
        e = H["digit_plus1"][n]
        log(f"   {n:8s} P(top1=s*+1)={e['top_plus1']:.3f} mean p(s*+1)={e['p_plus1']:.3f} p(y) when y=s*+1: {e['p_y_when_plus1']:.3f}")
    OUT["heldout"] = H

os.makedirs(os.path.dirname(a.out), exist_ok=True)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


with open(a.out, "w") as fh:
    json.dump(_clean(OUT), fh, indent=1)
log("saved", a.out)
