"""Run: ICE_CPU_ONLY=1 [T=4096 OFF=40000] python research/round3/verifier_walkthrough.py (needs local/runs/t200_G_VR, local/data).
Walk through G_VR's verifier on real held-out code: candidates, their scores, pi, and the final probability."""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
R = "/Users/phunghuy/code/gstar/proposal_ice"
sys.path.insert(0, f"{R}/ice_lm")
from ice_lm.config import model_config_from_dict
from ice_lm.data import alpha_mask_for_data, load_meta, open_tokens
from ice_lm.ice_features import compute_ice_features, set_alpha_mask, segments_from_eos, MAX_MATCH
from ice_lm.model import ICELM, V_DIST_MAX
from transformers import AutoTokenizer

D, RUN = f"{R}/local/data", f"{R}/local/runs/t200_G_VR"
torch.manual_seed(0)
meta = load_meta(D)
tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
cfg = json.load(open(f"{RUN}/config.json"))
mc = model_config_from_dict(cfg["model"])
set_alpha_mask(alpha_mask_for_data(D))
m = ICELM(mc)
m.load_state_dict(torch.load(f"{RUN}/model_final.pt", map_location="cpu", weights_only=False)["model"])
m.prepare_backend(torch.device("cpu"), torch.float32)
m.eval()
P = m.ice_ptr
N = len(mc.ice_orders)
names = [f"order{n}-{w}" for n in mc.ice_orders for w in ("last", "2nd")] + [f"chain-occ{i}" for i in range(3, 9)] \
    + [f"anchor{a}-key{o}-span{s}" for a in range(2) for o in (1, 2) for s in range(1, 5)]

code = np.asarray(open_tokens(D, "val_code"))
T = int(os.environ.get("T", 2048))
xs = code[int(os.environ.get("OFF", 3 * 4096)):][:T + 1].astype(np.int64)
x, y = xs[:-1], xs[1:]
seg = segments_from_eos(x, meta["eos_id"])
f = compute_ice_features(x, seg, mc.ice_orders, "vr", False)
ct, cl, cd = (torch.from_numpy(f[k]).T[None] for k in ("ice_v_tok", "ice_v_len", "ice_v_dist"))   # [1,T,K]
with torch.no_grad():
    h = m.embed(torch.from_numpy(x)[None])
    for b in m.blocks:
        h = b(h)
    hl = m.norm(h)
    w = m.embed.weight if m.lm_head is None else m.lm_head.weight
    z = F.linear(hl, w).float()
    l = z.log_softmax(-1)
    keep = torch.ones(1)
    lp, valid = P.log_pi(hl, l, keep, ct, cl, cd)                                    # [1,T,1+K]
    # score components
    src = torch.arange(P.K)
    sb = P.b.weight[:, 0].expand_as(ct.float())
    sL = P.L(src * (MAX_MATCH + 1) + cl.clamp(0, MAX_MATCH)).squeeze(-1)
    sD = P.D(cd.clamp(max=V_DIST_MAX)).squeeze(-1)
    sV = P.gamma * l.gather(-1, ct.clamp(min=0)).clamp(min=-20.0)
    s0w = P.w(P.norm(hl)).squeeze(-1) + P.c0
    s0b = P.beta * l.amax(-1)
    mix = m(torch.from_numpy(x)[None], ice={k: torch.from_numpy(f[k])[None] for k in ("ice_v_tok", "ice_v_len", "ice_v_dist")})
lp, l, mix = lp[0], l[0], mix[0]
Y = torch.from_numpy(y)
lpy = l.gather(-1, Y[:, None])[:, 0]                         # backbone log p(y)
lmy = mix.gather(-1, Y[:, None])[:, 0]                       # final log p(y)
pi0 = lp[:, 0].exp()
s_star = f["ice_s_star"]
in_c = ((ct[0] == Y[:, None]) & valid[0]).any(-1).numpy()
grp = dict(right=s_star == y, wrong_in=(s_star >= 0) & (s_star != y) & in_c,
           wrong_out=(s_star >= 0) & (s_star != y) & ~in_c, none=s_star < 0)

print(f"\n=== learned verifier weights (G_VR) ===  c0={P.c0.item():+.2f}  beta={P.beta.item():+.2f}")
for i in list(range(0, 12, 2)) + [12, 17, 18, 22]:
    print(f"  {names[i]:22s} b={P.b.weight[i,0].item():+5.2f}  gamma={P.gamma[i].item():+5.2f}")
print(f"\n=== averages over {T} positions of held-out code ===")
print(f"{'group':12s}{'share':>7s}{'pi0 (trust model)':>19s}{'loss backbone':>15s}{'loss final':>12s}{'change':>9s}")
for g, mk in grp.items():
    mk = torch.from_numpy(mk)
    print(f"{g:12s}{mk.float().mean().item():7.1%}{pi0[mk].mean().item():19.3f}{-lpy[mk].mean().item():15.3f}"
          f"{-lmy[mk].mean().item():12.3f}{(lpy[mk] - lmy[mk]).mean().item():+9.3f}")
mk = torch.from_numpy(grp["wrong_out"])
print(f"  wrong_out: cost -log pi0 = {(-lp[:,0][mk]).mean().item():.3f} on average (the only price of a useless lookup)")


def show(t, title):
    ctx = tok.decode(x[max(0, t - 15): t + 1].tolist())
    print(f"\n--- {title} (position {t}) ---")
    print(f"context ends: {ctx[-90:]!r}")
    print(f"true next: {tok.decode([int(y[t])])!r} | lookup's longest match s*: {tok.decode([int(s_star[t])]) if s_star[t] >= 0 else None!r}")
    tops = l[t].topk(3)
    print("backbone alone: " + ", ".join(f"{tok.decode([int(i)])!r} {p:.2f}" for p, i in zip(tops.values.exp(), tops.indices)))
    print(f"sentinel s0 = {s0w[0,t].item():+.2f} (from hidden state) {s0b[0,t].item():+.2f} (beta * max log p)  ->  pi0 = {pi0[t].item():.3f}")
    seen = {}
    for k in range(P.K):
        if not valid[0, t, k]:
            continue
        c = int(ct[0, t, k])
        seen.setdefault(c, []).append(k)
    rows = []
    for c, ks in seen.items():
        k = max(ks, key=lambda j: lp[t, 1 + j].item())
        rows.append((sum(lp[t, 1 + j].exp().item() for j in ks), c, k, len(ks)))
    for pk, c, k, n in sorted(rows, reverse=True)[:5]:
        mark = "  <- correct" if c == y[t] else ""
        print(f"  cand {tok.decode([c])!r:12s} pi={pk:.3f} ({n} slots; best {names[k]}, len {int(cl[0,t,k])}): "
              f"b {sb[0,t,k].item():+.2f}  L {sL[0,t,k].item():+.2f}  D {sD[0,t,k].item():+.2f}  "
              f"gamma*logp_model {sV[0,t,k].item():+.2f} (model p={l[t,c].exp().item():.3f}){mark}")
    print(f"p(true) backbone {lpy[t].exp().item():.3f}  ->  final {lmy[t].exp().item():.3f}")


rng = np.random.default_rng(int(os.environ.get("SEED", 1)))
gain = (lmy - lpy).numpy()
r = np.nonzero(grp["right"] & (gain > 0.5))[0]
wo = np.nonzero(grp["wrong_out"] & (l.exp().gather(-1, torch.from_numpy(np.maximum(s_star, 0))[:, None])[:, 0].numpy() < 0.2))[0]
wi = np.nonzero(grp["wrong_in"] & (gain > 0.3))[0]
wbad = np.nonzero((grp["wrong_in"] | grp["wrong_out"]) & (gain < -0.5))[0]
for arr, title in ((r, "A. lookup right, model unsure"), (wo, "B. lookup wrong, model disagrees"),
                   (wi, "C. longest match wrong, another candidate right"), (wbad, "D. lookup wrong and it hurts")):
    if len(arr):
        show(int(rng.choice(arr)), title)
