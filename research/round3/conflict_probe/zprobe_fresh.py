"""Zero-training probe (CPU): at held-out val_code positions where s* is wrong and y is in the VR slot set, how often
does each slot score put y at token-level rank 1?  Scores: G_VR's pointer, backbone ell, cosine of the current state
with the state at the candidate's source (i_k - 1), final and mid layer, a random 16-d int8 key, identifier overlap."""
import json, os, sys, time
sys.path.insert(0, "/Users/phunghuy/code/gstar/proposal_ice/ice_lm")
sys.path.insert(0, "/Users/phunghuy/code/gstar/proposal_ice/ice_lm/scripts")
import numpy as np, torch, torch.nn.functional as F
from ice_lm.config import model_config_from_dict
from ice_lm.data import alpha_mask_for_data, load_meta, open_tokens
from ice_lm.ice_features import segments_from_eos, set_alpha_mask, get_alpha_mask
from ice_lm.model import ICELM
from icek_probe import slot_scores
from rule_oracle2 import vr_features, slot_features

SP = "/Users/phunghuy/code/gstar/proposal_ice/local"
RUN, DATA = f"{SP}/runs/t200_G_VR", f"{SP}/data"
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 4
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # first window (decision3 4.4: fresh windows 100-299)
MID = 8
torch.set_grad_enabled(False)
cfg = model_config_from_dict(json.load(open(f"{RUN}/config.json"))["model"])
set_alpha_mask(alpha_mask_for_data(DATA))
alpha = get_alpha_mask()
model = ICELM(cfg)
model.prepare_backend(torch.device("cpu"), torch.float32)
model.load_state_dict(torch.load(f"{RUN}/model_final.pt", map_location="cpu", weights_only=False)["model"])
model.eval()
P = model.ice_ptr
eos = load_meta(DATA)["eos_id"]
data = open_tokens(DATA, "val_code")
L = 4096
g = torch.Generator().manual_seed(0)
R16 = torch.randn(768, 16, generator=g) / 768 ** 0.5

names = ["ptr_sum", "ptr", "ell", "cosF", "cosM", "cosF_src", "rp16", "lex", "ptr+2cosF", "ptr+5cosF", "ptr+10cosF",
         "ptr+5cosM", "ell+5cosF"]
acc = {s: {n: 0.0 for n in names + ["rand", "n"]} for s in ("swrong_yin", "amb_yin", "swrong_yin_tie")}
t0 = time.time()
for wi in range(START, START + NW):
    w = np.asarray(data[wi * L: wi * L + L + 1]).astype(np.int64)
    x, y = w[:-1], w[1:]
    f = vr_features(x, segments_from_eos(x, eos))
    tok, ln, ds, src = (f[k] for k in ("ice_v_tok", "ice_v_len", "ice_v_dist", "ice_v_src"))   # [K,T]
    sf = slot_features(tok, ln, src)
    xt = torch.from_numpy(x)[None]
    h = model.embed(xt)
    for li, blk in enumerate(model.blocks):
        h = blk(h)
        if li == MID - 1:
            hM = h[0].clone()
    hF = model.norm(h)[0]
    W = model.embed.weight
    tokT = torch.from_numpy(tok.T.copy())                    # [T,K]
    ell, lmax = [], []
    for a in range(0, L, 512):
        ls = F.linear(hF[a:a + 512], W).log_softmax(-1)
        ell.append(ls.gather(-1, tokT[a:a + 512].clamp(min=0)).clamp(min=-20.0)); lmax.append(ls.amax(-1))
    ell, lmax = torch.cat(ell), torch.cat(lmax)
    s, _ = slot_scores(P, hF[None], ell[None], lmax[None], tokT[None], torch.from_numpy(ln.T.copy())[None],
                       torch.from_numpy(ds.T.copy())[None])
    s = s[0]
    srcT = torch.from_numpy(src.T.copy())
    idx = (srcT - 1).clamp(min=0)
    def cos(H, ix):
        Hn = F.normalize(H, dim=-1)
        return (Hn[:, None, :] * Hn[ix]).sum(-1)
    cF, cM, cFs = cos(hF, idx), cos(hM, idx), cos(hF, srcT.clamp(min=0))
    kr = torch.tanh(F.rms_norm(hF, (768,)) @ R16); kr = torch.round(kr * 127) / 127
    r16 = (kr[:, None, :] * kr[idx]).sum(-1)
    valid = (tokT >= 0)
    yT = torch.from_numpy(y)
    hit = valid & (tokT == yT[:, None])
    sstar = torch.from_numpy(f["ice_s_star"])
    swrong = (sstar >= 0) & (sstar != yT)
    amb = torch.from_numpy(sf["ndist"] >= 2)
    yin = hit.any(-1)
    # tie: every y slot that is an occurrence slot has delta bucket 8 (= 0) or y only at delta<=0
    delta = torch.from_numpy(sf["delta"].T.copy())
    y_pos_delta = (hit & (delta > 8) & (delta < 17)).any(-1)
    sets = {"swrong_yin": swrong & amb & yin, "amb_yin": amb & yin, "swrong_yin_tie": swrong & amb & yin & ~y_pos_delta}
    scores = {"ptr": s, "ell": ell, "cosF": cF, "cosM": cM, "cosF_src": cFs, "rp16": r16,
              "ptr+2cosF": s + 2 * cF, "ptr+5cosF": s + 5 * cF, "ptr+10cosF": s + 10 * cF, "ptr+5cosM": s + 5 * cM,
              "ell+5cosF": ell + 5 * cF}
    ndist = torch.from_numpy(sf["ndist"]).float()
    for sn, m in sets.items():
        A = acc[sn]
        A["n"] += float(m.sum()); A["rand"] += float((1.0 / ndist.clamp(min=1))[m].sum())
        for n, sc in scores.items():
            sc = sc.masked_fill(~valid, -1e9)
            win = sc.masked_fill(~hit, -1e9).amax(-1) > sc.masked_fill(hit, -1e9).amax(-1)
            A[n] += float((win & m).sum())
        p = s.masked_fill(~valid, -1e9).softmax(-1)
        same = (tokT[:, :, None] == tokT[:, None, :]) & valid[:, :, None] & valid[:, None, :]
        ptok = (same.float() @ p[..., None]).squeeze(-1)
        py = (p * hit).sum(-1)
        A["ptr_sum"] += float(((py > ptok.masked_fill(hit | ~valid, -1.0).amax(-1)) & m).sum())
        # lexical: identifier overlap of the 32 tokens before t+1 and before the source
        pos = torch.nonzero(m).squeeze(-1).tolist()
        lw = 0
        for t in pos:
            cur = set(int(v) for v in x[max(0, t - 31): t + 1] if alpha[v])
            best_y, best_o = -1, -1
            for k in range(tok.shape[0]):
                sk = src[k, t]
                if sk < 0:
                    continue
                o = len(cur & set(int(v) for v in x[max(0, sk - 32): sk] if alpha[v]))
                if tok[k, t] == y[t]:
                    best_y = max(best_y, o)
                else:
                    best_o = max(best_o, o)
            lw += best_y > best_o
        A["lex"] += lw
    print(f"[{time.time() - t0:5.0f}s] window {wi}", {sn: int(acc[sn]["n"]) for sn in acc}, flush=True)

out = {sn: {n: (v / max(1, A["n"]) if n != "n" else v) for n, v in A.items()} for sn, A in acc.items()}
for sn, r in out.items():
    print(sn, " ".join(f"{n}={v:.3f}" if n != "n" else f"n={int(v)}" for n, v in r.items()))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"zprobe_fresh_{START}_{NW}.json"), "w"), indent=1)
