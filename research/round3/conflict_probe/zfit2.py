"""CPU post-hoc refit lite on frozen G_VR: does the backbone (ell, source-state similarity, trained 16-d keys) fix
slot selection where s* is wrong, and what does it buy in mixture NLL?  Fit on val_code windows 300.., test on 0..
Variants add terms to G_VR's trained slot scores s_ptr and sentinel s0:
  g        per-group extra gamma on ell + sentinel offset
  v2       g + V2 tables R[delta], A[cnt], N[rec], Q[q-rank] per group (decision2 section 2, token-only + ell rank)
  v2cos    v2 + per-group weights on cos(h_t, h_{i_k-1}) (final and mid layer); 0 trained key bytes (needs h stored)
  v2keys   v2 + 16-d int8 bilinear keys q_t . k_{i_k-1} (ICE-K, 24.6K params)
usage: zfit.py N_FIT N_TEST STEPS"""
import json, math, os, sys, time
sys.path.insert(0, "/Users/phunghuy/code/gstar/proposal_ice/ice_lm")
sys.path.insert(0, "/Users/phunghuy/code/gstar/proposal_ice/ice_lm/scripts")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ice_lm.config import model_config_from_dict
from ice_lm.data import alpha_mask_for_data, load_meta, open_tokens
from ice_lm.ice_features import segments_from_eos, set_alpha_mask
from ice_lm.model import ICELM, ICEPointer, _NEG
from icek_probe import slot_scores, losses, rank1
from rule_oracle2 import vr_features, slot_features, GID

SP = "/private/tmp/claude-501/-Users-phunghuy-code-gstar/a1fcb2a4-ae79-48f6-b6e8-350e5f8c37f0/scratchpad"
RUN, DATA = f"{SP}/ckpt/t200_G_VR", f"{SP}/ldata"
NF, NT, STEPS = (int(v) for v in sys.argv[1:4])
VARS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["g", "v2", "v2cos", "v2keys"]
R16 = torch.randn(768, 16, generator=torch.Generator().manual_seed(0)) / 768 ** 0.5
MID, L = 8, 4096
torch.manual_seed(0)
cfg = model_config_from_dict(json.load(open(f"{RUN}/config.json"))["model"])
set_alpha_mask(alpha_mask_for_data(DATA))
model = ICELM(cfg); model.prepare_backend(torch.device("cpu"), torch.float32)
model.load_state_dict(torch.load(f"{RUN}/model_final.pt", map_location="cpu", weights_only=False)["model"])
model.eval(); P = model.ice_ptr
eos = load_meta(DATA)["eos_id"]; data = open_tokens(DATA, "val_code")
t0 = time.time()
G = torch.from_numpy(GID)


@torch.no_grad()
def cache(wi):
    w = np.asarray(data[wi * L: wi * L + L + 1]).astype(np.int64)
    x, y = w[:-1], w[1:]
    f = vr_features(x, segments_from_eos(x, eos))
    tok, ln, ds, src = (f[k] for k in ("ice_v_tok", "ice_v_len", "ice_v_dist", "ice_v_src"))
    sf = slot_features(tok, ln, src)
    h = model.embed(torch.from_numpy(x)[None])
    for li, blk in enumerate(model.blocks):
        h = blk(h)
        if li == MID - 1:
            hM = h[0].clone()
    hF = model.norm(h)[0]
    tokT = torch.from_numpy(tok.T.copy())
    ell, ly, lmax, am = [], [], [], []
    yt = torch.from_numpy(y)
    for a in range(0, L, 512):
        ls = F.linear(hF[a:a + 512], model.embed.weight).log_softmax(-1)
        ell.append(ls.gather(-1, tokT[a:a + 512].clamp(min=0)).clamp(min=-20.0)); lmax.append(ls.amax(-1)); am.append(ls.argmax(-1))
        ly.append(ls.gather(-1, yt[a:a + 512, None]).squeeze(-1))
    ell, lmax, ly, am = torch.cat(ell), torch.cat(lmax), torch.cat(ly), torch.cat(am)
    s, s0 = slot_scores(P, hF[None], ell[None], lmax[None], tokT[None], torch.from_numpy(ln.T.copy())[None],
                        torch.from_numpy(ds.T.copy())[None])
    srcT = torch.from_numpy(src.T.copy()); idx = (srcT - 1).clamp(min=0)
    cF = (F.normalize(hF, dim=-1)[:, None] * F.normalize(hF, dim=-1)[idx]).sum(-1)
    cM = (F.normalize(hM, dim=-1)[:, None] * F.normalize(hM, dim=-1)[idx]).sum(-1)
    kr = torch.tanh(F.rms_norm(hF, (768,)) @ R16); kr = torch.round(kr * 127) / 127
    cR = F.cosine_similarity(kr[:, None], kr[idx], dim=-1)
    T_ = lambda a: torch.from_numpy(np.ascontiguousarray(a.T)).long()
    return dict(tok=tokT, y=yt, s=s[0], s0=s0[0], ly=ly, ell=ell, cF=cF, cM=cM, cR=cR, idx=idx, lmax=lmax, am=am,
                d=T_(sf["delta"]), n=T_(sf["cnt"]), r=T_(sf["rec"]), q=ICEPointer.q_rank(ell[None], tokT[None])[0],
                sstar=torch.from_numpy(f["ice_s_star"]), h=F.rms_norm(hF, (768,)).half())


fit = [cache(300 + i) for i in range(NF)]
print(f"[{time.time() - t0:5.0f}s] cached {NF} fit windows", flush=True)
test = [cache(i) for i in range(NT)]
print(f"[{time.time() - t0:5.0f}s] cached {NT} test windows", flush=True)


class Head(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v
        self.dg = nn.Parameter(torch.zeros(4)); self.e = nn.Parameter(torch.zeros(1))
        if v != "g":
            self.R = nn.Parameter(torch.zeros(4, 18)); self.A = nn.Parameter(torch.zeros(4, 6))
            self.N = nn.Parameter(torch.zeros(4, 4)); self.Q = nn.Parameter(torch.zeros(4, 3))
        if v == "v2gate":
            self.U = nn.Linear(768, 4, bias=False); nn.init.zeros_(self.U.weight)
        if v == "v2rp":
            self.wR = nn.Parameter(torch.zeros(4))
        if v == "v2cos":
            self.wF = nn.Parameter(torch.zeros(4)); self.wM = nn.Parameter(torch.zeros(4))
        if v == "v2keys":
            self.Wq = nn.Linear(768, 16, bias=False); self.Wk = nn.Linear(768, 16, bias=False)
            nn.init.zeros_(self.Wq.weight); nn.init.normal_(self.Wk.weight, 0, 1 / math.sqrt(768))

    def forward(self, c):
        s = c["s"] + self.dg[G] * c["ell"]
        if self.v != "g":
            s = s + self.R[G, c["d"]] + self.A[G, c["n"]] + self.N[G, c["r"]] + self.Q[G, c["q"]]
        if self.v == "v2cos":
            s = s + self.wF[G] * c["cF"] + self.wM[G] * c["cM"]
        if self.v == "v2rp":
            s = s + self.wR[G] * c["cR"]
        if self.v == "v2gate":
            s = s + self.U(c["h"].float())[:, G] * c["ell"]
        if self.v == "v2keys":
            h = c["h"].float()
            k = torch.tanh(self.Wk(h)); k = k + (torch.round(k * 127) / 127 - k).detach()
            s = s + (self.Wq(h)[:, None] * k[c["idx"]]).sum(-1) / 4.0
        return s.masked_fill(c["tok"] < 0, float("-inf")), c["s0"] + self.e


def evaluate(head, ws):
    tot = dict(n=0, nll=0.0, sw_n=0, sw_nll=0.0, swy_n=0, swy_r1=0, ay_n=0, ay_r1=0, swy_nll=0.0)
    per_w = []
    with torch.no_grad():
        for c in ws:
            s, s0 = head(c) if head is not None else (c["s"].masked_fill(c["tok"] < 0, float("-inf")), c["s0"])
            nll, lw, lwm, amb, hit = losses(s[None], s0[None], c["ly"][None], c["tok"][None], c["y"][None])
            nll, amb, hit = nll[0], amb[0], hit[0]
            r1 = rank1(s[None], c["tok"][None], hit[None])[0]
            sw = (c["sstar"] >= 0) & (c["sstar"] != c["y"])
            yin = hit.any(-1)
            tot["n"] += L; tot["nll"] += float(nll.sum()); per_w.append(float(nll.mean()))
            tot["sw_n"] += int(sw.sum()); tot["sw_nll"] += float(nll[sw].sum())
            m = sw & amb & yin
            tot["swy_n"] += int(m.sum()); tot["swy_r1"] += int((r1 & m).sum()); tot["swy_nll"] += float(nll[m].sum())
            lp = torch.cat([s0[:, None], s], -1).log_softmax(-1)
            lp0, lpk = lp[:, 0], lp[:, 1:].clamp(min=_NEG)
            tok = c["tok"]; valid = tok >= 0
            same = (tok[:, :, None] == tok[:, None, :]) & valid[:, :, None] & valid[:, None, :]
            agg = torch.where(same, lpk[:, None, :], torch.full_like(same, _NEG, dtype=lpk.dtype)).logsumexp(-1)
            mk = torch.logaddexp(lp0[:, None] + c["ell"], agg).masked_fill(~valid, _NEG)        # [T,K]
            am_in = (tok == c["am"][:, None]).any(-1)
            mam = torch.where(am_in, torch.full_like(lp0, _NEG), lp0 + c["lmax"])
            kbest = mk.argmax(-1); vbest = mk.max(-1).values
            top_tok = torch.where(vbest >= mam, tok.gather(-1, kbest[:, None]).squeeze(-1), c["am"])
            top1 = top_tok == c["y"]
            tot.setdefault("top1", 0); tot.setdefault("sw_top1", 0); tot.setdefault("sw_follow", 0)
            tot["top1"] += int(top1.sum()); tot["sw_top1"] += int((top1 & sw).sum())
            tot["sw_follow"] += int(((top_tok == c["sstar"]) & sw).sum())
            m2 = amb & yin
            tot["ay_n"] += int(m2.sum()); tot["ay_r1"] += int((r1 & m2).sum())
    return dict(nll=tot["nll"] / tot["n"], nll_swrong=tot["sw_nll"] / tot["sw_n"],
                nll_swrong_yin=tot["swy_nll"] / tot["swy_n"], r1_swrong_yin=tot["swy_r1"] / tot["swy_n"],
                r1_amb_yin=tot["ay_r1"] / tot["ay_n"], swy_frac=tot["swy_n"] / tot["n"], top1=tot["top1"] / tot["n"],
                top1_swrong=tot["sw_top1"] / tot["sw_n"], follow_swrong=tot["sw_follow"] / tot["sw_n"]), np.array(per_w)


res = {}
ref, ref_w = evaluate(None, test)
res["ref"] = ref
print("ref", {k: round(v, 4) for k, v in ref.items()}, flush=True)
for v in VARS:
    head = Head(v)
    kp = [p for n, p in head.named_parameters() if n.startswith('W') or n.startswith('U')]
    op = [p for n, p in head.named_parameters() if not (n.startswith('W') or n.startswith('U'))]
    opt = torch.optim.Adam([dict(params=op, lr=1e-2)] + ([dict(params=kp, lr=1e-3)] if kp else []))
    rng = np.random.default_rng(0)
    for step in range(STEPS):
        loss = 0.0
        for j in rng.choice(NF, size=min(4, NF), replace=False):
            c = fit[j]
            s, s0 = head(c)
            nll, lw, lwm, *_ = losses(s[None], s0[None], c["ly"][None], c["tok"][None], c["y"][None])
            loss = loss + (nll.mean() + 0.25 * lw.sum() / L) / 4
        opt.zero_grad(); loss.backward(); opt.step()
    r, rw = evaluate(head, test)
    d = rw - ref_w
    bs = np.random.default_rng(1).integers(0, len(d), (2000, len(d)))
    r["dnll"] = float(d.mean()); r["dnll_ci"] = [float(np.percentile(d[bs].mean(1), q)) for q in (2.5, 97.5)]
    r["fit_nll_end"] = float(loss.detach())
    res[v] = r
    print(f"[{time.time() - t0:5.0f}s]", v, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()},
          flush=True)
    if v == "g":
        print("   dg", head.dg.data.tolist(), "e", head.e.item())
    if v == "v2cos":
        print("   wF", head.wF.data.tolist(), "wM", head.wM.data.tolist())
json.dump(res, open(f"{SP}/conflict_probe/zfit2_{NF}_{NT}_{STEPS}_{'-'.join(VARS)}.json", "w"), indent=1)
