"""Hybrid GDN / full-attention language model with optional In-Context Engram (ICE).

Every layer is a HybridMixer with a head allocation:
    * GDN layer:            n_gdn = n_heads, n_attn = 0
    * full-attention layer: n_gdn = 0,       n_attn = n_heads (GQA with n_kv_heads)
    * head-level layer:     both (attention heads are one GQA group: n_heads // n_kv_heads queries, 1 KV head)
ICE (see proposal_ice/ice_explained.pdf):
    input side  (after block `ice_inject_after`):  h <- h + k * sum_n m_n g_n f_n
                f_n = W_n E[s_n] + L[len_n] + D[dist_n] (+ A_n[agree_n] + R_n[rel_n]),
                g_n = sigmoid(<RMSNorm h, RMSNorm f_n> / sqrt(d))
    output side (logits):  z[s*] += k * softplus(w^T RMSNorm(h_last) + b[len*] (+ a[agree*] + r[rel*]))
    agree / rel (ice_features.py) say whether earlier occurrences disagree and how often this order has been
    right so far in the document: they let the model learn when to ignore a misleading lookup.
    k = 1 normally, k = 0 for ICE-dropped sequences (and k = 1 always at test time).
ICE-V / VR (cfg.ice_v, ICEPointer): the backbone never sees ICE (unless cfg.ice_v_input); its logits z are mixed
    with K candidate tokens c_k proposed by the lookup (ice_features: V = successors of the last / second
    occurrence per order, VR = + suffix chain + identifier-anchor spans), a log-space pointer-sentinel:
        l = log_softmax(z) (detached in the scores)
        s_k = b[src_k] + L[src_k, len_k] + D[dist_k] + gamma[src_k] * clip(l[c_k], -20, 0)
        s_0 = w^T RMSNorm(h_last) + beta * max_v l + c0
        log pi = log_softmax([s_0, s_1..K])   (empty slots and k = 0: -inf)
        log p(y) = logaddexp(log pi_0 + log_softmax(z)[y], logsumexp_{k: c_k = y} log pi_k)
    trained with -log p(y) + ice_v_aux * CE(z, y). k = 0 gives exactly log_softmax(z).
ICE-V2 (cfg.ice_v2, research/round2/decision2.md section 2): slot group g(k) in {last, second, chain, anchor},
        s_k = b[src_k] + L[src_k, len_k] + D[dist_k] + softplus(gamma~[src_k]) * clip(l[c_k], -20, 0)
              + R_g[delta_k] + A_g[n_k] + N_g[r_k] + Q_g[q_k]
        s_0 = w^T RMSNorm(h_last) + c0                      (no beta)
    delta / n / r are token-only slot features (ice_features.V2_KEYS); q_k = rank of l[c_k] among the distinct slot
    tokens (0..2), computed here from the detached backbone log-probs. The tables start at 0. Optional listwise
    term (cfg.ice_v_listwise = lambda): + lambda * 1[y in the slots, >= 2 distinct tokens] *
    (-log sum_{c_k = y} softmax_{valid slots}(s)_k).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .gdn import GatedDeltaNetHeads, RMSNorm, resolve_backend, set_torch_kernel_compile
from .ice_features import (CNT_MAX, MAX_MATCH, N_AGREE_BUCKETS, N_DELTA, N_DIST_BUCKETS, N_LEN_BUCKETS, N_REL_BUCKETS,
                           REC_MAX, V2_KEYS, V_KEYS, n_slots, slot_groups)

OUT_KEYS = ("ice_s_star", "ice_len_star", "ice_agree_star", "ice_rel_star")
V_DIST_MAX = 12          # largest distance bucket seen in training at seq_len 4096: longer distances use it
_NEG = -1e4              # finite "-inf" where a gradient flows (exp underflows to exactly 0 in fp32)
Q_MAX = 2                # V2: rank of the backbone log-prob among the distinct slot tokens, clipped
V2_TABLES = dict(d=("R", N_DELTA), n=("A", CNT_MAX + 1), r=("N", REC_MAX + 1), q=("Q", Q_MAX + 1))
GAMMA0 = 0.3             # V2 fresh init: softplus(gamma~) = 0.3


# ---------------------------------------------------------------------------
# Layout: which heads are GDN / attention in each layer
# ---------------------------------------------------------------------------
def _evenly_spaced(n_layers: int, count: int) -> List[int]:
    count = max(1, min(count, n_layers))
    return sorted({round((j + 1) * n_layers / count) - 1 for j in range(count)})


def layer_plan(cfg: ModelConfig) -> List[Dict[str, int]]:
    """Per layer: dict(gdn=#GDN heads, attn=#attention query heads, kv=#KV heads)."""
    L, H, KV = cfg.n_layers, cfg.n_heads, cfg.n_kv_heads
    plan = [dict(gdn=H, attn=0, kv=0) for _ in range(L)]
    if cfg.layout == "layer":
        n_attn_layers = max(1, round(L / (cfg.ratio + 1)))
        for i in _evenly_spaced(L, n_attn_layers):
            plan[i] = dict(gdn=0, attn=H, kv=KV)
    elif cfg.layout == "gdn":
        pass                                               # pure recurrent model: no attention, no KV cache
    elif cfg.layout == "head":
        # match the KV budget of a layer-level hybrid with ratio head_kv_budget_ratio
        n_ref_layers = max(1, round(L / (cfg.head_kv_budget_ratio + 1)))
        kv_budget = n_ref_layers * KV                     # total KV heads in the reference model
        group = H // KV                                    # query heads per KV head
        for i in _evenly_spaced(L, kv_budget):
            plan[i] = dict(gdn=H - group, attn=group, kv=1)
        assigned = sum(p["kv"] for p in plan)
        if assigned != kv_budget:
            raise ValueError(f"head layout: cannot place {kv_budget} KV heads in {L} layers")
    else:
        raise ValueError(f"unknown layout {cfg.layout!r}")
    return plan


def kv_bytes_per_token(cfg: ModelConfig, bytes_per_el: int = 2) -> int:
    return sum(p["kv"] for p in layer_plan(cfg)) * cfg.head_dim * 2 * bytes_per_el


# ---------------------------------------------------------------------------
# Attention heads
# ---------------------------------------------------------------------------
class Rotary(nn.Module):
    def __init__(self, dim: int, theta: float):
        super().__init__()
        # a plain attribute, not a buffer: model.to(fp16) must not round the frequencies (wrong angles at 32K)
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)).float()
        self._cache: Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]] = (0, None, None)

    def forward(self, T: int, device, offset: int = 0):
        n, cos, sin = self._cache
        end = offset + T
        if n < end or cos is None or cos.device != device:
            t = torch.arange(max(2 * end, 1024), device=device, dtype=torch.float32)   # grow geometrically
            freqs = torch.outer(t, self.inv_freq.to(device))
            cos, sin = freqs.cos(), freqs.sin()
            self._cache = (cos.shape[0], cos, sin)
        return cos[offset:end], sin[offset:end]


def apply_rope(x, cos, sin):  # x: [B,T,H,D]
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)


class AttentionHeads(nn.Module):
    def __init__(self, d_model: int, n_q: int, n_kv: int, head_dim: int, rope_theta: float, qk_norm: bool):
        super().__init__()
        self.n_q, self.n_kv, self.head_dim = n_q, n_kv, head_dim
        self.q_proj = nn.Linear(d_model, n_q * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv * head_dim, bias=False)
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.rope = Rotary(head_dim, rope_theta)

    def forward(self, x, cache=None, pos: int = 0):
        """cache (inference only): object with preallocated .k/.v [B, n_kv, max_len, D]; x holds positions
        pos .. pos+T-1 and is written into the cache in place."""
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_q, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv, self.head_dim))
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim)
        cos, sin = self.rope(T, x.device, pos)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            cache.k[:, :, pos:pos + T] = k.transpose(1, 2).to(cache.k.dtype)
            cache.v[:, :, pos:pos + T] = v.transpose(1, 2).to(cache.v.dtype)
            if pos > 0:
                return self._attend_cache(q, cache.k[:, :, :pos + T], cache.v[:, :, :pos + T], pos)
        if self.n_kv != self.n_q:
            rep = self.n_q // self.n_kv
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return o.transpose(1, 2).reshape(B, T, self.n_q * self.head_dim)

    def _attend_cache(self, q, K, V, pos: int):
        """q [B,T,n_q,D] at positions pos..pos+T-1; K,V [B,n_kv,pos+T,D] (cache incl. the new positions)."""
        B, T = q.shape[:2]
        rep, D = self.n_q // self.n_kv, self.head_dim
        if T == 1:   # decode: grouped-query attention without copying the cache (query head i uses KV head i//rep):
            # the rep query heads of one KV head act as rep "query positions" that all see the whole cache
            qg = q.reshape(B, self.n_kv, rep, D).to(K.dtype)
            o = F.scaled_dot_product_attention(qg, K, V)                    # [B,n_kv,rep,D]
            return o.reshape(B, 1, self.n_q * D).to(q.dtype)
        # chunk with history (chunked prefill): explicit mask, SDPA's is_causal is aligned top-left
        L = K.shape[2]
        mask = torch.arange(L, device=q.device)[None, :] <= (pos + torch.arange(T, device=q.device))[:, None]
        if rep > 1:
            K, V = K.repeat_interleave(rep, dim=1), V.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), K.to(q.dtype), V.to(q.dtype), attn_mask=mask)
        return o.transpose(1, 2).reshape(B, T, self.n_q * D)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
class HybridMixer(nn.Module):
    def __init__(self, cfg: ModelConfig, n_gdn: int, n_attn: int, n_kv: int):
        super().__init__()
        self.gdn = (GatedDeltaNetHeads(cfg.d_model, n_gdn, cfg.head_dim, cfg.conv_size, cfg.gdn_backend, cfg.gdn_chunk)
                    if n_gdn > 0 else None)
        self.attn = (AttentionHeads(cfg.d_model, n_attn, n_kv, cfg.head_dim, cfg.rope_theta, cfg.qk_norm)
                     if n_attn > 0 else None)
        self.o_proj = nn.Linear((n_gdn + n_attn) * cfg.head_dim, cfg.d_model, bias=False)
        self.o_proj._is_residual_out = True

    def forward(self, x, cache=None, pos: int = 0):
        outs = []
        if self.gdn is not None:
            outs.append(self.gdn(x, cache))
        if self.attn is not None:
            outs.append(self.attn(x, cache, pos))
        return self.o_proj(torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0])


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        self.down._is_residual_out = True

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, plan: Dict[str, int]):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = HybridMixer(cfg, plan["gdn"], plan["attn"], plan["kv"])
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)

    def forward(self, x, cache=None, pos: int = 0):
        x = x + self.mixer(self.norm1(x), cache, pos)
        return x + self.mlp(self.norm2(x))


# ---------------------------------------------------------------------------
# ICE modules
# ---------------------------------------------------------------------------
class ICEInput(nn.Module):
    def __init__(self, d: int, n_orders: int, trust: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(n_orders, d, d))      # zero init: ICE starts silent
        self.L = nn.Embedding(N_LEN_BUCKETS, d)
        self.D = nn.Embedding(N_DIST_BUCKETS, d)
        nn.init.zeros_(self.L.weight)
        nn.init.zeros_(self.D.weight)
        self.n_orders, self.trust = n_orders, trust
        if trust:   # per-order "trust" embeddings (agreement of occurrences, online reliability)
            self.A = nn.Embedding(n_orders * N_AGREE_BUCKETS, d)
            self.R = nn.Embedding(n_orders * N_REL_BUCKETS, d)
            nn.init.zeros_(self.A.weight)
            nn.init.zeros_(self.R.weight)
        self.norm_h = RMSNorm(d)
        self.norm_f = RMSNorm(d)
        self.d = d

    def forward(self, h, tok_emb: nn.Embedding, succ, len_b, dist_b, keep, agree=None, rel=None):
        # h [B,T,d]; succ/len_b/dist_b/agree/rel [B,N,T]; keep [B]
        m = (succ >= 0).to(h.dtype)                                    # [B,N,T]
        e = tok_emb(succ.clamp(min=0))                                 # [B,N,T,d]
        f = torch.einsum("bntd,ned->bnte", e, self.W.to(e.dtype))      # W_n E[s_n]
        f = f + self.L(len_b) + self.D(dist_b)
        if self.trust:
            order = torch.arange(self.n_orders, device=succ.device)[None, :, None]
            f = f + self.A(order * N_AGREE_BUCKETS + agree) + self.R(order * N_REL_BUCKETS + rel)
        hn = self.norm_h(h).unsqueeze(1)                               # [B,1,T,d]
        g = torch.sigmoid((hn * self.norm_f(f)).float().sum(-1) / math.sqrt(self.d))   # [B,N,T]
        add = ((m * g.to(h.dtype)).unsqueeze(-1) * f).sum(1)           # [B,T,d]
        return h + keep.to(h.dtype)[:, None, None] * add


class ICEOutput(nn.Module):
    def __init__(self, d: int, trust: bool = True):
        super().__init__()
        self.norm = RMSNorm(d)
        self.w = nn.Linear(d, 1, bias=False)
        self.b = nn.Embedding(N_LEN_BUCKETS, 1)
        nn.init.zeros_(self.w.weight)
        nn.init.constant_(self.b.weight, -2.0)     # lambda ~ 0.13 at init
        self.trust = trust
        if trust:
            self.a = nn.Embedding(N_AGREE_BUCKETS, 1)
            self.r = nn.Embedding(N_REL_BUCKETS, 1)
            nn.init.zeros_(self.a.weight)
            nn.init.zeros_(self.r.weight)

    def forward(self, logits, h_last, s_star, len_star, keep, agree_star=None, rel_star=None):
        pre = self.w(self.norm(h_last)).float().squeeze(-1) + self.b(len_star).float().squeeze(-1)
        if self.trust:
            pre = pre + self.a(agree_star).float().squeeze(-1) + self.r(rel_star).float().squeeze(-1)
        lam = F.softplus(pre)
        lam = lam * (s_star >= 0).float() * keep.float()[:, None]      # [B,T]
        return logits.scatter_add(-1, s_star.clamp(min=0).unsqueeze(-1), lam.unsqueeze(-1).to(logits.dtype))


class ICEPointer(nn.Module):
    """ICE-V / VR log-space pointer-sentinel over K candidate slots (see the module docstring). Candidate
    features (tok, ln, dist) are [B,T,K]; slot k has its own source id (order x occurrence, chain depth, anchor
    key and span offset). v2: the ICE-V2 scorer, with the slot features (delta, cnt, rec) [B,T,K] appended to the
    inputs; groups = slot_groups(cand) and feats = the subset of "dnrq" whose tables are used."""

    def __init__(self, d: int, n_slots: int, verify: bool = True, v2: bool = False,
                 groups: Optional[List[int]] = None, feats: str = "dnrq"):
        super().__init__()
        self.K, self.verify, self.v2 = n_slots, verify, v2
        self.norm = RMSNorm(d)
        self.w = nn.Linear(d, 1, bias=False)
        self.b = nn.Embedding(n_slots, 1)
        self.L = nn.Embedding(n_slots * (MAX_MATCH + 1), 1)
        self.D = nn.Embedding(N_DIST_BUCKETS, 1)
        self.c0 = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.w.weight)
        nn.init.constant_(self.b.weight, -3.0)     # pi_k ~ 0.05 pi_0 per filled slot at init
        nn.init.zeros_(self.L.weight)
        nn.init.zeros_(self.D.weight)
        if v2:
            assert groups is not None and len(groups) == n_slots and set(feats) <= set(V2_TABLES), (groups, feats)
            self.feats = "".join(f for f in "dnrq" if f in feats)
            self.n_groups = max(groups) + 1
            self.register_buffer("group", torch.tensor(groups, dtype=torch.long), persistent=False)
            for f in self.feats:
                name, nb = V2_TABLES[f]
                setattr(self, name, nn.Embedding(self.n_groups * nb, 1))
                nn.init.zeros_(getattr(self, name).weight)            # V2 starts as the V scorer
            if verify:   # gamma = softplus(gamma_t) >= 0; no beta
                self.gamma_t = nn.Parameter(torch.full((n_slots,), math.log(math.expm1(GAMMA0))))
        elif verify:
            self.gamma = nn.Parameter(torch.zeros(n_slots))
            self.beta = nn.Parameter(torch.zeros(1))

    def log_pi(self, h_last, ell, keep, tok, ln, dist, *v2f):
        """ell = detached log_softmax(z) [B,T,V] -> log pi [B,T,1+K] (sentinel first; empty slots -inf).
        v2f (V2 only): delta, cnt, rec [B,T,K]."""
        if self.v2:
            return self._log_pi_v2(h_last, ell, keep, tok, ln, dist, *v2f)
        src = torch.arange(self.K, device=tok.device)
        s = (self.b.weight[:, 0] + self.L(src * (MAX_MATCH + 1) + ln.clamp(0, MAX_MATCH)).squeeze(-1)
             + self.D(dist.clamp(max=V_DIST_MAX)).squeeze(-1)).float()                     # [B,T,K]
        s0 = self.w(self.norm(h_last)).float().squeeze(-1) + self.c0.float()                # [B,T]
        if self.verify:
            s = s + self.gamma.float() * ell.gather(-1, tok.clamp(min=0)).clamp(min=-20.0)
            s0 = s0 + self.beta.float() * ell.amax(-1)
        valid = (tok >= 0) & (keep > 0)[:, None, None]
        s = s.masked_fill(~valid, float("-inf"))
        return torch.cat([s0.unsqueeze(-1), s], dim=-1).log_softmax(-1), valid

    @staticmethod
    def q_rank(ellk, tok):
        """Rank of the backbone log-prob of each slot's token among the distinct slot tokens (0 = most likely),
        clipped to Q_MAX: #distinct other tokens with a strictly larger log-prob. ellk, tok [B,T,K]."""
        K = tok.shape[-1]
        has = tok >= 0
        same = (tok.unsqueeze(-1) == tok.unsqueeze(-2)) & has.unsqueeze(-2)          # [B,T,k,j]
        tri = torch.ones(K, K, dtype=torch.bool, device=tok.device).tril(-1)          # j < k
        first = has & ~(same & tri).any(-1)                                           # first slot of its token
        above = first.unsqueeze(-2) & (ellk.unsqueeze(-2) > ellk.unsqueeze(-1))       # [B,T,k,j]
        return above.sum(-1).clamp(max=Q_MAX)

    def _log_pi_v2(self, h_last, ell, keep, tok, ln, dist, delta, cnt, rec):
        src = torch.arange(self.K, device=tok.device)
        s = (self.b.weight[:, 0] + self.L(src * (MAX_MATCH + 1) + ln.clamp(0, MAX_MATCH)).squeeze(-1)
             + self.D(dist.clamp(max=V_DIST_MAX)).squeeze(-1)).float()                     # [B,T,K]
        s0 = self.w(self.norm(h_last)).float().squeeze(-1) + self.c0.float()                # [B,T]
        ellk = ell.gather(-1, tok.clamp(min=0))
        if self.verify:
            s = s + F.softplus(self.gamma_t.float()) * ellk.clamp(min=-20.0)
        feat = dict(d=delta, n=cnt, r=rec)
        if "q" in self.feats:
            feat["q"] = self.q_rank(ellk, tok)
        for f in self.feats:
            name, nb = V2_TABLES[f]
            s = s + getattr(self, name)(self.group * nb + feat[f]).squeeze(-1).float()
        valid = (tok >= 0) & (keep > 0)[:, None, None]
        s = s.masked_fill(~valid, float("-inf"))
        return torch.cat([s0.unsqueeze(-1), s], dim=-1).log_softmax(-1), valid

    def mix(self, z, h_last, keep, tok, ln, dist, *v2f):
        """Full log-mixture [B,T,V] (eval paths). Duplicate candidates: every slot of token c carries the
        logsumexp of all slots of c, so out[c] = logaddexp(log pi_0 + l[c], that) is the same for all of them
        (equal to adding them one after the other)."""
        ls = z.log_softmax(-1)
        lp, valid = self.log_pi(h_last, ls.detach(), keep, tok, ln, dist, *v2f)
        inplace = not (torch.is_grad_enabled() and ls.requires_grad)       # no grad: no extra [B,T,V] copy
        out = ls.add_(lp[..., :1]) if inplace else ls + lp[..., :1]
        idx = tok.clamp(min=0)
        lpk = lp[..., 1:].clamp(min=_NEG)                                   # empty slots: mass 0 (exp underflows)
        same = idx.unsqueeze(-1) == idx.unsqueeze(-2)                       # [B,T,K,K]
        m = torch.where(same, lpk.unsqueeze(-2), _NEG).logsumexp(-1)        # [B,T,K]
        new = torch.logaddexp(out.gather(-1, idx), m)
        if inplace:
            return out.scatter_(-1, idx, new)
        # with autograd, scatter's backward gives out[c]'s gradient to every slot that wrote c: keep it on the
        # first one only (the values are identical)
        later = (same & torch.ones(self.K, self.K, dtype=torch.bool, device=idx.device).tril(-1)).any(-1)
        return out.scatter(-1, idx, torch.where(later, new.detach(), new))

    def loss_sum(self, z, h_last, labels, keep, tok, ln, dist, *v2f, aux: float, listwise: float = 0.0):
        """Training loss summed over the tokens: -log p_mix(y) + aux * CE(z, y) (+ the listwise slot term), only
        gathers (no mixture over the vocabulary)."""
        ls = z.log_softmax(-1)
        lp, valid = self.log_pi(h_last, ls.detach(), keep, tok, ln, dist, *v2f)
        y = labels.clamp(min=0)
        ly = ls.gather(-1, y.unsqueeze(-1)).squeeze(-1)                     # [B,T]
        is_y = valid & (tok == y.unsqueeze(-1))
        hit = torch.where(is_y, lp[..., 1:], _NEG).logsumexp(-1)
        nll = -torch.logaddexp(lp[..., 0] + ly, hit) - aux * ly
        if listwise:
            nll = nll + listwise * self.listwise_nll(lp, valid, tok, is_y, hit)
        return nll[labels != -100].sum()

    @staticmethod
    def listwise_nll(lp, valid, tok, is_y, hit):
        """-log sum_{c_k = y} softmax_{valid slots}(s)_k where y is in the slots and they hold >= 2 distinct tokens,
        else 0 [B,T]. lp = log pi (the sentinel's share cancels), hit = logsumexp of log pi over the y slots."""
        same_before = (tok.unsqueeze(-1) == tok.unsqueeze(-2)) & valid.unsqueeze(-2) & torch.ones(
            tok.shape[-1], tok.shape[-1], dtype=torch.bool, device=tok.device).tril(-1)
        n_distinct = (valid & ~same_before.any(-1)).sum(-1)
        on = is_y.any(-1) & (n_distinct >= 2)
        all_ = torch.where(valid, lp[..., 1:], _NEG).logsumexp(-1)
        return torch.where(on, all_ - hit, torch.zeros_like(hit))


# ---------------------------------------------------------------------------
# Inference cache
# ---------------------------------------------------------------------------
class LayerCache:
    """Decode state of one block: k, v [B, n_kv, max_len, D] (attention heads); conv [B, conv-1, 3*inner] and
    S [B, H, D, D] fp32 (GDN heads). Unused fields stay None."""

    def __init__(self):
        self.k = self.v = self.conv = self.S = None

    def tensors(self):
        return {n: t for n in ("k", "v", "conv", "S") if (t := getattr(self, n)) is not None}


class InferenceCache:
    def __init__(self, layers: List[LayerCache], max_len: int):
        self.layers, self.max_len, self.pos = layers, max_len, 0

    def nbytes(self) -> Dict[str, int]:
        """Bytes per kind: kv (grows with max_len), gdn_state + conv (fixed size)."""
        out = dict(kv=0, gdn_state=0, conv=0)
        for lc in self.layers:
            for n, t in lc.tensors().items():
                out["kv" if n in ("k", "v") else ("gdn_state" if n == "S" else "conv")] += t.numel() * t.element_size()
        return out


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------
class ICELM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.plan = layer_plan(cfg)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, p) for p in self.plan])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        n_orders = len(cfg.ice_orders)
        use_in = cfg.ice_v_input if cfg.ice_v else cfg.ice_input       # ICE-V: no input side unless asked
        self.ice_in = ICEInput(cfg.d_model, n_orders, cfg.ice_trust) if (cfg.ice and use_in) else None
        self.ice_out = ICEOutput(cfg.d_model, cfg.ice_trust) if (cfg.ice and cfg.ice_output and not cfg.ice_v) else None
        if cfg.ice and cfg.ice_v:
            v2 = dict(v2=True, groups=slot_groups(cfg.ice_v, n_orders), feats=cfg.ice_v2_feats) if cfg.ice_v2 else {}
            self.ice_ptr = ICEPointer(cfg.d_model, n_slots(cfg.ice_v, n_orders), cfg.ice_v_verify, **v2)
        else:
            self.ice_ptr = None
        self._init_weights()

    def _init_weights(self):
        std = self.cfg.init_std
        for name, mod in self.named_modules():
            if name.startswith("ice_"):
                continue
            if isinstance(mod, nn.Linear):
                s = std / math.sqrt(2 * self.cfg.n_layers) if getattr(mod, "_is_residual_out", False) else std
                nn.init.normal_(mod.weight, 0.0, s)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, 0.0, std)

    # -- utilities -------------------------------------------------------------
    def prepare_backend(self, device, dtype):
        """Run the GDN kernel self-test once (outside of autograd) for this device/dtype."""
        if any(p["gdn"] > 0 for p in self.plan):
            set_torch_kernel_compile(self.cfg.gdn_compile)
            resolve_backend(self.cfg.gdn_backend, torch.device(device), dtype, self.cfg.head_dim)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def flops_per_token(self, seq_len: int) -> float:
        """Approximate training FLOPs per token (forward + backward = 3x forward): 6 x (non-embedding params + the
        output projection) + causal attention (6 * seq_len * attention width per layer) + GDN state updates
        (~18 * head_dim^2 per GDN head). The comparison between arms is at equal tokens; report this next to it."""
        n = self.num_params(non_embedding=True) + self.embed.weight.numel()      # tied output projection
        attn = sum(6 * seq_len * p["attn"] * self.cfg.head_dim for p in self.plan)
        gdn = sum(18 * p["gdn"] * self.cfg.head_dim ** 2 for p in self.plan)
        return float(6 * n + attn + gdn)

    def ice_params(self) -> int:
        return sum(p.numel() for m in (self.ice_in, self.ice_out, self.ice_ptr) if m is not None
                   for p in m.parameters())

    def uses_ice(self) -> bool:
        return self.ice_in is not None or self.ice_out is not None or self.ice_ptr is not None

    def _out_feats(self, ice):
        """Output-side ICE features, each [B,T] or [B,T,K] (time second, so they can be sliced like h_last)."""
        if self.ice_ptr is not None:
            keys = V_KEYS + (V2_KEYS if self.ice_ptr.v2 else ())
            return tuple(ice[k].transpose(1, 2) for k in keys)              # [B,K,T] -> [B,T,K]
        return tuple(ice.get(k) for k in OUT_KEYS) if self.ice_out is not None else (None,) * len(OUT_KEYS)

    # -- inference cache -------------------------------------------------------
    def allocate_cache(self, batch: int, max_len: int, device, dtype=torch.float16) -> "InferenceCache":
        """Preallocated decode state: KV cache for attention heads, conv + recurrent state for GDN heads."""
        layers = []
        for blk, p in zip(self.blocks, self.plan):
            lc = LayerCache()
            if p["attn"] > 0:
                shape = (batch, p["kv"], max_len, self.cfg.head_dim)
                lc.k = torch.zeros(shape, device=device, dtype=dtype)
                lc.v = torch.zeros(shape, device=device, dtype=dtype)
            if p["gdn"] > 0:
                inner = p["gdn"] * self.cfg.head_dim
                lc.conv = torch.zeros(batch, self.cfg.conv_size - 1, 3 * inner, device=device, dtype=dtype)
                lc.S = torch.zeros(batch, p["gdn"], self.cfg.head_dim, self.cfg.head_dim, device=device,
                                   dtype=torch.float32)
            layers.append(lc)
        return InferenceCache(layers, max_len)

    @torch.no_grad()
    def forward_cached(self, input_ids, cache: "InferenceCache", ice: Optional[Dict[str, torch.Tensor]] = None,
                       last_only: bool = True):
        """Process input_ids [B,T] at positions cache.pos .. cache.pos+T-1 (T > 1: prefill / chunked prefill,
        T == 1: decode) and advance the cache. ICE features are those of these T positions (ice_lm.inference
        .ICEIndex computes them incrementally). Returns logits [B,1,V] (last_only) or [B,T,V]."""
        B, T = input_ids.shape
        pos = cache.pos
        if pos + T > cache.max_len:
            raise ValueError(f"cache holds {cache.max_len} positions, need {pos + T}")
        use_ice = self.uses_ice()
        if use_ice and ice is None:
            raise ValueError("this model uses ICE: pass the ICE features of the new positions")
        keep = torch.ones(B, device=input_ids.device)
        h = self.embed(input_ids)
        for i, blk in enumerate(self.blocks):
            h = blk(h, cache.layers[i], pos)
            if self.ice_in is not None and i + 1 == self.cfg.ice_inject_after:
                h = self.ice_in(h, self.embed, ice["ice_succ"], ice["ice_len"], ice["ice_dist"], keep,
                                ice.get("ice_agree"), ice.get("ice_rel"))
        h_last = self.norm(h)
        star = self._out_feats(ice)
        if last_only:
            h_last = h_last[:, -1:]
            star = tuple(None if f is None else f[:, -1:] for f in star)
        cache.pos = pos + T
        return self._logits(h_last, keep, *star)

    # -- forward ---------------------------------------------------------------
    def forward(self, input_ids, labels=None, ice: Optional[Dict[str, torch.Tensor]] = None,
                ice_keep: Optional[torch.Tensor] = None, reduction: str = "mean"):
        B, T = input_ids.shape
        use_ice = self.uses_ice()
        if use_ice and ice is None:
            raise ValueError("this model uses ICE: pass the ICE feature dict (see ice_lm.data)")
        if use_ice and ice_keep is None:
            ice_keep = torch.ones(B, device=input_ids.device)
        h = self.embed(input_ids)
        for i, blk in enumerate(self.blocks):
            if self.cfg.grad_checkpointing and self.training:
                h = checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
            if self.ice_in is not None and i + 1 == self.cfg.ice_inject_after:
                h = self.ice_in(h, self.embed, ice["ice_succ"], ice["ice_len"], ice["ice_dist"], ice_keep,
                                ice.get("ice_agree"), ice.get("ice_rel"))
        h_last = self.norm(h)
        # output-side ICE features, each [B,T] (or None): s*, len*, agree*, rel*; ICE-V: candidates [B,T,K]
        star = self._out_feats(ice)
        if labels is not None and reduction == "mean" and torch.is_grad_enabled():
            # training: never hold the full fp32 [B,T,V] logits (2 x 500 MB at T=4096, V=32K); compute the head
            # and the loss in chunks and recompute them in backward
            total = h_last.new_zeros((), dtype=torch.float32)
            for t0 in range(0, T, self.LOSS_CHUNK):
                sl = slice(t0, t0 + self.LOSS_CHUNK)
                total = total + checkpoint(self._loss_sum, h_last[:, sl], labels[:, sl], ice_keep,
                                           *(None if f is None else f[:, sl] for f in star), use_reentrant=False)
            return None, total / (labels != -100).sum().clamp_min(1)
        logits = self._logits(h_last, ice_keep, *star)
        if labels is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1), reduction=reduction)
        if reduction == "none":
            loss = loss.view(B, T)
        return logits, loss

    LOSS_CHUNK = 1024

    def _logits(self, h_last, keep, *star):
        """Logits, or for ICE-V the log-mixture (normalized log-probs: CE / argmax / log_softmax on it are
        unchanged). star: s*, len*, agree*, rel* (ICE) or the candidate slots (ICE-V)."""
        w = self.embed.weight if self.lm_head is None else self.lm_head.weight
        logits = F.linear(h_last, w).float()
        if self.ice_out is not None:
            s_star, len_star, agree_star, rel_star = star
            logits = self.ice_out(logits, h_last, s_star, len_star, keep, agree_star, rel_star)
        if self.ice_ptr is not None:
            logits = self.ice_ptr.mix(logits, h_last, keep, *star)
        return logits

    def _loss_sum(self, h_last, labels, keep, *star):
        if self.ice_ptr is not None:
            w = self.embed.weight if self.lm_head is None else self.lm_head.weight
            return self.ice_ptr.loss_sum(F.linear(h_last, w).float(), h_last, labels, keep, *star,
                                         aux=self.cfg.ice_v_aux, listwise=self.cfg.ice_v_listwise)
        logits = self._logits(h_last, keep, *star)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")
