"""Gated DeltaNet (GDN) heads with two interchangeable kernels.

Recurrence (per head, state S in R^{K x V}):
    S_t = exp(g_t) * S_{t-1} + beta_t * k_t (v_t - exp(g_t) * S_{t-1}^T k_t)^T
    o_t = S_t^T (scale * q_t)

Kernels
  * "fla"   : flash-linear-attention's Triton chunk kernel (fast; needs a recent GPU + Triton).
  * "torch" : pure-PyTorch chunked (WY / UT-transform) implementation. Works everywhere
              (CPU, T4, P100, ...). Slower, but exact.
"auto" runs a numerical self-test of fla against the torch reference on the actual device / dtype
and silently falls back to torch if fla is missing, broken, or disagrees.
"""
from __future__ import annotations

import inspect
import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def gated_delta_rule_recurrent(q, k, v, g, beta, scale: float):
    """Naive step-by-step reference. q,k: [B,T,H,K], v: [B,T,H,V], g,beta: [B,T,H]."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    S = q.new_zeros(B, H, K, V)
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt, vt = k[:, t], v[:, t]                                   # [B,H,K], [B,H,V]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + beta[:, t][..., None, None] * torch.einsum("bhk,bhv->bhkv", kt, vt - pred)
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
    return torch.stack(out, dim=1)


def gated_delta_rule_chunked(q, k, v, g, beta, scale: float, chunk_size: int = 64,
                             initial_state: Optional[torch.Tensor] = None, return_state: bool = False):
    """Chunked WY-form implementation (exact). Same shapes as the recurrent reference.
    Always computes in fp32 (autocast disabled), returns the input dtype.
    initial_state / returned state: [B,H,K,V] fp32 (used by the decode cache)."""
    global _COMPILE_TORCH_KERNEL
    with torch.autocast(device_type=q.device.type, enabled=False):
        if _COMPILE_TORCH_KERNEL and initial_state is None and torch.is_grad_enabled():   # training only
            try:
                o, S = _compiled_chunked()(q, k, v, g, beta, scale, chunk_size, None)
                return (o, S) if return_state else o
            except Exception as e:  # noqa: BLE001 - e.g. no compiler toolchain: fall back to eager
                warnings.warn(f"[gdn] torch.compile of the PyTorch kernel failed ({type(e).__name__}); using eager")
                _COMPILE_TORCH_KERNEL = False
        o, S = _chunked_fp32(q, k, v, g, beta, scale, chunk_size, initial_state)
    return (o, S) if return_state else o


_COMPILE_TORCH_KERNEL = False
_COMPILED = None


def set_torch_kernel_compile(on: bool):
    """Opt-in: torch.compile the pure-PyTorch chunked kernel (fewer kernel launches / fused elementwise ops; may help
    on GPUs where fla is unavailable, e.g. T4). Off by default: measure tok/s before relying on it."""
    global _COMPILE_TORCH_KERNEL
    _COMPILE_TORCH_KERNEL = bool(on)


def _compiled_chunked():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = torch.compile(_chunked_fp32, dynamic=False)
    return _COMPILED


def gated_delta_rule_step(q, k, v, g, beta, scale: float, S: torch.Tensor):
    """One decode step of the recurrence. q,k: [B,H,K], v: [B,H,V], g,beta: [B,H], S: [B,H,K,V] fp32.
    Returns (o [B,H,V] fp32, new S)."""
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    S = S * g.exp()[..., None, None]
    pred = torch.einsum("bhk,bhkv->bhv", k, S)
    S = S + beta[..., None, None] * torch.einsum("bhk,bhv->bhkv", k, v - pred)
    return torch.einsum("bhk,bhkv->bhv", q * scale, S), S


def _chunked_fp32(q, k, v, g, beta, scale: float, chunk_size: int, initial_state=None):
    in_dtype = q.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    pad = (-T) % C
    if pad:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
    Tp = T + pad
    N = Tp // C

    def to_chunks(x):  # [B,Tp,H,*] -> [B,H,N,C,*]
        x = x.transpose(1, 2)
        return x.reshape(B, H, N, C, *x.shape[3:])

    q, k, v = to_chunks(q) * scale, to_chunks(k), to_chunks(v)
    g, beta = to_chunks(g), to_chunks(beta)
    gc = g.cumsum(-1)                                                # [B,H,N,C]

    idx = torch.arange(C, device=q.device)
    lower = idx[:, None] >= idx[None, :]                             # incl. diagonal
    strict = idx[:, None] > idx[None, :]
    diff = gc[..., :, None] - gc[..., None, :]
    gam = torch.where(lower, diff, torch.full_like(diff, -float("inf"))).exp()   # decay i<-j

    kb = k * beta[..., None]
    A = (kb @ k.transpose(-1, -2)) * gam
    A = A.masked_fill(~strict, 0.0)
    eye = torch.eye(C, device=q.device, dtype=q.dtype)
    Tinv = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False)   # (I + A)^{-1}
    U = Tinv @ (v * beta[..., None])                                 # [B,H,N,C,V]
    W = Tinv @ (kb * gc.exp()[..., None])                            # [B,H,N,C,K]

    S = q.new_zeros(B, H, K, V) if initial_state is None else initial_state.float()
    outs = []
    for n in range(N):
        qn, kn, gcn = q[:, :, n], k[:, :, n], gc[:, :, n]
        v_new = U[:, :, n] - W[:, :, n] @ S
        attn = (qn @ kn.transpose(-1, -2)) * gam[:, :, n]
        outs.append((qn * gcn.exp()[..., None]) @ S + attn @ v_new)
        decay_to_end = (gcn[..., -1:] - gcn).exp()[..., None]       # [B,H,C,1]
        S = S * gcn[..., -1].exp()[..., None, None] + (kn * decay_to_end).transpose(-1, -2) @ v_new
    o = torch.stack(outs, dim=2).reshape(B, H, Tp, V)[:, :, :T]
    return o.transpose(1, 2).to(in_dtype), S      # padded steps have k=v=beta=g=0: S is the state after T


_FLA_FN = None
_FLA_KW: dict = {}


def _fla_call(q, k, v, g, beta, scale, initial_state=None, return_state: bool = False):
    global _FLA_FN, _FLA_KW
    if _FLA_FN is None:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        _FLA_FN = chunk_gated_delta_rule
        # old fla defaulted to head_first=True ([B,H,T,D]); newer fla removed the argument and raises if it is
        # passed (layout is always [B,T,H,D]). The self-test catches a wrong layout either way.
        try:
            p = inspect.signature(inspect.unwrap(_FLA_FN)).parameters.get("head_first")
        except (TypeError, ValueError):
            p = None
        _FLA_KW = dict(head_first=False) if (p is not None and p.default is True) else {}
    o, S = _FLA_FN(q, k, v, g, beta, scale=scale, initial_state=initial_state, output_final_state=return_state,
                   **_FLA_KW)
    return (o, S) if return_state else o


_BACKEND_CACHE = {}


def _fla_self_test(device, dtype, head_dim):
    gen = torch.Generator(device="cpu").manual_seed(0)
    B, T, H, D = 2, 200, 2, head_dim
    q = F.normalize(torch.randn(B, T, H, D, generator=gen), dim=-1).to(device)
    k = F.normalize(torch.randn(B, T, H, D, generator=gen), dim=-1).to(device)
    v = torch.randn(B, T, H, D, generator=gen).to(device)
    g = (-torch.rand(B, T, H, generator=gen) * 0.2).to(device)
    beta = torch.rand(B, T, H, generator=gen).to(device)
    scale = D ** -0.5
    ref = gated_delta_rule_chunked(q, k, v, g, beta, scale)
    qd, kd, vd = (t.to(dtype).requires_grad_() for t in (q, k, v))
    out = _fla_call(qd, kd, vd, g.float(), beta.to(dtype), scale)
    err = ((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
    out.float().sum().backward()
    grads_ok = all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in (qd, kd, vd))
    tol = 1e-3 if dtype == torch.float32 else 3e-2
    # the inference cache (chunked prefill) passes the state between calls: check a split run too
    state_ok = False
    try:
        with torch.no_grad():
            h = T // 2
            a = [t.to(dtype) for t in (q, k, v)]
            b = beta.to(dtype)
            o1, S1 = _fla_call(a[0][:, :h], a[1][:, :h], a[2][:, :h], g[:, :h].float(), b[:, :h], scale,
                               return_state=True)
            o2, _ = _fla_call(a[0][:, h:], a[1][:, h:], a[2][:, h:], g[:, h:].float(), b[:, h:], scale,
                              initial_state=S1, return_state=True)
            o12 = torch.cat([o1, o2], dim=1).float()
            state_ok = ((o12 - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item() < tol
    except Exception:  # noqa: BLE001
        state_ok = False
    _FLA_STATE_OK[(str(_norm_device(device)), dtype, head_dim)] = state_ok
    return (err < tol and grads_ok), f"rel_err={err:.2e}, grads_ok={grads_ok}, state_ok={state_ok}"


_FLA_STATE_OK = {}


def fla_state_ok(device, dtype, head_dim) -> bool:
    """Whether fla also passed the split (initial_state / final state) self-test; else the cache path uses torch."""
    return _FLA_STATE_OK.get((str(_norm_device(device)), dtype, head_dim), False)


def _norm_device(device) -> torch.device:
    """"cuda" and "cuda:0" must share one cache entry (prepare_backend gets the former, forward sees the latter)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def resolve_backend(requested: str, device: torch.device, dtype: torch.dtype, head_dim: int) -> str:
    device = _norm_device(device)
    if requested == "torch" or device.type != "cuda":
        if requested == "fla" and device.type != "cuda":
            raise RuntimeError("gdn_backend='fla' needs a CUDA device")
        return "torch"
    key = (requested, str(device), dtype, head_dim)
    if key in _BACKEND_CACHE:
        return _BACKEND_CACHE[key]
    ok, reason = False, ""
    try:
        # the self-test checks gradients too: it must work even if first reached inside torch.no_grad()
        with torch.autocast(device_type="cuda", enabled=False), torch.enable_grad():
            ok, reason = _fla_self_test(device, dtype, head_dim)
    except Exception as e:  # noqa: BLE001 - any failure means "do not use fla"
        reason = f"{type(e).__name__}: {e}"
    if requested == "fla" and not ok:
        raise RuntimeError(f"fla backend requested but self-test failed ({reason})")
    backend = "fla" if ok else "torch"
    if requested == "auto":
        msg = f"[gdn] backend=auto -> {backend} ({reason})"
        print(msg) if ok else warnings.warn(msg + " ; using the pure-PyTorch kernel")
    _BACKEND_CACHE[key] = backend
    return backend


# ---------------------------------------------------------------------------
# GDN heads module
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class ShortConv(nn.Module):
    """Depthwise causal conv1d + SiLU over [B,T,C]."""

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, padding=kernel - 1, bias=False)

    def forward(self, x):
        T = x.shape[1]
        y = self.conv(x.transpose(1, 2))[..., :T]
        return F.silu(y.transpose(1, 2))

    def step(self, x, state):
        """Cached version for prefill / decode. state: [B, kernel-1, C] = the last pre-conv inputs (zeros at the
        start, which reproduces the left zero-padding of forward). Returns (output, new state)."""
        xin = torch.cat([state.to(x.dtype), x], dim=1)                # [B, kernel-1+T, C]
        y = F.conv1d(xin.transpose(1, 2), self.conv.weight.to(x.dtype), groups=xin.shape[-1])
        return F.silu(y.transpose(1, 2)), xin[:, xin.shape[1] - (self.kernel - 1):]


class GatedDeltaNetHeads(nn.Module):
    """n_heads GDN heads of size head_dim. Returns [B, T, n_heads * head_dim] (before o_proj)."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, conv_size: int = 4,
                 backend: str = "auto", chunk_size: int = 64):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.backend, self.chunk_size = backend, chunk_size
        inner = n_heads * head_dim
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)     # decay
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)     # write strength beta
        self.g_proj = nn.Linear(d_model, inner, bias=False)       # output gate
        self.conv = ShortConv(3 * inner, conv_size)
        A = torch.empty(n_heads).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(n_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        self.o_norm = RMSNorm(head_dim)

    def forward(self, x, cache=None):
        """cache (inference only): object with .conv [B, conv-1, 3*inner] and .S [B,H,D,D] fp32, updated in place."""
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        proj = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        if cache is None:
            qkv = self.conv(proj)
        else:
            qkv, cache.conv = self.conv.step(proj, cache.conv)
        q, k, v = qkv.view(B, T, 3, H, D).unbind(2)
        q = F.normalize(q.float(), dim=-1).to(v.dtype)
        k = F.normalize(k.float(), dim=-1).to(v.dtype)
        beta = torch.sigmoid(self.b_proj(x).float())
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias.float())
        scale = D ** -0.5
        backend = resolve_backend(self.backend, x.device, v.dtype, D)
        if cache is not None and T == 1:                              # decode: one recurrent step
            o, cache.S = gated_delta_rule_step(q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], scale, cache.S)
            o = o[:, None]
        elif backend == "fla" and (cache is None or fla_state_ok(x.device, v.dtype, D)):
            args = (q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.to(v.dtype).contiguous(),
                    scale)
            if cache is None:
                o = _fla_call(*args)
            else:
                o, cache.S = _fla_call(*args, initial_state=cache.S, return_state=True)
        elif cache is None:
            o = gated_delta_rule_chunked(q, k, v, g, beta, scale, self.chunk_size)
        else:
            o, cache.S = gated_delta_rule_chunked(q, k, v, g, beta, scale, self.chunk_size, cache.S, True)
        o = self.o_norm(o.to(x.dtype)) * F.silu(self.g_proj(x)).view(B, T, H, D)
        return o.reshape(B, T, H * D)
