#!/usr/bin/env python
"""Inference cost per arm: prefill time, decode speed, and memory, including the ICE lookup and its tables.

    python scripts/benchmark.py --preset 0.6b --arms A B C F --contexts 4096 16384 32768 --data_dir data
    python scripts/benchmark.py --runs runs/0.6b_A runs/0.6b_C ...      # trained weights instead of random ones

Speed and memory do not depend on the weights, so random weights are fine; with --data_dir the prompt and
the decoded tokens are real held-out code (teacher-forced), so ICE hit rates are realistic and identical
across arms. Every arm runs the same loop: ICE lookup (CPU, ICE arms only) -> forward -> argmax -> .item().

Memory model reported per arm (cache memory only: activations / workspace are NOT included, so the max batch
is an upper bound; the measured peak memory per run is printed next to it):
    M(batch, T) = weights + batch * (fixed + per_token * T)
    fixed     = GDN recurrent state (fp32) + conv state (measured) + ICE order-1 table (modelled)
    per_token = KV bytes/token (measured) + ICE bytes/token (compact layout modelled from the distinct n-grams
                counted at the largest context, see ice_lm/inference.py; random tokens = worst case)
Prefill time is reported for the model and for the ICE prompt processing separately (the ICE index is
Python here; a C/GPU implementation would be much faster).
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ice_lm.config import build_configs, model_config_from_dict  # noqa: E402
from ice_lm.data import alpha_mask_for_data, load_meta, open_tokens  # noqa: E402
from ice_lm.ice_features import set_alpha_mask  # noqa: E402
from ice_lm.inference import ICEIndex, features_to_tensors, step_features_to_tensors  # noqa: E402
from ice_lm.model import ICELM, kv_bytes_per_token, layer_plan  # noqa: E402
from ice_lm.utils import pick_precision  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="0.6b")
ap.add_argument("--arms", nargs="+", default=["A", "B", "C", "F"])
ap.add_argument("--runs", nargs="*", default=None, help="trained run folders (override --preset/--arms)")
ap.add_argument("--contexts", type=int, nargs="+", default=[4096, 16384, 32768])
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--decode_steps", type=int, default=64)
ap.add_argument("--warmup_context", type=int, default=1024, help="untimed run per arm first (kernels, caches)")
ap.add_argument("--prefill_chunk", type=int, default=4096)
ap.add_argument("--data_dir", default=None)
ap.add_argument("--precision", default="auto")
ap.add_argument("--budgets_gb", type=float, nargs="+", default=[16, 24, 80])
ap.add_argument("--out", default="benchmark.json")
a = ap.parse_args()
a.contexts = sorted(a.contexts)          # the memory model uses the largest context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype, _ = pick_precision(a.precision, device)
dtype = amp_dtype or torch.float32
vocab = 32000
eos = None
stream = None
if a.data_dir:
    meta = load_meta(a.data_dir)
    vocab, eos = meta["vocab_size"], meta["eos_id"]
    stream = np.asarray(open_tokens(a.data_dir, "val_code"))
# ICE-VR identifier anchors: the tokenizer's alphabetic tokens (random tokens without --data_dir: all count)
set_alpha_mask(alpha_mask_for_data(a.data_dir) if a.data_dir else np.ones(vocab, dtype=bool))


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def mem():
    return torch.cuda.max_memory_allocated() if device.type == "cuda" else 0


def tokens_for(B, T, seed):
    rng = np.random.default_rng(seed)
    if stream is None or len(stream) < T + 1:
        return rng.integers(10, vocab - 10, size=(B, T)).astype(np.int64)
    starts = rng.integers(0, len(stream) - T - 1, size=B)
    return np.stack([stream[s: s + T].astype(np.int64) for s in starts])


def models():
    if a.runs:
        for run in a.runs:
            cfg = json.load(open(os.path.join(run, "config.json")))
            m = ICELM(model_config_from_dict(cfg["model"]))
            path = os.path.join(run, "model_final.pt")
            path = path if os.path.exists(path) else os.path.join(run, "ckpt_latest.pt")
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
            yield cfg["train"]["arm"], m
    else:
        for arm in a.arms:
            yield arm, ICELM(build_configs(a.preset, arm, vocab_size=vocab)[0])


def bench(arm, model, T, steps=None):
    cfg = model.cfg
    B, steps = a.batch, max(1, a.decode_steps if steps is None else steps)
    toks = tokens_for(B, T + steps, seed=T)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    base = mem()
    cache = model.allocate_cache(B, T + steps, device, dtype)
    cache_b = cache.nbytes()
    idx = [ICEIndex(cfg.ice_orders, eos, cfg.vocab_size, cfg.ice_v) for _ in range(B)] if cfg.ice else None

    # ICE prompt processing (CPU, vectorized), timed on its own
    ice_prefill_s, feats = 0.0, None
    if idx:
        t1 = time.perf_counter()
        feats = [ix.prefill(row) for ix, row in zip(idx, toks[:, :T])]
        ice_prefill_s = time.perf_counter() - t1
    # model prefill (chunked, so 32K fits on a 16 GB GPU)
    sync()
    t0 = time.perf_counter()
    for c0 in range(0, T, a.prefill_chunk):
        c1 = min(T, c0 + a.prefill_chunk)
        f = None
        if feats:
            f = features_to_tensors([{k: v[..., c0:c1] for k, v in fr.items()} for fr in feats], device)
        logits = model.forward_cached(torch.from_numpy(toks[:, c0:c1]).to(device), cache, f)
    logits.argmax(-1).cpu()
    sync()
    prefill_s = time.perf_counter() - t0
    peak_prefill = mem() - base

    # decode, teacher-forced on the real continuation
    step_ms, ice_us = [], []
    for s in range(steps):
        sync()
        t0 = time.perf_counter()
        nxt = toks[:, T + s]
        f = None
        if idx:
            t1 = time.perf_counter()
            f = step_features_to_tensors([ix.step(int(t)) for ix, t in zip(idx, nxt)], device)
            ice_us.append((time.perf_counter() - t1) * 1e6)
        logits = model.forward_cached(torch.from_numpy(nxt[:, None]).to(device), cache, f)
        logits.argmax(-1).cpu()                       # every arm syncs once per step, as in real sampling
        step_ms.append((time.perf_counter() - t0) * 1e3)
    warm = step_ms[min(4, len(step_ms) - 1):]          # drop the first steps (allocator / kernel warmup)
    ms = statistics.median(warm)
    ice_b = idx[0].compact_bytes() if idx else dict(total=0, tokens=0, tables=0, reliability=0)
    return dict(
        arm=arm, context=T, batch=B,
        prefill_s=prefill_s, prefill_tok_s=B * T / prefill_s, ice_prefill_s=ice_prefill_s,
        prefill_tok_s_incl_ice=B * T / (prefill_s + ice_prefill_s),
        decode_ms=ms, decode_tok_s=B * 1e3 / ms,
        # the ICE index here is Python: also report decode speed without its CPU time (a C/GPU index is ~free)
        decode_tok_s_model_only=B * 1e3 / max(1e-6, ms - (statistics.median(ice_us) / 1e3 if ice_us else 0.0)),
        ice_us_per_step_median=statistics.median(ice_us) if ice_us else 0.0,
        ice_us_per_step_p99=float(np.percentile(ice_us, 99)) if ice_us else 0.0,
        kv_bytes=cache_b["kv"], gdn_state_bytes=cache_b["gdn_state"], conv_bytes=cache_b["conv"],
        ice_bytes_per_seq=ice_b["total"], ice_table_bytes=ice_b["tables"],
        ice_distinct_ngrams=idx[0].distinct_ngrams() if idx else [],
        peak_prefill_bytes=peak_prefill, peak_decode_bytes=mem() - base,
    )


results, summary = [], {}
for arm, model in models():
    model = model.to(device=device, dtype=dtype).eval()
    model.prepare_backend(device, dtype)
    cfg = model.cfg
    w_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\n### {arm}: {model.num_params()/1e6:.1f}M params ({w_bytes/2**20:.0f} MiB in {dtype}), "
          f"layout {''.join('A' if p['gdn'] == 0 else ('G' if p['attn'] == 0 else 'h') for p in layer_plan(cfg))}")
    rows = []
    for T in [None] + a.contexts:                    # None = warmup run (kernels, caches), not reported
        oom = False
        try:
            r = bench(arm, model, min(a.warmup_context, a.contexts[0]), steps=8) if T is None else bench(arm, model, T)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            oom = True                               # handled outside the except block so its memory is freed
        if oom:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  T={T or 'warmup'}: out of memory")
            continue
        if T is None:
            continue
        r["weights_bytes"] = w_bytes
        rows.append(r)
        print(f"  T={T:>6}  prefill {r['prefill_s']:.2f}s ({r['prefill_tok_s']:,.0f} tok/s, ICE {r['ice_prefill_s']:.2f}s)"
              f"  decode {r['decode_tok_s']:,.1f} tok/s ({r['decode_ms']:.1f} ms/step, ICE {r['ice_us_per_step_median']:.0f} us)"
              f"  KV {r['kv_bytes']/2**20:.1f} MiB  GDN {r['gdn_state_bytes']/2**20:.1f} MiB"
              f"  ICE {r['ice_bytes_per_seq']/2**20:.2f} MiB"
              + (f"  peak {r['peak_decode_bytes']/2**20:.0f} MiB" if device.type == "cuda" else ""), flush=True)
    results += rows
    if not rows:
        continue
    # memory model: per-token slope and fixed part per sequence (B=1 terms)
    el = torch.tensor([], dtype=dtype).element_size()
    kv_tok = kv_bytes_per_token(cfg, el)
    last = rows[-1]
    ice_fixed = cfg.vocab_size * 8 if (cfg.ice and 1 in cfg.ice_orders) else 0
    ice_tok = (last["ice_bytes_per_seq"] - ice_fixed) / (last["context"] + max(1, a.decode_steps)) if cfg.ice else 0.0
    fixed = (last["gdn_state_bytes"] + last["conv_bytes"]) / a.batch + ice_fixed
    per_tok = kv_tok + ice_tok
    s = dict(arm=arm, weights_bytes=w_bytes, kv_bytes_per_token=kv_tok, ice_bytes_per_token=ice_tok,
             fixed_bytes_per_seq=fixed, per_token_bytes=per_tok,
             decode_tok_s={r["context"]: r["decode_tok_s"] for r in rows},
             peak_decode_bytes={r["context"]: r["peak_decode_bytes"] for r in rows})
    for T in (32768, 131072):
        s[f"seq_bytes@{T}"] = fixed + per_tok * T
        for gb in a.budgets_gb:
            s[f"max_batch@{T}@{gb:g}GB"] = max(0, math.floor((gb * 2**30 - w_bytes) / (fixed + per_tok * T)))
    summary[arm] = s
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

with open(a.out, "w") as f:
    json.dump(dict(results=results, summary=summary, device=str(device), dtype=str(dtype)), f, indent=2)

print("\n== memory model per sequence (weights shared): M(batch,T) = W + batch*(fixed + per_token*T)"
      "\n   cache memory only (no activations/workspace): max batch is an upper bound")
hdr = ["arm", "W MiB", "fixed MiB", "KV B/tok", "ICE B/tok", "seq@32K MiB", "seq@128K MiB"] + \
      [f"batch@32K/{gb:g}GB" for gb in a.budgets_gb]
print("".join(h.rjust(16) for h in hdr))
for arm, s in summary.items():
    vals = [arm, s["weights_bytes"] / 2**20, s["fixed_bytes_per_seq"] / 2**20, s["kv_bytes_per_token"],
            s["ice_bytes_per_token"], s["seq_bytes@32768"] / 2**20, s["seq_bytes@131072"] / 2**20] + \
           [s[f"max_batch@32768@{gb:g}GB"] for gb in a.budgets_gb]
    print("".join((f"{v:16.1f}" if isinstance(v, float) else str(v).rjust(16)) for v in vals))
print(f"\nsaved {a.out}")
