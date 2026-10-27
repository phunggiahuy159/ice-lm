#!/usr/bin/env python
"""Code line completion, exact match: the code use case that held-out loss does not measure.

    python scripts/line_completion.py --runs runs/t200_B runs/t200_G_VR --data_dir data --out linecomp.json

Items (built from the data only, so identical for every arm): in held-out code windows (seq_len tokens, laid out
like the loss eval), lines that start >= --min_ctx tokens into the window, are 3..--max_line tokens long including
the newline, stay inside one document and contain non-blank text. The model reads the window up to the start of the
line and decodes greedily with its inference cache (ICE arms update their index token by token, as at inference)
until it emits a newline or --max_line tokens. Metrics, overall and per kind of line:
  em        the generated line equals the reference exactly (tokens, newline included)
  edit_sim  difflib character similarity of the two lines (as in CrossCodeEval)
  prefix    leading tokens right / reference length
kinds:
  repeat    the exact line already appears earlier in the context
  reuse     not a repeat, but it contains an identifier (>= 3 chars, not a keyword) seen earlier in the context
  novel     neither
"""
import argparse
import difflib
import json
import keyword
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ice_lm.config import model_config_from_dict  # noqa: E402
from ice_lm.data import alpha_mask_for_data, load_meta, open_tokens  # noqa: E402
from ice_lm.ice_features import segments_from_eos, set_alpha_mask  # noqa: E402
from ice_lm.inference import ICEIndex, features_to_tensors, step_features_to_tensors  # noqa: E402
from ice_lm.model import ICELM  # noqa: E402
from ice_lm.utils import pick_precision  # noqa: E402

KINDS = ("repeat", "reuse", "novel")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
COMMON = set(keyword.kwlist) | {"self", "cls", "the", "and", "for", "int", "str", "len", "var", "let", "const",
                                "function", "public", "private", "static", "void", "return", "true", "false", "null"}


def load_run(run, data_dir, device, amp_dtype):
    cfg = json.load(open(os.path.join(run, "config.json")))
    mcfg = model_config_from_dict(cfg["model"])
    if mcfg.ice and mcfg.ice_v == "vr":
        set_alpha_mask(alpha_mask_for_data(data_dir))
    model = ICELM(mcfg).to(device)
    model.load_state_dict(torch.load(os.path.join(run, "model_final.pt"), map_location=device,
                                     weights_only=False)["model"])
    model.prepare_backend(device, amp_dtype or torch.float32)
    model.eval()
    return model, cfg["train"]["arm"], cfg["train"]["seq_len"]


def contains(hay: np.ndarray, needle: np.ndarray) -> bool:
    """needle occurs as a contiguous run in hay."""
    if len(needle) > len(hay):
        return False
    for p in np.nonzero(hay[: len(hay) - len(needle) + 1] == needle[0])[0]:
        if np.array_equal(hay[p: p + len(needle)], needle):
            return True
    return False


def build_items(tok, data, eos, nl, seq_len, n_windows, per_window, min_ctx, max_line, seed):
    """-> list of dict(window, start, line): fixed by the data and seed."""
    items = []
    n_win = min(n_windows, (len(data) - 1) // seq_len)
    for j in range(n_win):
        x = np.asarray(data[j * seq_len: (j + 1) * seq_len]).astype(np.int64)
        seg = segments_from_eos(x, eos)
        ends = np.nonzero(x == nl)[0]
        cands = []
        for e0, e1 in zip(ends[:-1], ends[1:]):
            p = int(e0) + 1                                      # line = x[p .. e1], newline included
            if p < min_ctx or not 3 <= e1 - p + 1 <= max_line or seg[p] != seg[e1] or eos in x[p:e1 + 1]:
                continue
            if not tok.decode(x[p:e1].tolist()).strip():
                continue
            cands.append((p, int(e1)))
        rng = np.random.default_rng([seed, j])
        for k in rng.permutation(len(cands))[:per_window]:
            p, e1 = cands[k]
            line = x[p:e1 + 1]
            if contains(x[:p], line):
                kind = "repeat"
            else:
                ctx = set(IDENT.findall(tok.decode(x[max(0, p - 3000):p].tolist())))
                ids = set(IDENT.findall(tok.decode(line.tolist()))) - COMMON
                kind = "reuse" if ids & ctx else "novel"
            items.append(dict(window=j, start=p, line=line.tolist(), kind=kind))
    return items


@torch.no_grad()
def check_cached(model, prompt, device, cache_dtype, eos, chunk):
    """The cached prefill (chunked, incremental ICE index) must give the full forward's last-position logits."""
    cfg = model.cfg
    idx = ICEIndex(cfg.ice_orders, eos, cfg.vocab_size, cfg.ice_v, cfg.ice_v2) if model.uses_ice() else None
    feats = idx.prefill(prompt) if idx is not None else None
    x = torch.tensor([prompt], device=device)
    full = model(x, ice=features_to_tensors([feats], device) if feats is not None else None)[0, -1].float()
    cache = model.allocate_cache(1, len(prompt), device, cache_dtype)
    for c0 in range(0, len(prompt), chunk):
        c1 = min(len(prompt), c0 + chunk)
        f = features_to_tensors([{k: v[..., c0:c1] for k, v in feats.items()}], device) if feats is not None else None
        logits = model.forward_cached(x[:, c0:c1], cache, f)
    err = (logits[0, -1].float() - full).abs().max().item()
    assert int(logits[0, -1].argmax()) == int(full.argmax()) and err < 0.05, f"cached decode != forward ({err:.4f})"
    return err


@torch.no_grad()
def complete(model, prompt, nl, max_new, device, cache_dtype, eos, chunk):
    cfg = model.cfg
    cache = model.allocate_cache(1, len(prompt) + max_new + 1, device, cache_dtype)
    idx = ICEIndex(cfg.ice_orders, eos, cfg.vocab_size, cfg.ice_v, cfg.ice_v2) if model.uses_ice() else None
    feats = idx.prefill(prompt) if idx is not None else None
    arr = np.asarray(prompt, dtype=np.int64)
    for c0 in range(0, len(prompt), chunk):
        c1 = min(len(prompt), c0 + chunk)
        f = features_to_tensors([{k: v[..., c0:c1] for k, v in feats.items()}], device) if feats is not None else None
        logits = model.forward_cached(torch.from_numpy(arr[None, c0:c1]).to(device), cache, f)
    out = []
    while True:
        t = int(logits[0, -1].float().argmax())
        out.append(t)
        if t == nl or len(out) >= max_new:
            return out
        f = step_features_to_tensors([idx.step(t)], device) if idx is not None else None
        logits = model.forward_cached(torch.tensor([[t]], device=device), cache, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--per_window", type=int, default=3)
    ap.add_argument("--min_ctx", type=int, default=1024)
    ap.add_argument("--max_line", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk (memory)")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--out", default="linecomp.json")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, _ = pick_precision(a.precision, device)
    meta = load_meta(a.data_dir)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
    nl = next(i for i in tok.encode("a\nb", add_special_tokens=False) if tok.decode([i]) == "\n")
    data = open_tokens(a.data_dir, "val_code")
    eos = meta["eos_id"]

    res, items = {}, None
    for run in a.runs:
        model, arm, seq_len = load_run(run, a.data_dir, device, amp_dtype)
        if items is None:
            items = build_items(tok, data, eos, nl, seq_len, a.windows, a.per_window, a.min_ctx, a.max_line,
                                a.seed)
            if not items:
                raise SystemExit("no lines qualify: lower --min_ctx or use more --windows")
            print(f"[linecomp] {len(items)} lines: " + ", ".join(
                f"{k} {sum(i['kind'] == k for i in items)}" for k in KINDS), flush=True)
        x0 = np.asarray(data[items[0]["window"] * seq_len: items[0]["window"] * seq_len + items[0]["start"]])
        with torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else torch.no_grad():
            err = check_cached(model, x0.astype(np.int64).tolist(), device, amp_dtype or torch.float32, eos, a.chunk)
        print(f"[linecomp] {arm}: cached decode matches the full forward (max |dlogit| {err:.2e})", flush=True)
        t0 = time.time()
        rows = []
        for n_done, it in enumerate(items, 1):
            x = np.asarray(data[it["window"] * seq_len: it["window"] * seq_len + it["start"]]).astype(np.int64)
            ref = it["line"]
            with torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else torch.no_grad():
                gen = complete(model, x.tolist(), nl, a.max_line, device, amp_dtype or torch.float32, eos, a.chunk)
            pre = 0
            while pre < min(len(gen), len(ref)) and gen[pre] == ref[pre]:
                pre += 1
            rows.append(dict(kind=it["kind"], em=int(gen == ref), prefix=pre / len(ref),
                             edit_sim=difflib.SequenceMatcher(None, tok.decode(gen), tok.decode(ref)).ratio()))
            if n_done % 50 == 0:
                print(f"[linecomp] {arm}: {n_done}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
        r = {}
        for kind in ("all",) + KINDS:
            sel = [x for x in rows if kind == "all" or x["kind"] == kind]
            r[f"{kind}/n"] = len(sel)
            for m in ("em", "edit_sim", "prefix"):
                r[f"{kind}/{m}"] = float(np.mean([x[m] for x in sel])) if sel else float("nan")
        r["items_em"] = [x["em"] for x in rows]
        res[arm] = r
        print(f"[linecomp] {arm}: " + "  ".join(f"{k} em {r[k + '/em']:.3f}" for k in ("all",) + KINDS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    arms = list(res)
    print("\nline completion (greedy, exact match / edit similarity)")
    print("metric".ljust(20) + "".join(x.rjust(16) for x in arms))
    for kind in ("all",) + KINDS:
        for m in ("n", "em", "edit_sim", "prefix"):
            vals = [res[x][f"{kind}/{m}"] for x in arms]
            print(f"{kind}/{m}".ljust(20) + "".join((f"{v:16d}" if m == "n" else f"{v:16.3f}") for v in vals))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(results=res, items=[{k: v for k, v in i.items() if k != "line"} for i in items]), f,
                  indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
