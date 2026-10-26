#!/usr/bin/env python
"""D1: behaviour beyond the 4K training length.

    python scripts/long_context_eval.py --runs /vol/runs/r3090_B /vol/runs/r3090_C --data_dir /vol/data \
        --clamp_dist --out /vol/long_ctx_r3090.json

(a) Held-out code loss by position bucket (0-4K, 4-8K, 8-16K, 16-32K) and token group (all / copyable = longest
    match >= 4 / lk_right / lk_wrong / lk_none / copy_far) on GENUINE single documents from val_code that are
    longer than --min_doc tokens. Each document is scored alone (no packing; truncated at --max_len), so ICE never
    resets inside it: concatenating documents would not test ICE, which resets at every EOS. Groups are defined by
    the exact lookup alone, so they are identical tokens for every arm.
(b) CodeRecall at gaps 6144 / 12288 / 24576 (ice_lm.coderecall.run_coderecall, unchanged).

--clamp_dist: for ICE arms, clamp the distance-bucket feature ice_dist (the only distance bucket the model reads;
ICEOutput uses s*/len*/agree*/rel* only, and ice_dist_star is a raw distance used for the eval split copy_far)
to the largest bucket reachable at the training length (bucket 12 for 4096: dist <= 4095). This is done by
wrapping model.forward, so eval_loss_splits-style scoring and score_answers (CodeRecall) both see clamped features
and the model's own forward is used everywhere. --compare_unclamped also scores the unclamped ICE arm.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ice_lm.coderecall import CATEGORIES, run_coderecall, summarize  # noqa: E402
from ice_lm.config import model_config_from_dict  # noqa: E402
from ice_lm.data import ICE_KEYS, load_meta, open_tokens  # noqa: E402
from ice_lm.evals import _autocast, uses_ice  # noqa: E402
from ice_lm.ice_features import _dist_bucket, compute_ice_features  # noqa: E402
from ice_lm.model import ICELM  # noqa: E402
from ice_lm.utils import pick_precision  # noqa: E402

BUCKETS = ((0, 4096), (4096, 8192), (8192, 16384), (16384, 32768))
GROUPS = ("all", "copyable", "other", "copy_far", "lk_right", "lk_wrong", "lk_none")


def bname(lo, hi):
    return f"{lo // 1024}-{hi // 1024}K"


def long_docs(data, eos: int, min_len: int):
    """(start, length) of every document (EOS-delimited, EOS excluded) with length > min_len."""
    arr = np.asarray(data)
    ends = np.nonzero(arr == eos)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    lens = ends - starts
    keep = lens > min_len
    return list(zip(starts[keep].tolist(), lens[keep].tolist())), lens


def clamp_wrap(model, max_bucket: int):
    """Wrap model.forward so every call clamps ice['ice_dist'] to max_bucket (no library change)."""
    orig = model.forward

    def fwd(*args, **kw):
        ice = kw.get("ice")
        if ice is not None:             # every caller (this script, score_answers) passes ice as a keyword
            ice = dict(ice)
            for k in ("ice_dist", "ice_v_dist"):    # ice_v_dist: candidate-slot buckets of ICE-V/VR arms, if any
                if k in ice:
                    ice[k] = ice[k].clamp(max=max_bucket)
            kw["ice"] = ice
        return orig(*args, **kw)

    model.forward = fwd
    return lambda: model.__dict__.pop("forward", None)    # undo: back to the class forward


@torch.no_grad()
def long_doc_loss(model, docs, data, device, amp_dtype, max_len: int, far: int = 512, copy_min_len: int = 4):
    model.eval()
    with_ice = uses_ice(model)
    tot = {(b, g): [0.0, 0, 0] for b in range(len(BUCKETS)) for g in GROUPS}
    per_doc = []
    for start, length in docs:
        L = min(length, max_len + 1)
        w = np.asarray(data[start: start + L]).astype(np.int64)
        x_np, y_np = w[:-1], w[1:]
        f = compute_ice_features(x_np, None, model.cfg.ice_orders)       # one document: no EOS inside
        x = torch.from_numpy(x_np).to(device)[None]
        y = torch.from_numpy(y_np).to(device)[None]
        ice = {k: torch.from_numpy(f[k]).to(device)[None] for k in ICE_KEYS} if with_ice else None
        keep = torch.ones(1, device=device) if with_ice else None
        with _autocast(device, amp_dtype):
            logits, loss = model(x, y, ice=ice, ice_keep=keep, reduction="none")
        loss = loss[0].float().cpu().numpy()
        correct = (logits[0].argmax(-1) == y[0]).cpu().numpy()
        del logits
        ss, ml, ds = f["ice_s_star"], f["ice_match_len"], f["ice_dist_star"]
        cp = ml >= copy_min_len
        masks = dict(all=np.ones_like(cp), copyable=cp, other=~cp, copy_far=cp & (ds > far), lk_right=ss == y_np,
                     lk_wrong=(ss >= 0) & (ss != y_np), lk_none=ss < 0)
        pos = np.arange(len(x_np))
        d = {}
        for b, (lo, hi) in enumerate(BUCKETS):
            inb = (pos >= lo) & (pos < hi)
            for g, m in masks.items():
                mm = inb & m
                s, n, c = float(loss[mm].sum()), int(mm.sum()), int(correct[mm].sum())
                tot[(b, g)][0] += s
                tot[(b, g)][1] += n
                tot[(b, g)][2] += c
                if g in ("all", "copyable"):
                    d[f"{bname(lo, hi)}/{g}"] = [s, n]
        per_doc.append(d)
    model.train()
    res = {}
    for (b, g), (s, n, c) in tot.items():
        k = f"long/{bname(*BUCKETS[b])}/{g}"
        res[k + "_loss"] = s / n if n else float("nan")
        res[k + "_acc"] = c / n if n else float("nan")
        res[k + "_n"] = n
    return res, per_doc


def load_model(run, device, amp_dtype):
    cfg = json.load(open(os.path.join(run, "config.json")))
    mcfg = model_config_from_dict(cfg["model"])
    model = ICELM(mcfg).to(device)
    path = os.path.join(run, "model_final.pt")
    if not os.path.exists(path):
        path = os.path.join(run, "ckpt_latest.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.prepare_backend(device, amp_dtype or torch.float32)
    return model, cfg, ck.get("state", {}).get("tokens", 0), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--split", default="val_code")
    ap.add_argument("--min_doc", type=int, default=8192, help="documents longer than this many tokens")
    ap.add_argument("--max_len", type=int, default=32768, help="score at most this many positions per document")
    ap.add_argument("--max_docs", type=int, default=0, help="0 = all long documents")
    ap.add_argument("--coderecall_gaps", type=int, nargs="+", default=[6144, 12288, 24576])
    ap.add_argument("--n_coderecall", type=int, default=100)
    ap.add_argument("--skip_coderecall", action="store_true")
    ap.add_argument("--clamp_dist", action="store_true", help="ICE arms: clamp ice_dist to the training maximum")
    ap.add_argument("--compare_unclamped", action="store_true", help="with --clamp_dist, also score unclamped")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--out", default="long_ctx.json")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, _ = pick_precision(a.precision, device)
    meta = load_meta(a.data_dir)
    eos = meta["eos_id"]
    data = open_tokens(a.data_dir, a.split)
    docs, all_lens = long_docs(data, eos, a.min_doc)
    if a.max_docs:
        docs = docs[: a.max_docs]
    n_scored = [min(l, a.max_len) for _, l in docs]
    pos_counts = {bname(lo, hi): int(sum(max(0, min(n, hi) - lo) for n in n_scored)) for lo, hi in BUCKETS}
    doc_info = dict(split=a.split, split_tokens=int(len(data)), n_docs_total=int(len(all_lens)),
                    min_doc=a.min_doc, n_long_docs=len(docs), long_doc_tokens=int(sum(l for _, l in docs)),
                    scored_positions=int(sum(n_scored)), positions_per_bucket=pos_counts,
                    docs_over_16k=int(sum(l > 16384 for _, l in docs)), docs_over_32k=int(sum(l > 32768 for _, l in docs)),
                    longest_doc=int(max([l for _, l in docs], default=0)))
    print("long documents:", json.dumps(doc_info, indent=1), flush=True)
    if not docs:
        print(f"NO document in {a.split} is longer than {a.min_doc} tokens: part (a) cannot be measured")

    tok = None
    if meta.get("tokenizer") and not a.skip_coderecall:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(meta["tokenizer"])

    all_res = {"_docs": doc_info}
    for run in a.runs:
        model, cfg, tokens, path = load_model(run, device, amp_dtype)
        name = cfg["train"]["arm"]
        train_len = cfg["train"]["seq_len"]
        max_bucket = int(_dist_bucket(np.array([train_len - 1]))[0])
        ice = uses_ice(model)
        variants = [(name, False)]
        if ice and a.clamp_dist:
            variants = [(name + "_clamp", True)] + ([(name, False)] if a.compare_unclamped else [])
        for label, clamp in variants:
            undo = clamp_wrap(model, max_bucket) if clamp else None
            t0 = time.time()
            print(f"\n### {label} ({path}, {tokens / 1e6:,.1f}M tokens, train len {train_len}, "
                  f"clamp ice_dist <= {max_bucket if clamp else '-'})", flush=True)
            res = dict(arm=label, run=run, tokens=tokens, train_seq_len=train_len, clamp_dist=clamp,
                       clamp_bucket=max_bucket if clamp else None)
            if docs:
                r, per_doc = long_doc_loss(model, docs, data, device, amp_dtype, a.max_len)
                res.update(r)
                res["per_doc"] = per_doc
                print(f"  long-doc loss done in {time.time() - t0:.0f}s", flush=True)
            if tok is not None:
                res.update(run_coderecall(model, tok, open_tokens(a.data_dir, "val_code"), eos, device, amp_dtype,
                                          tuple(a.coderecall_gaps), a.n_coderecall))
                for cat, v in summarize(res, "dec_acc").items():
                    res[f"cr/{cat}"] = v
                print(f"  coderecall done in {time.time() - t0:.0f}s", flush=True)
            if undo:
                undo()
            all_res[f"{run}#{label}"] = res
            with open(a.out, "w") as f:                     # save after every arm
                json.dump(all_res, f, indent=1)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    arms = [r for k, r in all_res.items() if not k.startswith("_")]

    def fmt(v):
        return f"{v:12.4f}" if isinstance(v, float) and not math.isnan(v) else f"{'-':>12}"

    print(f"\n== held-out code loss on {doc_info['n_long_docs']} single documents > {a.min_doc} tokens "
          f"({doc_info['scored_positions']:,} positions; per bucket {pos_counts})")
    print("bucket/group".ljust(26) + "".join(r["arm"].rjust(12) for r in arms) + "       n")
    for lo, hi in BUCKETS:
        for g in ("all", "copyable", "copy_far", "lk_right", "lk_wrong", "lk_none"):
            k = f"long/{bname(lo, hi)}/{g}"
            n = arms[0].get(k + "_n", 0) if arms else 0
            print(f"{bname(lo, hi) + ' ' + g:26s}" + "".join(fmt(r.get(k + "_loss")) for r in arms) + f"{n:8d}")
    print("\n== degradation vs 0-4K (loss in bucket minus loss at 0-4K; D1 drops the long-context claim if B "
          "degrades by < 0.10 beyond 8K)")
    for g in ("all", "copyable"):
        for lo, hi in BUCKETS[1:]:
            k, k0 = f"long/{bname(lo, hi)}/{g}_loss", f"long/0-4K/{g}_loss"
            print(f"{bname(lo, hi) + ' ' + g:26s}" + "".join(fmt(r.get(k, float('nan')) - r.get(k0, float('nan')))
                                                            for r in arms))
    if tok is not None:
        print(f"\n== CodeRecall deciding-digit accuracy, gaps {a.coderecall_gaps} (mean over gaps)")
        print("category".ljust(26) + "".join(r["arm"].rjust(12) for r in arms))
        for cat in CATEGORIES:
            print(cat.ljust(26) + "".join(fmt(r.get(f"cr/{cat}")) for r in arms))
        print("\nper gap:")
        for cat in CATEGORIES:
            for gap in (a.coderecall_gaps if cat not in ("increment", "track") else a.coderecall_gaps[:1]):
                k = f"coderecall/{cat}@{gap}/dec_acc"
                print(f"{cat + '@' + str(gap):26s}" + "".join(fmt(r.get(k)) for r in arms))
    with open(a.out, "w") as f:
        json.dump(all_res, f, indent=1)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
