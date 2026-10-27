#!/usr/bin/env python
"""Evaluate trained arms and print comparison tables.

    python scripts/evaluate.py --runs runs/A runs/B runs/C --data_dir data --out results.json
Uses model_final.pt if present, otherwise ckpt_latest.pt.

For ICE arms the same evaluations are repeated with ICE switched off ("<arm>_off", keep = 0): if C_off is worse
than B, the model has become lazy (relies on the lookup instead of learning to recall). This control is only
meaningful for arms trained with ICE-dropout (not C_nodrop, for which keep = 0 is out of distribution).

CodeRecall is reported on the deciding digit (see ice_lm/coderecall.py) with paired bootstrap 95% intervals
for the difference to the reference arm (B if present): every arm scores identical items.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ice_lm.coderecall import CATEGORIES, run_coderecall, summarize  # noqa: E402
from ice_lm.config import model_config_from_dict  # noqa: E402
from ice_lm.data import alpha_mask_for_data, load_meta, open_tokens  # noqa: E402
from ice_lm.evals import eval_loss_splits, run_synthetic_tasks, uses_ice  # noqa: E402
from ice_lm.ice_features import set_alpha_mask  # noqa: E402
from ice_lm.model import ICELM, kv_bytes_per_token  # noqa: E402
from ice_lm.utils import pick_precision  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="*", default=[])
ap.add_argument("--from_json", nargs="*", default=[], help="merge results saved by earlier evaluate.py calls "
                "(no model evaluation for those; only the comparison tables)")
ap.add_argument("--data_dir", default="data")
ap.add_argument("--seq_len", type=int, default=None, help="default: training seq_len")
ap.add_argument("--val_batches", type=int, default=50)
ap.add_argument("--batch_size", type=int, default=1)
ap.add_argument("--context_lens", type=int, nargs="+", default=[1024, 4096],
                help="lengths above the training seq_len test extrapolation (attention degrades, ICE does not)")
ap.add_argument("--n_task", type=int, default=50, help="examples per synthetic task and length")
ap.add_argument("--coderecall_gaps", type=int, nargs="+", default=[256, 1024, 3072],
                help="total filler tokens per item; keep below the training seq_len")
ap.add_argument("--n_coderecall", type=int, default=100, help="items per CodeRecall category and gap")
ap.add_argument("--reference", default="B", help="arm used for paired differences / excess follow rate")
ap.add_argument("--no_ice_off", action="store_true", help="skip the ICE-switched-off control")
ap.add_argument("--precision", default="auto")
ap.add_argument("--out", default="results.json")
a = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype, _ = pick_precision(a.precision, device)
meta = load_meta(a.data_dir)
tok = None
if meta.get("tokenizer"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])


def evaluate(model, seq_len, keep):
    res = eval_loss_splits(model, a.data_dir, seq_len, a.batch_size, a.val_batches, device, amp_dtype,
                           ice_keep=keep)
    res.update(run_synthetic_tasks(model, a.data_dir, device, amp_dtype, tuple(a.context_lens), a.n_task,
                                   ice_keep=keep))
    if tok is not None:
        res.update(run_coderecall(model, tok, open_tokens(a.data_dir, "val_code"), meta["eos_id"], device,
                                  amp_dtype, tuple(a.coderecall_gaps), a.n_coderecall, ice_keep=keep))
    return res


prints = {json.load(open(os.path.join(r, "config.json"))).get("data_meta", {}).get("fingerprint") for r in a.runs}
if not a.runs and not a.from_json:
    ap.error("give --runs and/or --from_json")
if len(prints - {None}) > 1:
    print(f"WARNING: these runs were trained on different data (fingerprints {sorted(prints - {None})}): "
          f"the comparison is not fair")

all_res = {}
for fj in a.from_json:
    all_res.update(json.load(open(fj)))
for run in a.runs:
    cfg = json.load(open(os.path.join(run, "config.json")))
    mcfg = model_config_from_dict(cfg["model"])
    if mcfg.ice and mcfg.ice_v == "vr":
        set_alpha_mask(alpha_mask_for_data(a.data_dir))      # ICE-VR identifier anchors
    model = ICELM(mcfg).to(device)
    path = os.path.join(run, "model_final.pt")
    if not os.path.exists(path):
        path = os.path.join(run, "ckpt_latest.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.prepare_backend(device, amp_dtype or torch.float32)
    seq_len = a.seq_len or cfg["train"]["seq_len"]
    name = cfg["train"]["arm"]
    if mcfg.ice and not mcfg.ice_trust and "notrust" not in name:
        name += "(v1)"                       # trained before the trust features existed
    tokens = ck.get("state", {}).get("tokens", 0)
    print(f"\n### {name} ({run}, {tokens/1e6:,.1f}M tokens)")
    base = dict(arm=name, tokens=tokens, params=model.num_params(), ice_params=model.ice_params(),
                kv_bytes_per_token=kv_bytes_per_token(mcfg), seed=cfg["train"].get("seed"),
                train_gflops_per_token=model.flops_per_token(cfg["train"]["seq_len"]) / 1e9)
    variants = [(name, 1.0)] + ([(f"{name}_off", 0.0)] if uses_ice(model) and not a.no_ice_off else [])
    for label, keep in variants:
        res = dict(base, arm=label, **evaluate(model, seq_len, keep))
        for k, v in res.items():
            if not isinstance(v, list):
                print(f"  {k:48s} {v:.4f}" if isinstance(v, float) else f"  {k:48s} {v}")
        all_res[f"{run}#{label}"] = res
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

def table(title, rows):
    cols = list(all_res.values())
    print(f"\n== {title}\n" + "metric".ljust(40) + "".join(r["arm"].rjust(15) for r in cols))
    for label, key in rows:
        vals = [r.get(key) for r in cols]
        if all(v is None for v in vals):
            continue
        print(label.ljust(40) + "".join(
            (f"{v:15.4f}" if isinstance(v, float) and not math.isnan(v) else f"{'-' if v is None else v:>15}")
            for v in vals))


table("cost", [("params", "params"), ("ICE params", "ice_params"), ("KV bytes/token", "kv_bytes_per_token"),
              ("train GFLOPs/token", "train_gflops_per_token"), ("seed", "seed"), ("tokens", "tokens")])
table("held-out loss (lower is better)", [
    (f"{s}/{g}", f"{s}/{g}_loss") for s in ("code", "prose")
    for g in ("all", "other", "copyable", "copy_far", "lk_right", "lk_wrong", "lk_none")])
ref = next((r for r in all_res.values() if r["arm"] == a.reference), None)
for r in all_res.values():
    for s_ in ("code", "prose"):
        if ref is not None:
            r[f"{s_}/follow_excess"] = r[f"{s_}/follow_wrong_lookup"] - ref[f"{s_}/follow_wrong_lookup"]
table("misleading lookups: P(top-1 = wrong lookup token); excess = minus the reference arm's rate",
      [(f"{s}/{m}", f"{s}/{m}") for s in ("code", "prose") for m in ("follow_wrong_lookup", "follow_excess")])


def paired_diff(x, y, n_boot=2000, seed=0):
    """mean(x - y) and a paired bootstrap 95% interval (x, y: 0/1 per item, same items)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


if tok is not None:
    for r in all_res.values():
        for cat, v in summarize(r, "dec_acc").items():
            r[f"cr/{cat}"] = v
        for cat, v in summarize(r, "dec_lookup_wrong_frac").items():
            r[f"cr_wrongfrac/{cat}"] = v
        for cat, v in summarize(r, "dec_follow_wrong_lookup").items():
            r[f"cr_follow/{cat}"] = v
    table("CodeRecall: accuracy on the deciding digit (mean over gaps)", [(c, f"cr/{c}") for c in CATEGORIES])
    table("CodeRecall: share of items where the lookup is wrong on the deciding digit",
          [(c, f"cr_wrongfrac/{c}") for c in CATEGORIES])
    table("CodeRecall: follow rate on wrong lookups (deciding digit)", [(c, f"cr_follow/{c}") for c in CATEGORIES])
    if ref is not None:
        print(f"\n== CodeRecall: deciding-digit accuracy minus {a.reference} [paired bootstrap 95% CI], items pooled over gaps")
        print("category".ljust(16) + "".join(r["arm"].rjust(26) for r in all_res.values()))
        for cat in CATEGORIES:
            cells = []
            for r in all_res.values():
                xs, ys = [], []
                for k, v in r.items():
                    if k.startswith(f"coderecall/{cat}@") and k.endswith("/items_dec") \
                            and len(ref.get(k, [])) == len(v):
                        xs += v
                        ys += ref[k]
                m, lo, hi = paired_diff(xs, ys)
                cells.append(f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]")
            print(cat.ljust(16) + "".join(c.rjust(26) for c in cells))
table("synthetic / needle tasks", [(k, k) for k in next(iter(all_res.values()))
                                   if k.endswith(("/acc", "/em")) and not k.startswith("coderecall/")])
with open(a.out, "w") as f:
    json.dump(all_res, f, indent=2)
print(f"\nsaved {a.out}")
