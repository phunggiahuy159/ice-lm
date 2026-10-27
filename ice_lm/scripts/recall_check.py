#!/usr/bin/env python
"""Is the hybrid's near-zero synthetic recall a real failure or an artefact of the task?

    python scripts/recall_check.py --runs runs/t200_B runs/t200_G_VR --data_dir data --out recall_check.json

evaluate.py's repeat / MQAR draw tokens uniformly from ~31K vocabulary ids, most of them rare tokens whose embeddings
barely trained: attention has to match them through those embeddings, the exact lookup matches ids. This check
scores the same two tasks (same construction and teacher-forced scoring as ice_lm.evals) with three token pools:
  full      uniform over [V/32, V-100), as in evaluate.py (reproduces its setting)
  frequent  the --top most frequent ids of the held-out data
  natural   (repeat only) a real held-out prose passage, repeated verbatim
and short to long spans. If B does well on frequent / natural tokens, its 0.00 in evaluate.py is an artefact.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ice_lm.config import model_config_from_dict  # noqa: E402
from ice_lm.data import alpha_mask_for_data, load_meta, open_tokens  # noqa: E402
from ice_lm.evals import score_answers  # noqa: E402
from ice_lm.ice_features import set_alpha_mask  # noqa: E402
from ice_lm.model import ICELM  # noqa: E402
from ice_lm.utils import pick_precision  # noqa: E402


def load_run(run, data_dir, device, amp_dtype):
    cfg = json.load(open(os.path.join(run, "config.json")))
    mcfg = model_config_from_dict(cfg["model"])
    if mcfg.ice and mcfg.ice_v == "vr":
        set_alpha_mask(alpha_mask_for_data(data_dir))
    model = ICELM(mcfg).to(device)
    model.load_state_dict(torch.load(os.path.join(run, "model_final.pt"), map_location=device,
                                     weights_only=False)["model"])
    model.prepare_backend(device, amp_dtype or torch.float32)
    return model, cfg["train"]["arm"]


def repeat_item(rng, draw, length, sep, prefix_len=16):
    """Like evals.task_repeat with tokens from draw(rng, n)."""
    s, pre = draw(rng, length), draw(rng, prefix_len)
    seq = pre + s + sep + s
    return seq, (len(pre) + len(s) + len(sep) + 1, len(seq))     # skip the first copied token


def natural_item(rng, text, eos, length, sep, prefix_len=16):
    """A real passage (no document boundary inside) repeated verbatim after a separator."""
    for _ in range(100):
        st = int(rng.integers(0, len(text) - length - prefix_len - 1))
        w = np.asarray(text[st: st + prefix_len + length]).tolist()
        if eos not in w:
            pre, s = w[:prefix_len], w[prefix_len:]
            seq = pre + s + sep + s
            return seq, (len(pre) + len(s) + len(sep) + 1, len(seq))
    raise RuntimeError("no passage without a document boundary")


def mqar_item(rng, pool, n_pairs, n_queries):
    """Like evals.task_mqar with keys / values from pool (keys distinct)."""
    keys = rng.choice(pool, size=n_pairs, replace=False).tolist()
    vals = rng.choice(pool, size=n_pairs).tolist()
    seq = []
    for k_, v_ in zip(keys, vals):
        seq += [k_, v_]
    spans = []
    for j in rng.choice(n_pairs, size=min(n_queries, n_pairs), replace=False):
        seq += [keys[j], vals[j]]
        spans.append((len(seq) - 1, len(seq)))
    return seq, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--n", type=int, default=50, help="items per (task, pool, length)")
    ap.add_argument("--top", type=int, default=5000, help="size of the frequent-token pool")
    ap.add_argument("--repeat_lens", type=int, nargs="+", default=[32, 128, 512, 2000])
    ap.add_argument("--mqar_pairs", type=int, nargs="+", default=[8, 32, 128, 496])
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--out", default="recall_check.json")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, _ = pick_precision(a.precision, device)
    meta = load_meta(a.data_dir)
    eos = meta["eos_id"]
    code, prose = open_tokens(a.data_dir, "val_code"), open_tokens(a.data_dir, "val_prose")
    V = meta["vocab_size"]
    lo, hi = max(10, V // 32), max(20, V - 100)                  # evals.run_synthetic_tasks' range
    counts = np.bincount(np.concatenate([np.asarray(code), np.asarray(prose)]).astype(np.int64), minlength=V)
    counts[[0, 1, 2, eos]] = 0                                   # unk / bos / eos never appear in the pools
    pools = dict(full=np.arange(lo, hi), frequent=np.argsort(-counts)[: a.top])
    sep = [lo + 1, lo + 2, lo + 3]                               # as in evals.run_synthetic_tasks

    def drawer(pool):
        return lambda rng, n: rng.choice(pool, size=n).tolist()

    # identical items for every run: one generator per (task, pool, length)
    items = {}
    for L in a.repeat_lens:
        for name, pool in pools.items():
            rng = np.random.default_rng([1, L, len(name)])
            items[f"repeat/{name}@{L}"] = [repeat_item(rng, drawer(pool), L, sep) for _ in range(a.n)]
        rng = np.random.default_rng([2, L])
        items[f"repeat/natural@{L}"] = [natural_item(rng, prose, eos, L, sep) for _ in range(a.n)]
    for P in a.mqar_pairs:
        for name, pool in pools.items():
            rng = np.random.default_rng([3, P, len(name)])
            items[f"mqar/{name}@{P}"] = [mqar_item(rng, pool, P, 16) for _ in range(a.n)]

    res = {}
    for run in a.runs:
        model, arm = load_run(run, a.data_dir, device, amp_dtype)
        r = {}
        for key, its in items.items():
            acc, em, _ = score_answers(model, [s for s, _ in its], [sp for _, sp in its], device, amp_dtype)
            r[f"{key}/acc"], r[f"{key}/em"] = acc, em
            print(f"[recall] {arm:12s} {key:24s} acc {acc:.3f}  em {em:.3f}", flush=True)
        res[arm] = r
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    arms = list(res)
    print("\ntoken accuracy (em = whole span right; mqar: one token per query)")
    print("task".ljust(26) + "".join(x.rjust(14) for x in arms))
    for key in items:
        print(key.ljust(26) + "".join(f"{res[x][key + '/acc']:14.3f}" for x in arms))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(results=res, n=a.n, top=a.top), f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
