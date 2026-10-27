#!/usr/bin/env python
"""Stale-version edit test: an old version of a file in context, then the current one. Does the model copy the old
values where the file changed?

    python scripts/stale_edit.py --runs runs/t200_B runs/t200_G_VR --data_dir data --out stale_edit.json

Items (fixed by the data and seed, identical for every arm): held-out code documents of --min_tokens..--max_tokens
tokens. v1 = the document with --frac of its distinct identifiers renamed (a prefix such as "old_", so v1 and v2 differ
from the first token of the name) and the same share of its numeric literals changed (a different first digit); v2 = the original document. The model reads v1, a blank line, then v2
(teacher-forced), and v2's tokens are scored in two groups:
  changed    tokens overlapping a renamed identifier or changed literal in v2 (the lookup proposes the stale v1 text)
  unchanged  every other v2 token (a verbatim copy of v1 is right)
plus a control with v2 alone (no v1 in context). Reports loss and top-1 accuracy per group, and for changed tokens
the rate of predicting exactly the stale v1 token ("follow stale").
"""
import argparse
import json
import keyword
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ice_lm.config import model_config_from_dict  # noqa: E402
from ice_lm.data import alpha_mask_for_data, load_meta, model_ice_keys, open_tokens  # noqa: E402
from ice_lm.ice_features import compute_ice_features, set_alpha_mask  # noqa: E402
from ice_lm.model import ICELM  # noqa: E402
from ice_lm.utils import pick_precision  # noqa: E402

IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
NUMBER = re.compile(r"\b\d+\b")
SKIP = set(keyword.kwlist) | {"self", "cls", "True", "False", "None", "print", "range", "len", "int", "str", "list",
                              "dict", "return", "import", "from", "def", "class", "function", "var", "let", "const",
                              "public", "private", "static", "void", "new", "this", "null", "true", "false"}
PREFIXES = ("old_", "prev_", "tmp_", "orig_", "my_")   # v1 differs from v2 at the first token of the name


def make_items(tok, data, eos, n, frac, min_tokens, max_tokens, seed):
    """-> list of dict(v1, v2 texts, changed char spans in v2, map of renamed / changed strings)."""
    x = np.asarray(data)
    ends = np.nonzero(x == eos)[0]
    starts = np.concatenate([[0], ends + 1])[:-1]
    rng = np.random.default_rng(seed)
    items = []
    for d in rng.permutation(len(starts)):
        s, e = int(starts[d]), int(ends[d])
        if not min_tokens <= e - s <= max_tokens:
            continue
        v2 = tok.decode(x[s:e].tolist())
        names = sorted({m for m in IDENT.findall(v2) if m not in SKIP})
        nums = sorted(set(NUMBER.findall(v2)))
        k_n, k_l = max(1, round(frac * len(names))), round(frac * len(nums))
        if len(names) < 5:
            continue
        r = np.random.default_rng([seed, int(d)])
        ren = {m: PREFIXES[int(r.integers(len(PREFIXES)))] + m for m in r.choice(names, size=k_n, replace=False)}
        lit = {}
        for m in (r.choice(nums, size=k_l, replace=False) if k_l else []):
            first = str((int(m[0]) + int(r.integers(1, 9))) % 10)
            lit[m] = (first if first != "0" or len(m) == 1 else "9") + m[1:]
        spans = [(mt.start(), mt.end()) for mt in IDENT.finditer(v2) if mt.group() in ren]
        spans += [(mt.start(), mt.end()) for mt in NUMBER.finditer(v2) if mt.group() in lit]
        v1 = IDENT.sub(lambda mt: ren.get(mt.group(), mt.group()), v2)
        v1 = NUMBER.sub(lambda mt: lit.get(mt.group(), mt.group()), v1)
        items.append(dict(doc=int(d), v1=v1, v2=v2, spans=sorted(spans), renamed=ren, literals=lit))
        if len(items) >= n:
            break
    return items


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
    return model, cfg["train"]["arm"]


@torch.no_grad()
def token_losses(model, ids, device, amp_dtype):
    """Teacher-forced per-token loss and argmax for ids[1:] given ids[:-1]."""
    x = np.asarray(ids[:-1], dtype=np.int64)
    ice = None
    if model.uses_ice():
        f = compute_ice_features(x, None, model.cfg.ice_orders, model.cfg.ice_v, model.cfg.ice_v2)
        ice = {k: torch.from_numpy(f[k]).to(device)[None] for k in model_ice_keys(model.cfg.ice_v, model.cfg.ice_v2)}
    xt, yt = torch.from_numpy(x).to(device)[None], torch.tensor(ids[1:], device=device)[None]
    with torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else torch.no_grad():
        logits, loss = model(xt, yt, ice=ice, reduction="none")
    return loss[0].float().cpu().numpy(), logits[0].argmax(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--min_tokens", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=1900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--out", default="stale_edit.json")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, _ = pick_precision(a.precision, device)
    meta = load_meta(a.data_dir)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
    items = make_items(tok, open_tokens(a.data_dir, "val_code"), meta["eos_id"], a.n, a.frac, a.min_tokens,
                       a.max_tokens, a.seed)
    # token ids and the changed / unchanged masks of the v2 part (identical for every arm)
    prepared = []
    for it in items:
        e1 = tok(it["v1"], add_special_tokens=False)["input_ids"]
        enc = tok(it["v2"], add_special_tokens=False, return_offsets_mapping=True)
        e2, offs = enc["input_ids"], enc["offset_mapping"]
        sep = tok("\n\n", add_special_tokens=False)["input_ids"]
        changed = np.array([any(s < b and a_ < e for s, e in it["spans"]) for a_, b in offs], dtype=bool)
        # the token v1 has at the aligned position: v1 and v2 tokenize identically outside the edits only, so the
        # stale token is the token of v1 at the same character offset of the renamed text (first token of a span)
        enc1 = tok(it["v1"], add_special_tokens=False, return_offsets_mapping=True)
        prepared.append(dict(full=e1 + sep + e2, alone=e2, n1=len(e1) + len(sep), changed=changed, e2=e2,
                             first=np.array([c and (i == 0 or not changed[i - 1]) for i, c in enumerate(changed)]),
                             v1_ids=enc1["input_ids"], v1_offs=enc1["offset_mapping"], offs=offs, spans=it["spans"],
                             v1=it["v1"], v2=it["v2"], ren=it["renamed"], lit=it["literals"]))
    print(f"[stale] {len(prepared)} files, {sum(p['changed'].sum() for p in prepared)} changed and "
          f"{sum((~p['changed']).sum() for p in prepared)} unchanged v2 tokens", flush=True)

    res = {}
    for run in a.runs:
        model, arm = load_run(run, a.data_dir, device, amp_dtype)
        acc = {k: [0.0, 0, 0] for k in ("unchanged", "changed", "changed_first", "alone_unchanged", "alone_changed")}
        stale_follow = [0, 0]
        for p in prepared:
            loss, pred = token_losses(model, p["full"], device, amp_dtype)
            sl = slice(p["n1"] - 1, p["n1"] - 1 + len(p["e2"]))       # predictions of the v2 tokens
            lv, pv, y = loss[sl], pred[sl], np.asarray(p["e2"])
            for key, m in (("unchanged", ~p["changed"]), ("changed", p["changed"]), ("changed_first", p["first"])):
                acc[key][0] += float(lv[m].sum()); acc[key][1] += int(m.sum()); acc[key][2] += int((pv[m] == y[m]).sum())
            # follow stale: at the first token of each changed span, the model predicts the v1 token that starts the
            # stale version of that span (the renamed identifier / changed number)
            for i in np.nonzero(p["first"])[0]:
                c0 = p["offs"][i][0]
                stale_text_start = len(_v1_prefix(p, c0))
                j = next((k for k, (s, e) in enumerate(p["v1_offs"]) if s <= stale_text_start < e), None)
                if j is not None and p["v1_ids"][j] != y[i]:
                    stale_follow[0] += int(pv[i] == p["v1_ids"][j]); stale_follow[1] += 1
            la, pa = token_losses(model, p["alone"], device, amp_dtype)
            ya = np.asarray(p["alone"][1:])
            for key, m in (("alone_unchanged", ~p["changed"][1:]), ("alone_changed", p["changed"][1:])):
                acc[key][0] += float(la[m].sum()); acc[key][1] += int(m.sum()); acc[key][2] += int((pa[m] == ya[m]).sum())
        r = {f"{k}/loss": v[0] / max(1, v[1]) for k, v in acc.items()}
        r.update({f"{k}/acc": v[2] / max(1, v[1]) for k, v in acc.items()})
        r.update({f"{k}/n": v[1] for k, v in acc.items()})
        r["follow_stale"] = stale_follow[0] / max(1, stale_follow[1])
        r["follow_stale_n"] = stale_follow[1]
        res[arm] = r
        print(f"[stale] {arm}: " + "  ".join(f"{k} {r[k + '/loss']:.3f}/{r[k + '/acc']:.3f}" for k in acc)
              + f"  follow_stale {r['follow_stale']:.3f}", flush=True)
        del model

    arms = list(res)
    print("\nloss / top-1 on v2 tokens (v1 in context unless 'alone')")
    print("group".ljust(22) + "".join(x.rjust(18) for x in arms))
    for k in ("unchanged", "changed", "changed_first", "alone_unchanged", "alone_changed"):
        print(k.ljust(22) + "".join(f"{res[x][k + '/loss']:9.3f}/{res[x][k + '/acc']:.3f}".rjust(18) for x in arms))
    print("follow_stale".ljust(22) + "".join(f"{res[x]['follow_stale']:18.3f}" for x in arms))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(results=res, n_files=len(prepared), frac=a.frac, seed=a.seed), open(a.out, "w"), indent=1)
    print("saved", a.out)


def _v1_prefix(p, c0):
    """v1 text up to the v2 character offset c0 (apply the same edits to v2[:c0])."""
    pre = IDENT.sub(lambda mt: p["ren"].get(mt.group(), mt.group()), p["v2"][:c0])
    return NUMBER.sub(lambda mt: p["lit"].get(mt.group(), mt.group()), pre)


if __name__ == "__main__":
    main()
