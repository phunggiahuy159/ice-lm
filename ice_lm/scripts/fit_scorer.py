#!/usr/bin/env python
"""Fit only the ICE-V / VR scorer (ICEPointer) on a FROZEN trained model.

    python scripts/fit_scorer.py --base_run /vol/runs/t200_G --cand vr --data_dir /vol/data \
        --out_dir /vol/runs/fit_G_vr --tokens 2e7

The backbone (every weight of the base run) is frozen; a fresh ICEPointer with the candidate set --cand is
trained on the training stream (different windows than the base run saw first: --seed). Because the backbone
never sees ICE and is not updated, "ICE off" is exactly the base model. The result is saved as an ordinary run
folder (config.json + model_final.pt) so scripts/evaluate.py and scripts/benchmark.py work unchanged.

--init_from <run> starts the scorer from another run's ICEPointer (e.g. the jointly trained G_VR's), to separate
"what the scorer learns" from "what the backbone co-adapts".

ICE-V2 (research/round2/decision2.md section 2):
    --v2 [--v2_feats dnrq]   the V2 scorer (slot-feature tables R/A/N/Q, softplus gamma, no beta). With --init_from
                             an old (V) scorer, its b / L / D / w / norm / c0 are copied, gamma~ = softplus^-1(max(
                             gamma, 1e-3)), beta is dropped and the V2 tables start at 0
    --listwise 0.25          weight of the listwise slot CE
    --syn_frac 0.02          that fraction of the fit windows is replaced by synthetic conflict / override items
                             (ice_lm/syn_conflicts.py FIT families; disjoint from CodeRecall): "task-supervised"
"""
import argparse
import dataclasses
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ice_lm.config import dump_json, model_config_from_dict, to_dict  # noqa: E402
from ice_lm.data import alpha_mask_for_data, collate, load_meta, split_batch  # noqa: E402
from ice_lm.ice_features import set_alpha_mask  # noqa: E402
from ice_lm.model import ICELM  # noqa: E402
from ice_lm.syn_conflicts import FIT_FAMILIES, SynMixWindows, item_pool  # noqa: E402
from ice_lm.utils import pick_precision  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base_run", required=True)
ap.add_argument("--cand", default="vr", choices=["v", "vr"])
ap.add_argument("--verify", default="true")
ap.add_argument("--init_from", default=None, help="run folder whose ICEPointer initialises the scorer")
ap.add_argument("--v2", action="store_true", help="fit the ICE-V2 scorer")
ap.add_argument("--v2_feats", default="dnrq", help="V2 tables: d delta, n count, r recency, q backbone rank")
ap.add_argument("--listwise", type=float, default=0.0, help="weight of the listwise slot CE")
ap.add_argument("--syn_frac", type=float, default=0.0, help="fraction of fit windows replaced by synthetic items")
ap.add_argument("--syn_pool", type=int, default=4000, help="synthetic items tokenized up front")
ap.add_argument("--data_dir", default="data")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--tokens", type=float, default=2e7)
ap.add_argument("--micro_batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-2)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--num_workers", type=int, default=6)
a = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype, _ = pick_precision("auto", device)
saved = json.load(open(os.path.join(a.base_run, "config.json")))
base_cfg = model_config_from_dict(saved["model"])
cfg = dataclasses.replace(base_cfg, ice=True, ice_v=a.cand, ice_v_input=False,
                          ice_v_verify=a.verify.lower() in ("1", "true", "yes"), grad_checkpointing=False,
                          ice_v2=a.v2, ice_v2_feats=a.v2_feats if a.v2 else "dnrq", ice_v_listwise=a.listwise)
model = ICELM(cfg).to(device)
model.prepare_backend(device, amp_dtype or torch.float32)

sd = torch.load(os.path.join(a.base_run, "model_final.pt"), map_location="cpu")["model"]
sd = {k: v for k, v in sd.items() if not k.startswith(("ice_ptr.", "ice_in.", "ice_out."))}
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected, unexpected
assert all(k.startswith("ice_ptr.") for k in missing), [k for k in missing if not k.startswith("ice_ptr.")]
if a.init_from:
    sd2 = torch.load(os.path.join(a.init_from, "model_final.pt"), map_location="cpu")["model"]
    ptr = {k[len("ice_ptr."):]: v for k, v in sd2.items() if k.startswith("ice_ptr.")}
    if a.v2 and "gamma" in ptr:     # old scorer -> V2: gamma >= 0 through softplus, beta dropped
        g = ptr.pop("gamma").float().clamp(min=1e-3)
        ptr["gamma_t"] = torch.log(torch.expm1(g))
        print(f"[fit] init_from: dropped beta={ptr.pop('beta').item():.4f}; gamma clamped >= 1e-3 "
              f"({int((g <= 1e-3).sum())} slots)", flush=True)
    missing, unexpected = model.ice_ptr.load_state_dict(ptr, strict=False)
    assert not unexpected, unexpected
    assert all(k[:1] in "RANQ" and k.endswith(".weight") or k == "gamma_t" for k in missing), missing
    print(f"[fit] init_from {a.init_from}: fresh {missing}", flush=True)
for n, p in model.named_parameters():
    p.requires_grad_(n.startswith("ice_ptr."))
params = [p for p in model.parameters() if p.requires_grad]
print(f"[fit] base={a.base_run} cand={a.cand} v2={a.v2}({cfg.ice_v2_feats}) listwise={a.listwise} "
      f"verify={cfg.ice_v_verify} trainable={sum(p.numel() for p in params)}"
      f" frozen={sum(p.numel() for p in model.parameters() if not p.requires_grad)}", flush=True)

if a.cand == "vr":
    set_alpha_mask(alpha_mask_for_data(a.data_dir))
seq_len = saved["train"]["seq_len"]
pool = None
if a.syn_frac > 0:
    from transformers import AutoTokenizer
    pool = item_pool(AutoTokenizer.from_pretrained(load_meta(a.data_dir)["tokenizer"]), a.syn_pool, seed=a.seed)
    print(f"[fit] synthetic items: {len(pool)} ({', '.join(FIT_FAMILIES)}), syn_frac={a.syn_frac}", flush=True)
ds = SynMixWindows(a.data_dir, seq_len, cfg.ice_orders, True, 0.0, a.seed, 0, 1, start_index=0,
                   micro_batch=a.micro_batch, cand=a.cand, v2=a.v2, pool=pool, syn_frac=a.syn_frac)
loader = DataLoader(ds, batch_size=a.micro_batch, collate_fn=collate, num_workers=a.num_workers,
                    pin_memory=device.type == "cuda", persistent_workers=a.num_workers > 0, prefetch_factor=4)
steps = math.ceil(a.tokens / (a.micro_batch * seq_len))
opt = torch.optim.Adam(params, lr=a.lr)
model.train()
it = iter(loader)
t0, log = time.time(), []
for step in range(1, steps + 1):
    lr = a.lr * min(1.0, step / 20) * 0.5 * (1 + math.cos(math.pi * step / steps))
    for g in opt.param_groups:
        g["lr"] = lr
    x, y, ice, keep, _ = split_batch(next(it), device)
    with torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else torch.enable_grad():
        _, loss = model(x, y, ice=ice, ice_keep=keep)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 25 == 0 or step == 1:
        tps = step * a.micro_batch * seq_len / (time.time() - t0)
        print(f"[fit] step {step}/{steps} loss {loss.item():.4f} lr {lr:.2e} tok/s {tps:,.0f}", flush=True)
        log.append(dict(step=step, loss=loss.item(), lr=lr))

os.makedirs(a.out_dir, exist_ok=True)
tag = a.cand + ("2" if a.v2 else "") + ("+syn" if a.syn_frac > 0 else "")      # e.g. fit_vr2+syn: task-supervised
train_cfg = dict(saved["train"], fit=vars(a), arm=f'{saved["train"]["arm"]}+fit_{tag}')
dump_json(os.path.join(a.out_dir, "config.json"),
          dict(model=to_dict(cfg), train=train_cfg, data_meta=saved.get("data_meta", {}), world=1,
               fitted_on=a.base_run))
state = dict(step=saved.get("state", {}).get("step", 0), tokens=int(float(saved["train"]["total_tokens"])))
torch.save(dict(model=model.state_dict(), model_config=to_dict(cfg), state=state),
           os.path.join(a.out_dir, "model_final.pt"))
with open(os.path.join(a.out_dir, "log.jsonl"), "w") as fh:
    for r in log:
        fh.write(json.dumps(r) + "\n")
print(f"[fit] saved {a.out_dir} in {time.time() - t0:.0f}s", flush=True)
