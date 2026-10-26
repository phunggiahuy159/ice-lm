#!/usr/bin/env python
"""Upload finished runs' log.jsonl to Weights & Biases (for runs trained without WANDB_API_KEY).

    WANDB_API_KEY=... python scripts/wandb_backfill.py runs/t200_B runs/t200_G_C ...

One W&B run per folder, named after the folder, with the run's config.json as config. A folder that already has
a wandb_id.txt (it logged live) is skipped.
"""
import argparse
import json
import os

import wandb

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "ice-lm"))
a = ap.parse_args()

for run_dir in a.runs:
    name = os.path.basename(os.path.normpath(run_dir))
    log_path = os.path.join(run_dir, "log.jsonl")
    if os.path.exists(os.path.join(run_dir, "wandb_id.txt")):
        print(f"[backfill] {name}: already logged live, skipped")
        continue
    if not os.path.exists(log_path):
        print(f"[backfill] {name}: no log.jsonl, skipped")
        continue
    cfg_path = os.path.join(run_dir, "config.json")
    config = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    run = wandb.init(project=a.project, name=name, config=config, notes="uploaded from log.jsonl after training",
                     reinit=True)
    n = 0
    for line in open(log_path):
        rec = json.loads(line)
        step = rec.pop("step", None)
        rec.pop("time", None)
        run.log(rec, step=step)
        n += 1
    run.finish()
    with open(os.path.join(run_dir, "wandb_id.txt"), "w") as f:
        f.write(run.id)
    print(f"[backfill] {name}: {n} records -> {run.url}")
