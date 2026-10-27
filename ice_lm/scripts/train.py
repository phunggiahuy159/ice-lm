#!/usr/bin/env python
"""Train one arm.  Single GPU:  python scripts/train.py --preset 0.6b --arm C --data_dir data --out_dir runs/C
Multi-GPU (e.g. Kaggle 2xT4):  torchrun --nproc_per_node=2 scripts/train.py ...
Re-running the same command resumes from out_dir/ckpt_latest.pt (if present)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation on 16 GB GPUs

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ice_lm.config import TrainConfig, ModelConfig, build_configs, dump_json, model_config_from_dict, to_dict
from ice_lm.data import TrainWindows, alpha_mask_for_data, collate, load_meta, split_batch
from ice_lm.evals import eval_loss_splits
from ice_lm.ice_features import set_alpha_mask
from ice_lm.model import ICELM, kv_bytes_per_token, layer_plan
import shutil  # noqa: E402

from ice_lm.utils import (JsonlLogger, build_optimizer, cleanup_distributed, init_wandb, is_main, load_checkpoint,
                          load_optimizer_shard, lr_at, lr_wsd, pick_precision, wsd_decay_steps, save_checkpoint, save_optimizer_shard,
                          setup_distributed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="140m")
    ap.add_argument("--arm", default="A")
    tfields = {f.name: f for f in dataclasses.fields(TrainConfig)}
    mfields = {f.name: f for f in dataclasses.fields(ModelConfig)}
    for name, f in {**mfields, **tfields}.items():
        if name in ("preset", "arm", "ice_orders"):
            continue
        typ = f.type if isinstance(f.type, type) else eval(f.type)  # noqa: S307 - dataclass annotations
        if typ is bool:
            ap.add_argument(f"--{name}", type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=None)
        else:
            conv = float if typ is float else ((lambda v: int(float(v))) if typ is int else str)
            ap.add_argument(f"--{name}", type=conv, default=None)
    return ap.parse_args()


# settings that define the learning-rate schedule and the data: a resumed run keeps its saved values.
# (micro_batch / num_workers / GPU count may change: the samples of optimizer step s are always the indices
#  [s*S, (s+1)*S) with S = global batch / seq_len, see ice_lm/data.py TrainWindows)
RESUME_KEYS = ("total_tokens", "global_batch_tokens", "lr", "min_lr_ratio", "warmup_tokens", "seq_len", "seed",
               "ice_dropout", "weight_decay", "beta1", "beta2", "grad_clip", "schedule", "decay_frac")


def main():
    args = vars(parse_args())
    preset, arm = args.pop("preset"), args.pop("arm")
    rank, world, local, device = setup_distributed()
    meta = load_meta(args.get("data_dir") or "data")
    args["vocab_size"] = args.get("vocab_size") or meta["vocab_size"]
    mcfg, tcfg = build_configs(preset, arm, **args)
    torch.manual_seed(tcfg.seed)
    main_proc = is_main(rank)
    os.makedirs(tcfg.out_dir, exist_ok=True)
    if tcfg.resume and os.path.exists(os.path.join(tcfg.out_dir, "model_final.pt")):
        if main_proc:
            print(f"[train] {tcfg.out_dir} is already finished (model_final.pt); nothing to do "
                  f"(use a new out_dir, or --resume false to retrain from scratch)")
        cleanup_distributed()
        return
    saved_cfg = os.path.join(tcfg.out_dir, "config.json")
    if tcfg.branch_from and not os.path.exists(os.path.join(tcfg.out_dir, "ckpt_latest.pt")):
        # a WSD decay branch: continue the longer run's stable phase from its saved state, with this run's budget
        assert tcfg.schedule == "wsd" and world == 1, "branch_from: wsd schedule on one process"
        shutil.copy(tcfg.branch_from, os.path.join(tcfg.out_dir, "ckpt_latest.pt"))
        shutil.copy(os.path.join(os.path.dirname(tcfg.branch_from), "config.json"), saved_cfg)
    data_meta = meta
    if tcfg.resume and os.path.exists(os.path.join(tcfg.out_dir, "ckpt_latest.pt")) and os.path.exists(saved_cfg):
        # continuing a run: keep the architecture and the schedule / data order it was started with
        with open(saved_cfg) as f:
            saved = json.load(f)
        old = model_config_from_dict(saved["model"])
        if tcfg.keep_resume_config:
            fixed = {k: v for k, v in saved.get("train", {}).items() if k in RESUME_KEYS and getattr(tcfg, k) != v
                     and not (tcfg.branch_from and k == "total_tokens")}       # a branch has its own budget
            if fixed:
                if main_proc:
                    print(f"[train] resuming with the run's saved settings (ignoring the new values): "
                          f"{ {k: (v, getattr(tcfg, k)) for k, v in fixed.items()} }  [saved, given]")
                tcfg = dataclasses.replace(tcfg, **fixed)
        old_fp = saved.get("data_meta", {}).get("fingerprint")
        if old_fp and meta.get("fingerprint") and old_fp != meta["fingerprint"] and main_proc:
            print(f"[train] WARNING: the data differs from the data this run started with (fingerprint "
                  f"{meta['fingerprint']} vs {old_fp}), e.g. a re-download in a new session: the comparison with "
                  f"other arms is no longer exact")
        data_meta = saved.get("data_meta", meta)         # keep the original record
        if old != mcfg:
            if main_proc:
                diff = {k: (getattr(old, k), getattr(mcfg, k)) for k in to_dict(mcfg) if getattr(old, k) != getattr(mcfg, k)}
                print(f"[train] resuming with the run's saved model config (differs from the current defaults: {diff})")
            mcfg = dataclasses.replace(old, grad_checkpointing=mcfg.grad_checkpointing, gdn_backend=mcfg.gdn_backend)

    amp_dtype, use_scaler = pick_precision(tcfg.precision, device)
    model = ICELM(mcfg).to(device)
    model.prepare_backend(device, amp_dtype or torch.float32)
    if main_proc:
        plan = layer_plan(mcfg)
        print(f"[train] preset={preset} arm={arm} device={device} world={world} amp={amp_dtype} scaler={use_scaler}")
        print(f"[train] params total={model.num_params()/1e6:.1f}M  non-embedding={model.num_params(True)/1e6:.1f}M"
              f"  ICE={model.ice_params()/1e6:.2f}M")
        print(f"[train] layout: " + " ".join("A" if p['gdn'] == 0 else ("G" if p['attn'] == 0 else "h") for p in plan)
              + f"   (A=attention, G=GDN, h=mixed heads)   KV bytes/token={kv_bytes_per_token(mcfg)}")
        dump_json(os.path.join(tcfg.out_dir, "config.json"),
                  dict(model=to_dict(mcfg), train=to_dict(tcfg), data_meta=data_meta, world=world))

    tokens_per_micro = tcfg.micro_batch * tcfg.seq_len * world
    grad_accum = max(1, round(tcfg.global_batch_tokens / tokens_per_micro))
    if main_proc and tokens_per_micro * grad_accum != tcfg.global_batch_tokens:
        print(f"[train] WARNING: global batch {tcfg.global_batch_tokens:,} tokens is not a multiple of "
              f"micro_batch*seq_len*GPUs = {tokens_per_micro:,}; using {tokens_per_micro * grad_accum:,} "
              f"(runs with different settings will not see the same data per step)")
    tokens_per_step = tokens_per_micro * grad_accum
    total_steps = max(1, math.ceil(tcfg.total_tokens / tokens_per_step))
    warmup_steps = max(1, round(tcfg.warmup_tokens / tokens_per_step))
    zero = world > 1 and tcfg.zero_optimizer
    opt = build_optimizer(model, tcfg.lr, tcfg.weight_decay, (tcfg.beta1, tcfg.beta2), device, zero=zero)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    state = dict(step=0, tokens=0, samples=0, best_val=None)
    ckpt_path = os.path.join(tcfg.out_dir, "ckpt_latest.pt")
    if tcfg.resume and os.path.exists(ckpt_path):
        state = load_checkpoint(ckpt_path, model, None if zero else opt, scaler, map_location=device)
        opt_ok = True
        if zero:
            opt_ok = load_optimizer_shard(ckpt_path, opt, rank, world, state["step"], map_location=device)
            flag = torch.tensor([float(opt_ok)], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if not bool(flag[0] > 0):     # some rank is missing its shard: fresh optimizer on all ranks
                opt = build_optimizer(model, tcfg.lr, tcfg.weight_decay, (tcfg.beta1, tcfg.beta2), device, zero=zero)
            opt_ok = bool(flag[0] > 0)
        if not zero:
            opt_ok = len(opt.state) > 0          # False e.g. for a ZeRO checkpoint (sharded optimizer) on 1 GPU
        if main_proc:
            print(f"[train] resumed from {ckpt_path} at step {state['step']} ({state['tokens']/1e6:.1f}M tokens)"
                  + ("" if opt_ok else "  [optimizer state not found for this GPU count: Adam moments reset]"))
    if state["step"] >= total_steps:
        if main_proc:
            print("[train] already finished")
        cleanup_distributed()
        return

    if tcfg.compile:
        model = torch.compile(model)
    # gradient_as_bucket_view: no second copy of the gradients (matters on 16 GB GPUs)
    ddp_model = (DDP(model, device_ids=[local] if device.type == "cuda" else None, gradient_as_bucket_view=True)
                 if world > 1 else model)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    with_ice = mcfg.ice
    if with_ice and mcfg.ice_v == "vr":
        set_alpha_mask(alpha_mask_for_data(tcfg.data_dir))          # identifier anchors (training + eval)
    ds = TrainWindows(tcfg.data_dir, tcfg.seq_len, mcfg.ice_orders, with_ice, tcfg.ice_dropout, tcfg.seed,
                      rank, world, start_index=state.get("samples", state["step"] * tokens_per_step // tcfg.seq_len),
                      micro_batch=tcfg.micro_batch, cand=mcfg.ice_v, v2=mcfg.ice_v2)
    loader = DataLoader(ds, batch_size=tcfg.micro_batch, collate_fn=collate, num_workers=tcfg.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=tcfg.num_workers > 0,
                        prefetch_factor=4 if tcfg.num_workers > 0 else None)
    it = iter(loader)
    wb = (init_wandb(tcfg.out_dir, os.path.basename(os.path.normpath(tcfg.out_dir)),
                     dict(model=to_dict(mcfg), train=to_dict(tcfg), data_meta=data_meta, world=world,
                          params=model.num_params(), ice_params=model.ice_params()))
          if main_proc else None)
    logger = JsonlLogger(os.path.join(tcfg.out_dir, "log.jsonl") if main_proc else None, wb)
    autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else nullcontext

    if main_proc:
        print(f"[train] steps={total_steps} (now {state['step']})  grad_accum={grad_accum}  "
              f"tokens/step={tokens_per_step:,}  warmup_steps={warmup_steps}")
    branch_steps = {}
    if tcfg.branch_tokens:
        assert tcfg.schedule == "wsd" and not zero, "branch_tokens: wsd schedule without a sharded optimizer"
        for b in (float(v) for v in tcfg.branch_tokens.split(",")):
            tb = max(1, math.ceil(b / tokens_per_step))
            branch_steps[tb - wsd_decay_steps(tb, tcfg.decay_frac)] = b
    t_start = time.time()
    t_ckpt = time.time()
    t_log, tok_log = time.time(), 0
    ddp_model.train()
    stop_reason = None
    while state["step"] < total_steps:
        step = state["step"]
        lr = (lr_wsd(step, total_steps, warmup_steps, tcfg.lr, tcfg.min_lr_ratio, tcfg.decay_frac)
              if tcfg.schedule == "wsd" else lr_at(step, total_steps, warmup_steps, tcfg.lr, tcfg.min_lr_ratio))
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        loss_acc = torch.zeros((), device=device)      # accumulated on the GPU: no sync per micro-step
        for micro in range(grad_accum):
            batch = next(it)
            x, y, ice, keep, _ = split_batch(batch, device)
            sync = nullcontext() if (world == 1 or micro == grad_accum - 1) else ddp_model.no_sync()
            with sync:
                with autocast():
                    _, loss = ddp_model(x, y, ice=ice, ice_keep=keep)
                loss = loss / grad_accum
                (scaler.scale(loss) if scaler else loss).backward()
            loss_acc += loss.detach().float()
        if scaler:
            scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), tcfg.grad_clip)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        state["step"] += 1
        state["tokens"] += tokens_per_step
        state["samples"] += tcfg.micro_batch * grad_accum * world
        if state["step"] in branch_steps and main_proc:         # wsd: where a shorter budget's decay starts
            save_checkpoint(os.path.join(tcfg.out_dir, f"ckpt_branch_{branch_steps[state['step']]:.0e}.pt"),
                            raw_model, opt, scaler, dict(state))
            print(f"[train] saved the decay-branch point of the {branch_steps[state['step']]:.0e}-token run "
                  f"at step {state['step']}", flush=True)
        tok_log += tokens_per_step

        if state["step"] % tcfg.log_every == 0 or state["step"] == 1:      # same decision on every rank
            if world > 1:
                dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)  # SUM works on NCCL and gloo
            loss_val, gnorm = loss_acc.item() / world, float(gnorm)
            if main_proc:
                dt = time.time() - t_log
                tps = tok_log / max(dt, 1e-9)
                eta_h = (total_steps - state["step"]) * tokens_per_step / max(tps, 1e-9) / 3600
                extra = f"  scale {scaler.get_scale():.0f}" if scaler else ""
                print(f"step {state['step']:>6}/{total_steps}  loss {loss_val:.4f}  lr {lr:.2e}  gnorm {gnorm:.2f}  "
                      f"tok/s {tps:,.0f}  tokens {state['tokens']/1e6:,.1f}M  eta {eta_h:.1f}h{extra}", flush=True)
                logger.log(step=state["step"], loss=loss_val, lr=lr, gnorm=gnorm, tok_per_s=tps, tokens=state["tokens"],
                           **({"loss_scale": scaler.get_scale()} if scaler else {}))
                t_log, tok_log = time.time(), 0

        if tcfg.eval_every and state["step"] % tcfg.eval_every == 0 and main_proc:
            res = eval_loss_splits(raw_model, tcfg.data_dir, tcfg.seq_len, tcfg.micro_batch, tcfg.eval_batches,
                                   device, amp_dtype)
            print("[eval] " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()
                                        if k.split("/")[-1] in ("all_loss", "copyable_loss", "lk_wrong_loss",
                                                                 "follow_wrong_lookup")), flush=True)
            logger.log(step=state["step"], tokens=state["tokens"], **{f"val/{k}": v for k, v in res.items()})

        # stop / checkpoint decisions must be identical on all ranks
        flags = torch.tensor([float((time.time() - t_start) / 3600 >= tcfg.max_hours),
                              float((time.time() - t_ckpt) / 60 >= tcfg.ckpt_every_minutes)], device=device)
        if world > 1:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        if flags[0] > 0:
            stop_reason = f"max_hours={tcfg.max_hours} reached"
        time_for_ckpt = bool(flags[1] > 0)
        finishing = state["step"] >= total_steps and not tcfg.keep_checkpoint    # model_final.pt follows
        if (stop_reason or time_for_ckpt or state["step"] >= total_steps) and not finishing:
            if zero:
                save_optimizer_shard(ckpt_path, opt, rank, world, state["step"])
            if main_proc:
                save_checkpoint(ckpt_path, raw_model, None if zero else opt, scaler, state)
                print(f"[train] checkpoint saved at step {state['step']}", flush=True)
            t_ckpt = time.time()
            if world > 1:
                dist.barrier()
        if stop_reason:
            break

    if main_proc:
        if state["step"] >= total_steps:
            final = os.path.join(tcfg.out_dir, "model_final.pt")
            torch.save(dict(model=raw_model.state_dict(), model_config=to_dict(mcfg), state=state), final + ".tmp")
            os.replace(final + ".tmp", final)
            # the run is complete: the resume checkpoint + optimizer shards are no longer needed (Kaggle disk is
            # ~20 GB; for 0.6b they are ~7 GB). Set keep_checkpoint to keep them.
            if not tcfg.keep_checkpoint:
                for f in os.listdir(tcfg.out_dir):
                    if f.startswith("ckpt_latest.pt"):
                        os.remove(os.path.join(tcfg.out_dir, f))
            print(f"[train] finished: {state['tokens']/1e6:,.1f}M tokens -> {final}")
        else:
            print(f"[train] stopped ({stop_reason}); re-run the same command to resume")
        logger.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
