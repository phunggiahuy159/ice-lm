"""Distributed helpers, precision selection, optimizer, LR schedule, checkpoints, logging."""
from __future__ import annotations

import datetime
import json
import math
import os
import time
import uuid
from typing import Optional

import torch
import torch.distributed as dist


def setup_distributed():
    """Returns (rank, world_size, local_rank, device). Works with or without torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # generous timeout: the other ranks wait while rank 0 runs the periodic evaluation alone
        dist.init_process_group(backend, timeout=datetime.timedelta(minutes=60))
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            device = torch.device("cuda", local)
        else:
            device = torch.device("cpu")
        return rank, world, local, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device


def is_main(rank: int) -> bool:
    return rank == 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def pick_precision(requested: str, device: torch.device):
    """-> (amp_dtype or None, use_grad_scaler)."""
    if device.type != "cuda" or requested == "fp32":
        return None, False
    if requested == "bf16" or (requested == "auto" and torch.cuda.is_bf16_supported()):
        # T4 / P100 report bf16 as "supported" only via slow emulation on some torch versions
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8 or requested == "bf16":
            return torch.bfloat16, False
    return torch.float16, True


def build_optimizer(model, lr, weight_decay, betas, device, zero: bool = False):
    """AdamW; zero=True shards the Adam states over the DDP ranks (ZeRO-1: 0.6B on 2 GPUs saves ~2.4 GB each)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ice_table = name.startswith("ice_") and name.split(".")[-2] in ("L", "D", "b", "A", "R", "a", "r")
        if p.ndim < 2 or getattr(p, "_no_weight_decay", False) or "norm" in name or "embed" in name or ice_table:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]
    kw = dict(lr=lr, betas=betas)
    if device.type == "cuda":
        kw["fused"] = True
    def make(**kw):
        if zero:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            return ZeroRedundancyOptimizer(groups, optimizer_class=torch.optim.AdamW, **kw)
        return torch.optim.AdamW(groups, **kw)

    try:
        return make(**kw)
    except (TypeError, RuntimeError):
        kw.pop("fused", None)
        return make(**kw)


def lr_at(step: int, total_steps: int, warmup_steps: int, lr: float, min_ratio: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def save_checkpoint(path: str, model, optimizer, scaler, state: dict):
    tmp = path + ".tmp"
    raw = model.module if hasattr(model, "module") else model
    torch.save(dict(model=raw.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
                    scaler=scaler.state_dict() if scaler is not None else None, state=state), tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model, optimizer=None, scaler=None, map_location="cpu") -> dict:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    raw = model.module if hasattr(model, "module") else model
    try:
        raw.load_state_dict(ck["model"])
    except RuntimeError as e:
        raise RuntimeError(f"checkpoint {path} does not fit this model (different preset/arm/vocab?). "
                           f"Use a new out_dir or delete the old checkpoint.\n{e}") from None
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scaler is not None and ck.get("scaler") is not None:
        scaler.load_state_dict(ck["scaler"])
    return ck.get("state", {})


def _shard_path(path: str, rank: int, world: int) -> str:
    return f"{path}.optim{rank}of{world}"


def save_optimizer_shard(path: str, optimizer, rank: int, world: int, step: int):
    """ZeRO: every rank saves its own optimizer partition (no gather of the full state onto one GPU)."""
    tmp = _shard_path(path, rank, world) + ".tmp"
    torch.save(dict(optim=optimizer.optim.state_dict(), step=step), tmp)
    os.replace(tmp, _shard_path(path, rank, world))


def load_optimizer_shard(path: str, optimizer, rank: int, world: int, step: int, map_location="cpu") -> bool:
    p = _shard_path(path, rank, world)
    if not os.path.exists(p):
        return False
    ck = torch.load(p, map_location=map_location, weights_only=False)
    if ck.get("step") != step:
        return False
    optimizer.optim.load_state_dict(ck["optim"])
    return True


def init_wandb(out_dir: str, name: str, config: dict):
    """Weights & Biases run when WANDB_API_KEY is set (e.g. from a Modal secret), else None. The run id is kept in
    out_dir/wandb_id.txt, so a resumed training run continues the same W&B run. Any W&B error disables it."""
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
        id_path = os.path.join(out_dir, "wandb_id.txt")
        run_id = open(id_path).read().strip() if os.path.exists(id_path) else uuid.uuid4().hex[:12]
        with open(id_path, "w") as f:
            f.write(run_id)
        return wandb.init(project=os.environ.get("WANDB_PROJECT", "ice-lm"), name=name, id=run_id, resume="allow",
                          config=config, dir=out_dir)
    except Exception as e:  # noqa: BLE001 - logging must never stop training
        print(f"[wandb] disabled ({type(e).__name__}: {e})")
        return None


class JsonlLogger:
    def __init__(self, path: Optional[str], wandb_run=None):
        self.path = path
        self.wandb = wandb_run

    def log(self, **kv):
        kv["time"] = time.time()
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(kv) + "\n")
        if self.wandb is not None:
            try:
                self.wandb.log({k: v for k, v in kv.items() if k not in ("step", "time")}, step=kv.get("step"))
            except Exception as e:  # noqa: BLE001
                print(f"[wandb] log failed, disabling ({type(e).__name__}: {e})")
                self.wandb = None

    def finish(self):
        if self.wandb is not None:
            self.wandb.finish()
