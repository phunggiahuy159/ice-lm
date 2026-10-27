"""Model / training configs, size presets and experiment arms (A-F + ablations).

Arms follow the proposal (proposal_ice/proposal.tex, Section 2.2):

    A  layer-level 3:1                       no ICE
    B  layer-level 7:1                       no ICE
    C  layer-level 7:1                       ICE (input + output + trust signals, dropout 0.2)
    D  head-level hybrid, KV budget of B     no ICE
    E  head-level hybrid, KV budget of B     ICE
    F  layer-level 15:1                      ICE
    C_in / C_out / C_nodrop / C_notrust      ablations of C
    B_pm                                     B with as many parameters as C (capacity control)
    G / G_C                                  no attention (pure GDN, zero KV cache) without / with ICE
    G_MQA                                    15 GDN + 1 one-KV-head attention layer (equal-bytes baseline)
    G_V / G_VR                               G + ICE-V / ICE-VR: a lookup-blind backbone scores the lookup's
                                             candidates (log-space pointer-sentinel, see model.ICEPointer)
    G_V_in / G_V_nover                       G_V + input-side ICE / G_V without the backbone-logprob terms
    C_V / C_VR                               the same at 7:1
    G_VR2                                    G_VR with the ICE-V2 scorer (relative slot features, gamma >= 0, no
                                             beta) + the listwise slot loss (research/round2/decision2.md section 2)
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12            # total heads per layer (GDN heads + attention query heads)
    head_dim: int = 64
    n_kv_heads: int = 4          # KV heads of a full-attention layer (GQA)
    mlp_hidden: int = 2048
    # hybrid layout
    layout: str = "layer"        # "layer": whole layers are GDN or attention; "head": mixed heads; "gdn": no attention
    ratio: int = 3               # linear:full ratio for layout="layer"
    head_kv_budget_ratio: int = 7  # layout="head": match the KV budget of a layer-level hybrid with this ratio
    conv_size: int = 4
    rope_theta: float = 10000.0
    qk_norm: bool = True
    gdn_backend: str = "auto"    # "auto" | "fla" | "torch"
    gdn_chunk: int = 64
    gdn_compile: bool = False    # torch.compile the pure-PyTorch GDN kernel (only used when fla is not)
    tie_embeddings: bool = True
    grad_checkpointing: bool = False
    init_std: float = 0.02
    # ICE
    ice: bool = False
    ice_input: bool = True
    ice_output: bool = True
    ice_inject_after: int = 2    # inject after this many blocks
    ice_trust: bool = True       # agreement + online-reliability features (when to ignore a lookup)
    ice_orders: Tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    # ICE-V / VR: "" = off (the ICE above), "v" / "vr" = pointer-sentinel over the candidate set (replaces the
    # output-side ICE; the backbone sees no ICE unless ice_v_input)
    ice_v: str = ""
    ice_v_input: bool = False    # also inject the input-side ICE (ICEInput) into the backbone
    ice_v_verify: bool = True    # backbone log-prob terms in the slot / sentinel scores (gamma, beta)
    ice_v_aux: float = 0.5       # weight of the auxiliary CE of the backbone logits
    ice_v2: bool = False         # ICE-V2 scorer (model.ICEPointer): + R/A/N/Q slot-feature tables, softplus gamma, no beta
    ice_v2_feats: str = "dnrq"   # V2 tables used: d = match margin, n = agreement count, r = recency rank, q = l rank
    ice_v_listwise: float = 0.0  # weight of the listwise slot CE (0 = off)


@dataclass
class TrainConfig:
    preset: str = "140m"
    arm: str = "A"
    data_dir: str = "data"
    out_dir: str = "runs/A"
    seq_len: int = 4096
    micro_batch: int = 4             # sequences per GPU per micro-step
    global_batch_tokens: int = 2 ** 19   # ~524K tokens per optimizer step
    total_tokens: int = 2_500_000_000
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_tokens: int = 50_000_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    ice_dropout: float = 0.2
    seed: int = 1234
    precision: str = "auto"          # "auto" | "bf16" | "fp16" | "fp32"
    compile: bool = False
    zero_optimizer: bool = True      # multi-GPU: shard the Adam states over the GPUs (ZeRO-1)
    num_workers: int = 2
    log_every: int = 10
    eval_every: int = 250
    eval_batches: int = 8
    ckpt_every_minutes: float = 30.0
    max_hours: float = 1e9           # stop (and checkpoint) after this wall-clock time
    resume: bool = True
    keep_resume_config: bool = True  # a resumed run keeps its saved schedule / data-order settings
    keep_checkpoint: bool = False    # keep ckpt_latest.pt (+ optimizer shards) after model_final.pt is written
    schedule: str = "cosine"         # "cosine" (warmup, cosine to min_lr_ratio) | "wsd" (warmup, constant, linear decay)
    decay_frac: float = 0.2          # wsd: share of the steps in the final decay
    branch_tokens: str = ""          # wsd: comma-separated shorter budgets (e.g. "2e8,4e8"); the run saves
                                     # ckpt_branch_<tokens>.pt where each one's decay starts
    branch_from: str = ""            # wsd: start this (shorter-budget) run from such a ckpt_branch_*.pt


# ---------------------------------------------------------------------------
# Size presets
# ---------------------------------------------------------------------------
PRESETS = {
    # tiny model for CPU tests / smoke runs
    "smoke": dict(
        model=dict(d_model=128, n_layers=4, n_heads=4, head_dim=32, n_kv_heads=2, mlp_hidden=256,
                   gdn_chunk=32),
        train=dict(seq_len=256, micro_batch=8, global_batch_tokens=256 * 8, total_tokens=2_000_000,
                   lr=2e-3, warmup_tokens=100_000, eval_every=50, eval_batches=4, log_every=10,
                   num_workers=0),
    ),
    # main scale: ~150M parameters. Recipe for a ~400M-token budget (2-3 tokens/param, Kaggle 2xT4):
    # batch 131K tokens -> ~3K optimizer steps, lr 1e-3 (QK-norm keeps it stable), 5% warmup, cosine to 10%
    "140m": dict(
        model=dict(d_model=768, n_layers=16, n_heads=12, head_dim=64, n_kv_heads=4, mlp_hidden=2048),
        train=dict(seq_len=4096, micro_batch=4, global_batch_tokens=2 ** 17,
                   total_tokens=400_000_000, lr=1e-3, warmup_tokens=20_000_000),
    ),
    # ~0.6B parameters (with a 32K vocabulary); at ~300M tokens this is a sanity check, not a converged model
    "0.6b": dict(
        model=dict(d_model=1280, n_layers=24, n_heads=10, head_dim=128, n_kv_heads=2, mlp_hidden=3840,
                   grad_checkpointing=True),
        train=dict(seq_len=4096, micro_batch=1, global_batch_tokens=2 ** 18,
                   total_tokens=300_000_000, lr=6e-4, warmup_tokens=20_000_000),
    ),
}

ARMS = {
    "A": dict(layout="layer", ratio=3, ice=False),
    "B": dict(layout="layer", ratio=7, ice=False),
    "C": dict(layout="layer", ratio=7, ice=True),
    "D": dict(layout="head", head_kv_budget_ratio=7, ice=False),
    "E": dict(layout="head", head_kv_budget_ratio=7, ice=True),
    "F": dict(layout="layer", ratio=15, ice=True),
    # no attention at all (zero KV cache): can the lookup stand in for attention?
    "G": dict(layout="gdn", ice=False),
    "G_C": dict(layout="gdn", ice=True),
    # equal-bytes attention baseline: 15 GDN + 1 full-attention layer with ONE KV head (256 B/token at every length,
    # 4x the ICE-VR table) -- also the missing 15:1 arm without ICE
    "G_MQA": dict(layout="layer", ratio=15, ice=False, n_kv_heads=1),
    # ablations of C
    "C_in": dict(layout="layer", ratio=7, ice=True, ice_output=False),
    "C_out": dict(layout="layer", ratio=7, ice=True, ice_input=False),
    "C_nodrop": dict(layout="layer", ratio=7, ice=True, _ice_dropout=0.0),
    "C_notrust": dict(layout="layer", ratio=7, ice=True, ice_trust=False),   # original ICE (no agree / rel)
    # control: B with a wider MLP so it has as many parameters as C (is C's gain just extra capacity?)
    "B_pm": dict(layout="layer", ratio=7, ice=False, _match_params_of="C"),
    # ICE-V(R): the lookup proposes candidates, a backbone that never sees the lookup decides. No ICE-dropout:
    # the auxiliary CE trains the ICE-off path on every token
    "G_V": dict(layout="gdn", ice=True, ice_v="v", _ice_dropout=0.0),
    "G_VR": dict(layout="gdn", ice=True, ice_v="vr", _ice_dropout=0.0),
    "G_V_in": dict(layout="gdn", ice=True, ice_v="v", ice_v_input=True, _ice_dropout=0.0),
    "G_V_nover": dict(layout="gdn", ice=True, ice_v="v", ice_v_verify=False, _ice_dropout=0.0),
    "C_V": dict(layout="layer", ratio=7, ice=True, ice_v="v", _ice_dropout=0.0),
    "C_VR": dict(layout="layer", ratio=7, ice=True, ice_v="vr", _ice_dropout=0.0),
    # ICE-V2, trained jointly (G_VR's gradient paths: p_bb(y) and h attached, aux CE 0.5) + the listwise term
    "G_VR2": dict(layout="gdn", ice=True, ice_v="vr", ice_v2=True, ice_v_listwise=0.25, _ice_dropout=0.0),
}


def ice_param_count(m: ModelConfig) -> int:
    """Parameters ICE adds to a model (input module with trust embeddings + output module, or the ICE-V pointer)."""
    from .ice_features import (CNT_MAX, MAX_MATCH, N_AGREE_BUCKETS, N_DELTA, N_DIST_BUCKETS, N_LEN_BUCKETS,
                               N_REL_BUCKETS, REC_MAX, slot_groups, n_slots)
    d, n = m.d_model, len(m.ice_orders)
    p_in = n * d * d + (N_LEN_BUCKETS + N_DIST_BUCKETS) * d + 2 * d
    p_out = d + d + N_LEN_BUCKETS
    if m.ice_trust:
        p_in += n * (N_AGREE_BUCKETS + N_REL_BUCKETS) * d
        p_out += N_AGREE_BUCKETS + N_REL_BUCKETS
    if m.ice_v:
        k = n_slots(m.ice_v, n)
        # norm + w (sentinel), b[src], L[src, len], D[dist], c0 (+ gamma[src], beta)
        p_ptr = 2 * d + k + k * (MAX_MATCH + 1) + N_DIST_BUCKETS + 1 + ((k + 1) if m.ice_v_verify else 0)
        if m.ice_v2:   # no beta; per-group tables R / A / N / Q (model.V2_TABLES)
            nb = dict(d=N_DELTA, n=CNT_MAX + 1, r=REC_MAX + 1, q=3)
            p_ptr += (max(slot_groups(m.ice_v, n)) + 1) * sum(nb[f] for f in set(m.ice_v2_feats)) \
                - (1 if m.ice_v_verify else 0)
        return (p_in if m.ice_v_input else 0) + p_ptr
    return (p_in if m.ice_input else 0) + (p_out if m.ice_output else 0)


def build_configs(preset: str, arm: str, **overrides) -> Tuple[ModelConfig, TrainConfig]:
    """Combine preset + arm + explicit overrides (overrides may target either config)."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {list(ARMS)}")
    m = ModelConfig(**PRESETS[preset]["model"])
    t = TrainConfig(preset=preset, arm=arm, **PRESETS[preset]["train"])
    arm_cfg = dict(ARMS[arm])
    if "_ice_dropout" in arm_cfg:
        t.ice_dropout = arm_cfg.pop("_ice_dropout")
    match = arm_cfg.pop("_match_params_of", None)
    m = dataclasses.replace(m, **arm_cfg)
    if match is not None:
        # widen the MLP (3 * d_model params per layer per hidden unit) to absorb the other arm's ICE parameters
        other = dataclasses.replace(ModelConfig(**PRESETS[preset]["model"]),
                                    **{k: v for k, v in ARMS[match].items() if not k.startswith("_")})
        per_unit = 3 * m.d_model * m.n_layers
        m = dataclasses.replace(m, mlp_hidden=m.mlp_hidden + -(-ice_param_count(other) // per_unit))
    mfields = {f.name for f in dataclasses.fields(ModelConfig)}
    tfields = {f.name for f in dataclasses.fields(TrainConfig)}
    for k, v in overrides.items():
        if v is None:
            continue
        if k in mfields:
            m = dataclasses.replace(m, **{k: v})
        elif k in tfields:
            t = dataclasses.replace(t, **{k: v})
        else:
            raise ValueError(f"unknown config key {k!r}")
    return m, t


def to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)


def model_config_from_dict(d: dict) -> ModelConfig:
    d = dict(d)
    d.setdefault("ice_trust", False)      # runs saved before the trust features existed
    if "ice_orders" in d:
        d["ice_orders"] = tuple(d["ice_orders"])
    return ModelConfig(**d)


def dump_json(path: str, obj) -> None:
    """Atomic write: other processes (DDP ranks) may read the file at the same time."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
