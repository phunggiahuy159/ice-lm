"""Synthetic conflict / override items for the ICE-V2+syn scorer fit, and a held-out set of perturbed templates.

decision2.md section 2 (V2+syn): a small fraction of the fit windows are replaced by windows of synthetic items in
which several occurrences of the same key compete (conflicts: the record named in the query holds the answer;
overrides: the most recent write holds it, written with a different syntax than the query). Every template family
here is DISJOINT from CodeRecall's (ice_lm/coderecall.py): other key names and object types, >= 3 records or
writes, other syntax, and no `Server(...)`, `settings`, `config[...]`, `config.update(...)`, `os.environ`,
`Migration(...)`.

  FIT families (training windows only):
    fit_conflict_call      register_queue(topic="refunds", durable=True, ..., max_inflight=4821) x 3..8, the
                           query repeats one record (never the last) up to the key
    fit_conflict_dict      limits["search"] = {"owner": "ops", ..., "cpu_millis": 4821} x 3..8
    fit_override_setattr   opts.max_jobs = v1, then >= 2 later writes via setattr(opts, "max_jobs", v) /
                           opts.set("max_jobs", v), asked with the original syntax: the answer is the last write
  HELD-OUT perturbed families (evaluation only, never in a fit window):
    ho_conflict_positional add_route("/users", "GET", 30, 4821): positional arguments, the identifier first
    ho_conflict_swapped    Probe(kind="http", host="billing", interval=5, budget=4821): shared fields on both
                           sides of the identifier, the target may be the FIRST record
    ho_override_state      self.spill_mb = v1 in a class, then 3..4 writes via app.state.spill_mb = v /
                           patch_state(spill_mb=v) / runtime_set("spill_mb", v), asked as `self.spill_mb =`

All values in an item are 4-digit numbers with different leading digits, so the first digit decides (scored like
CodeRecall: coderecall.build_items / evals.score_answers). Items are tagged "task-supervised" in the results.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .coderecall import _encoder, _values, build_items
from .data import TrainWindows, make_example
from .evals import encode_with_answer, score_answers

TOPICS = ["invoices", "signups", "refunds", "alerts", "exports", "clicks", "payouts", "uploads", "digests",
          "receipts", "tickets", "metrics"]
QUEUE_KEYS = ["max_inflight", "prefetch", "ack_deadline", "dlq_after", "visibility_s"]
QUEUE_SHARED = ['durable=True', 'acks="all"', 'codec="zstd"', "partitions=8", "compact=False"]
TEAMS = ["search", "billing_ui", "growth", "infra", "mobile", "payments", "identity", "ads", "maps", "support",
         "storage", "video"]
DICT_SHARED = ['"owner": "ops"', '"tier": 2', '"region": "us"', '"burst": False']
DICT_KEYS = ["cpu_millis", "mem_mb", "disk_gb", "iops"]
OPT_OBJS = ["opts", "runtime", "job_cfg", "tuning"]
OPT_KEYS = ["max_jobs", "spill_mb", "fanout", "prefetch_depth", "idle_secs", "burst_cap", "sweep_every",
            "keep_last"]
PATHS = ["/users", "/orders", "/carts", "/search", "/login", "/upload", "/export", "/health", "/billing", "/admin",
         "/feed", "/inbox"]
HOSTS = ["billing", "auth", "catalog", "ledger", "notify", "gateway", "media", "report", "scheduler", "tracker",
         "vault", "wiki"]
PROBE_PRE = ['kind="http"', "port=443"]
PROBE_POST = ["interval=5", "jitter=2", 'path="/ping"']
PROBE_KEYS = ["budget", "slo_ms", "cooldown"]
STATE_KEYS = ["spill_mb", "max_conns_hint", "reap_after", "drain_ms", "hedge_ms", "backfill_cap"]

FIT_FAMILIES = ("fit_conflict_call", "fit_conflict_dict", "fit_override_setattr")
HELDOUT_FAMILIES = ("ho_conflict_positional", "ho_conflict_swapped", "ho_override_state")
FAMILIES = FIT_FAMILIES + HELDOUT_FAMILIES


def _pick(rng, pool, n):
    return [str(v) for v in rng.choice(pool, size=n, replace=False)]


def _conflict(rng, n_distract, records, query, last_ok=False):
    """records(i) -> line of record i, query(j) -> query text up to the key; answer = value of record j."""
    n = int(rng.integers(3, int(np.clip(n_distract + 1, 3, 8)) + 1))      # 3..8 records
    vals = _values(rng, n)
    j = int(rng.integers(n if last_ok else n - 1))
    head = "".join(records(i, vals[i]) for i in range(n))
    return [head, None, query(j)], vals[j]


def gen_fit_conflict_call(rng, n_distract):
    topics = _pick(rng, TOPICS, 8)
    key = str(rng.choice(QUEUE_KEYS))
    shared = "".join(f"{e}, " for e in _pick(rng, QUEUE_SHARED, int(rng.integers(1, 5))))
    parts, v = _conflict(rng, n_distract, lambda i, v: f'register_queue(topic="{topics[i]}", {shared}{key}={v})\n',
                         lambda j: f'\nprimary = register_queue(topic="{topics[j]}", {shared}{key}=')
    return parts, f"{v}"


def gen_fit_conflict_dict(rng, n_distract):
    teams = _pick(rng, TEAMS, 8)
    key = str(rng.choice(DICT_KEYS))
    shared = "".join(f"{e}, " for e in _pick(rng, DICT_SHARED, int(rng.integers(1, 4))))
    parts, v = _conflict(rng, n_distract, lambda i, v: f'limits["{teams[i]}"] = {{{shared}"{key}": {v}}}\n',
                         lambda j: f'\nlimits["{teams[j]}"] = {{{shared}"{key}":')
    return parts, f" {v}"


def _override(rng, n_writes, first, later, query, distract):
    """first / later(i) / query: text of the original write, of later writes (different syntax) and of the
    query (original syntax); distract(v) -> lines for other keys. Returns parts with a gap between writes."""
    vals = _values(rng, n_writes + 2)
    parts = [distract(vals[n_writes:]) + first(vals[0])]
    for i in range(1, n_writes):
        parts += [None, later(int(rng.integers(1 << 30)), vals[i])]
    return parts + [None, query], f" {vals[n_writes - 1]}"


def gen_fit_override_setattr(rng, n_distract):
    obj = str(rng.choice(OPT_OBJS))
    key, *others = _pick(rng, OPT_KEYS, 3)
    forms = [lambda v: f'setattr({obj}, "{key}", {v})  # tuned\n', lambda v: f'{obj}.set("{key}", {v})\n']
    return _override(rng, int(rng.integers(3, 5)), lambda v: f"{obj}.{key} = {v}\n",
                     lambda r, v: "\n" + forms[r % 2](v), f"\n# current value\n{obj}.{key} =",
                     lambda vs: "".join(f"{obj}.{k} = {v}\n" for k, v in zip(others, vs)))


def gen_ho_conflict_positional(rng, n_distract):
    paths = _pick(rng, PATHS, 8)
    shared = ", ".join(str(v) for v in [f'"{rng.choice(["GET", "POST", "PUT"])}"', int(rng.integers(5, 90)),
                                        "True"][:int(rng.integers(2, 4))])
    parts, v = _conflict(rng, n_distract, lambda i, v: f'add_route("{paths[i]}", {shared}, {v})\n',
                         lambda j: f'\nrouter.mount(add_route("{paths[j]}", {shared},')
    return parts, f" {v}"


def gen_ho_conflict_swapped(rng, n_distract):
    hosts = _pick(rng, HOSTS, 8)
    key = str(rng.choice(PROBE_KEYS))
    pre = "".join(f"{e}, " for e in PROBE_PRE)
    post = "".join(f"{e}, " for e in _pick(rng, PROBE_POST, int(rng.integers(1, 4))))
    parts, v = _conflict(rng, n_distract, lambda i, v: f'    Probe({pre}host="{hosts[i]}", {post}{key}={v}),\n',
                         lambda j: f'\nwatch(Probe({pre}host="{hosts[j]}", {post}{key}=')
    parts[0] = "probes = (\n" + parts[0] + ")\n"
    return parts, f"{v}"


def gen_ho_override_state(rng, n_distract):
    key, *others = _pick(rng, STATE_KEYS, 3)
    forms = [lambda v: f"app.state.{key} = {v}\n", lambda v: f"patch_state({key}={v})\n",
             lambda v: f'runtime_set("{key}", {v})  # hot reload\n']
    return _override(rng, int(rng.integers(4, 6)), lambda v: f"        self.{key} = {v}\n",
                     lambda r, v: "\n" + forms[r % 3](v), f"\n    def reload(self):\n        self.{key} =",
                     lambda vs: "class Worker:\n    def __init__(self):\n"
                     + "".join(f"        self.{k} = {v}\n" for k, v in zip(others, vs)))


_GEN = {f: globals()[f"gen_{f}"] for f in FAMILIES}


def make_syn_item(family: str, rng, n_distract: int):
    """(parts, answer) like coderecall.make_item (None = a gap of real code)."""
    return _GEN[family](rng, n_distract)


# ------------------------------------------------------------------ fit windows
def item_pool(tok, n_items: int, seed: int = 0, families: Sequence[str] = FIT_FAMILIES,
              n_distract: int = 8) -> List[List[Optional[List[int]]]]:
    """Tokenized FIT items: each a list of token-id parts and None (gap). The query is encoded with its answer."""
    enc, n_nl = _encoder(tok)
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    rng = np.random.default_rng([seed, 5501])
    pool = []
    for i in range(n_items):
        parts, answer = make_syn_item(families[i % len(families)], rng, n_distract)
        q_full, _ = encode_with_answer(tok, "\n" + parts[-1], answer)
        cut = n_nl if q_full[:n_nl] == nl else 0
        pool.append([None if p is None else enc(p) for p in parts[:-1]] + [q_full[cut:]])
    return pool


def syn_window(pool, rng, data, n_tokens: int, eos_id: Optional[int], gap=(32, 768)) -> np.ndarray:
    """n_tokens of synthetic items packed back to back, gaps (and a random lead-in) filled with real training
    tokens (EOS removed, so the window is one document)."""
    def filler(g):
        start = int(rng.integers(0, max(1, len(data) - g - 1)))
        return [t for t in np.asarray(data[start: start + g]).tolist() if t != eos_id]

    out = filler(int(rng.integers(0, gap[1])))
    while len(out) < n_tokens:
        for p in pool[int(rng.integers(len(pool)))]:
            out += filler(int(rng.integers(*gap))) if p is None else p
    return np.asarray(out[:n_tokens], dtype=np.int64)


class SynMixWindows(TrainWindows):
    """TrainWindows where sample i is replaced, with probability syn_frac (fixed per index), by a window of
    synthetic FIT items (syn_window)."""

    def __init__(self, *args, pool=None, syn_frac: float = 0.0, **kw):
        super().__init__(*args, **kw)
        self.pool, self.syn_frac = pool, syn_frac

    def sample(self, data, i: int, cache: dict):
        r = np.random.default_rng([self.seed, 4241, i])
        if not self.pool or r.random() >= self.syn_frac:
            return super().sample(data, i, cache)
        w = syn_window(self.pool, r, data, self.seq_len + 1, self.eos)
        ex = make_example(w, self.eos, self.orders, self.with_ice, self.cand, self.v2)
        ex["ice_keep"] = torch.tensor(1.0)
        return ex


# ------------------------------------------------------------------ evaluation
def run_syn_eval(model, tok, filler: np.ndarray, eos_id: int, device, amp_dtype,
                 gaps: Sequence[int] = (256, 1024), n: int = 100, seed: int = 1, n_distract: int = 8,
                 ice_keep: float = 1.0, families: Sequence[str] = FAMILIES) -> Dict[str, float]:
    """Deciding-digit accuracy per family and gap (keys syn/<family>@<gap>/<metric>), like run_coderecall. FIT
    families with these seeds are fresh items of the fitted templates; HELDOUT families were never fitted."""
    res = {}
    for fam in families:
        for gap in gaps:
            rng = np.random.default_rng([seed, 9173, FAMILIES.index(fam), gap])
            seqs, spans, decide = build_items(tok, rng, fam, n, gap, filler, eos_id, n_distract, make=make_syn_item)
            d = score_answers(model, seqs, spans, device, amp_dtype, ice_keep, details=True, decide=decide)
            for k in ("em", "dec_acc", "dec_acc_lookup_right", "dec_acc_lookup_wrong", "dec_lookup_wrong_frac",
                      "dec_follow_wrong_lookup"):
                res[f"syn/{fam}@{gap}/{k}"] = d[k]
            res[f"syn/{fam}@{gap}/items_dec"] = d["items_dec"]
    return res


def summarize_syn(res: Dict[str, float], metric: str = "dec_acc") -> Dict[str, float]:
    """Mean over gaps per family, plus the means over the FIT and the HELD-OUT families."""
    out = {}
    for fam in FAMILIES:
        v = [x for k, x in res.items() if k.startswith(f"syn/{fam}@") and k.endswith(f"/{metric}")
             and not (isinstance(x, float) and np.isnan(x))]
        if v:
            out[fam] = float(np.mean(v))
    for name, fams in (("fit_mean", FIT_FAMILIES), ("heldout_mean", HELDOUT_FAMILIES)):
        v = [out[f] for f in fams if f in out]
        if v:
            out[name] = float(np.mean(v))
    return out
