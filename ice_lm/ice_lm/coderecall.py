"""CodeRecall: one concrete use case for ICE -- recalling identifiers and configuration values in Python code.

Every item is short template code embedded in held-out real code (val_code), scored teacher-forced. Values
are 4-digit numbers and all values inside one item have DIFFERENT leading digits, so the first digit alone says
whether the right value was recalled. Llama-style tokenizers split numbers into digits and, teacher-forced, the
later digits can simply be copied once the first one is given, so the headline metric is accuracy on the
DECIDING token (the first digit; the last digit for `increment`), split by what the exact lookup suggested
for that token (right / wrong / none), plus exact match of the whole value.

  copy_same      value defined once, asked with the SAME syntax           lookup mostly right  (ICE should help)
  copy_diff      value defined once, asked with DIFFERENT wording
                 (settings.x / settings.get("x") / os.environ["X"])        lookup useless (needs attention)
  conflict_g{g}  same key under several objects; the object's name sits g identical kwargs before the key.
                 The target is never the last object, so for large g the lookup (most recent occurrence)
                 points at another object's value                          g=1 right, g>=2 wrong
  override_same  value reassigned later; asked with the same syntax     lookup right   (recency)
  override_diff  value changed later with a different syntax (config.update), asked with the ORIGINAL syntax:
                 the lookup points at the stale value                     lookup wrong   ("revise a value")
  increment      Migration(id=1041), (id=1042), ... next id             last digit: lookup wrong (copying is wrong)
  track          x = 3; x = x + 4; print(x)  # prints 7                   lookup points at an operand (wrong / none)

Statuses are measured per item, not assumed: the split by lookup status is what to read.
Every arm scores identical tokens.

Example (override_diff):
    config["timeout"] = 30
    config["retries"] = 5
    <~D tokens of real code>
    config.update(timeout=60)  # raised after the outage
    <~D tokens of real code>
    # re-apply the current value
    config["timeout"] =  ->  " 60"
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .evals import encode_with_answer, score_answers

KEYS = ["db_port", "cache_ttl", "max_retries", "worker_count", "batch_size", "timeout_ms", "pool_size",
        "queue_limit", "log_every", "shard_count", "page_size", "rate_limit", "buffer_kb", "max_conns",
        "grace_period", "flush_interval", "chunk_bytes", "retry_delay", "lease_secs", "warmup_steps"]
CONSTS = ["TIMEOUT", "RETRIES", "MAX_WORKERS", "PORT", "BATCH", "CACHE_SIZE", "POOL", "LIMIT", "INTERVAL",
          "BACKLOG", "QUOTA", "WINDOW"]
SERVERS = ["dev", "prod", "staging", "qa", "canary", "backup", "edge", "batch", "ml", "api", "web", "db"]
EXTRAS = ['region="eu"', "replicas=3", 'tier="gold"', "healthcheck=True", "weight=10"]
MIGRATIONS = ["add_users", "add_orders", "index_email", "drop_legacy", "add_invoices", "split_names",
              "add_audit", "fix_dates", "add_tags", "merge_carts"]
GAPS = (1, 2, 5)          # with the TinyLlama tokenizer: g=1 lookup right, g>=2 wrong (beyond order 8)
CATEGORIES = ("copy_same", "copy_diff", *(f"conflict_g{g}" for g in GAPS), "override_same", "override_diff",
              "increment", "track")


def _values(rng, n: int) -> List[int]:
    """n <= 9 four-digit values with pairwise different leading digits."""
    lead = rng.choice(np.arange(1, 10), size=n, replace=False)
    return [int(d) * 1000 + int(rng.integers(0, 1000)) for d in lead]




# Each generator returns (parts, answer): parts is a list of strings and None (= a gap of real code); the last
# part is the query, the answer is the expected continuation. `gap` is the TOTAL amount of filler per item (split
# evenly when an item has two gaps), so items stay within the training length (default gaps up to 3072 < 4096):
# beyond it attention must extrapolate while ICE does not, which would confound the comparison.
def gen_copy(rng, n_distract: int, same: bool):
    """Defined once in a dict (or as environment variables); asked with the same or a different wording."""
    keys = list(rng.choice(KEYS, size=min(9, n_distract + 1), replace=False))
    vals = _values(rng, len(keys))
    j = int(rng.integers(len(keys)))
    head = "settings = {\n" + "".join(f'    "{k}": {v},\n' for k, v in zip(keys, vals)) + "}\n"
    if same:
        return [head, None, f'\noverride = {{"{keys[j]}":'], f" {vals[j]}"
    form = int(rng.integers(3))
    if form == 0:
        return [head, None, f"\nassert settings.{keys[j]} =="], f" {vals[j]}"
    if form == 1:
        return [head, None, f'\nlimit = settings.get("{keys[j]}", default='], f"{vals[j]}"
    head = "".join(f'os.environ["{k.upper()}"] = "{v}"\n' for k, v in zip(keys, vals))
    return [head, None, f"\nlimit = int(os.getenv('{keys[j].upper()}', '"], f"{vals[j]}"


def gen_conflict(rng, n_distract: int, gap: int):
    names = list(rng.choice(SERVERS, size=min(9, max(2, n_distract + 1)), replace=False))
    ports = _values(rng, len(names))
    extras = "".join(f"{e}, " for e in EXTRAS[:gap])
    j = int(rng.integers(len(names) - 1))            # never the last (most recent) server
    head = "servers = [\n" + "".join(f'    Server(name="{n}", {extras}port={p}),\n' for n, p in zip(names, ports)) + "]\n"
    query = f'\nprimary = Server(name="{names[j]}", {extras}port='
    return [head, None, query], f"{ports[j]}"


def gen_override(rng, n_distract: int, same: bool):
    if same:
        names = list(rng.choice(CONSTS, size=min(8, n_distract + 1), replace=False))
        vals = _values(rng, len(names) + 1)
        head = "".join(f"{k} = {v}\n" for k, v in zip(names, vals))
        new, j = vals[-1], int(rng.integers(len(names)))          # override a random line, not always the first
        mid = f"\n{names[j]} = {new}  # raised after the outage\n"
        query = f"\n# keep the current value\n{names[j]} ="
    else:
        keys = list(rng.choice(KEYS, size=min(len(KEYS), n_distract + 1), replace=False))[:8]
        vals = _values(rng, len(keys) + 1)
        head = "".join(f'config["{k}"] = {v}\n' for k, v in zip(keys, vals))
        new, j = vals[-1], int(rng.integers(len(keys)))
        mid = f"\nconfig.update({keys[j]}={new})  # raised after the outage\n"
        query = f'\n# re-apply the current value\nconfig["{keys[j]}"] ='
    return [head, None, mid, None, query], f" {new}"


def gen_increment(rng, n_items: int = 4):
    base = int(rng.integers(100, 900)) * 10 + int(rng.integers(0, 10 - n_items))   # no carry: only the last digit moves
    names = list(rng.choice(MIGRATIONS, size=n_items + 1, replace=False))
    lines = "".join(f'    Migration(id={base + i}, name="{names[i]}"),\n' for i in range(n_items))
    return [None, "\nMIGRATIONS = [\n" + lines + "    Migration(id="], f"{base + n_items}"


def gen_track(rng):
    a = int(rng.integers(0, 6))
    b = int(rng.integers(1, 10 - a))
    var = str(rng.choice(["x", "count", "total", "n", "acc"]))
    return [None, f"\n{var} = {a}\n{var} = {var} + {b}\nprint({var})  # prints"], f" {a + b}"


def make_item(category: str, rng, n_distract: int):
    if category == "copy_same":
        return gen_copy(rng, n_distract, True)
    if category == "copy_diff":
        return gen_copy(rng, n_distract, False)
    if category.startswith("conflict_g"):
        return gen_conflict(rng, n_distract, int(category[len("conflict_g"):]))
    if category == "override_same":
        return gen_override(rng, n_distract, True)
    if category == "override_diff":
        return gen_override(rng, n_distract, False)
    if category == "increment":
        return gen_increment(rng)
    if category == "track":
        return gen_track(rng)
    raise ValueError(category)


def _encoder(tok):
    """Tokenize text as it would appear in the middle of a file (after a newline): SentencePiece otherwise adds a
    word-initial "▁" to every separately encoded part, which changes how the first identifier is split."""
    nl = tok("\n", add_special_tokens=False)["input_ids"]

    def enc(text: str) -> List[int]:
        ids = tok("\n" + text, add_special_tokens=False)["input_ids"]
        return ids[len(nl):] if ids[:len(nl)] == nl else ids
    return enc, len(nl)


def build_items(tok, rng, category: str, n: int, gap_tokens: int, filler: np.ndarray, eos_id: int,
                n_distract: int = 8):
    """-> (seqs, spans, deciding positions). Gaps are filled with real code (EOS removed). The answer span covers
    the digits only (a leading "▁" token, if any, is moved into the prompt); the deciding position is the
    first digit (last digit for `increment`)."""
    enc, n_nl = _encoder(tok)
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    seqs, spans, decide = [], [], []
    for _ in range(n):
        parts, answer = make_item(category, rng, n_distract)
        n_gaps = max(1, sum(p is None for p in parts))
        g = gap_tokens // n_gaps
        ids: List[int] = []
        for p in parts[:-1]:
            if p is None:
                start = int(rng.integers(0, max(1, len(filler) - g - 1)))
                ids += [t for t in np.asarray(filler[start: start + g]).tolist() if t != eos_id]
            else:
                ids += enc(p)
        q_full, (qa, qb) = encode_with_answer(tok, "\n" + parts[-1], answer)
        cut = n_nl if q_full[:n_nl] == nl else 0
        q_ids, qa, qb = q_full[cut:], qa - cut, qb - cut
        while qa < qb - 1 and not tok.decode([q_ids[qa]]).strip():          # whitespace-only token: not an answer
            qa += 1
        a0 = len(ids)
        spans.append((a0 + qa, a0 + qb))
        decide.append(a0 + (qb - 1 if category == "increment" else qa))
        seqs.append(ids + q_ids)
    return seqs, spans, decide


def run_coderecall(model, tok, filler: np.ndarray, eos_id: int, device, amp_dtype,
                   gaps: Sequence[int] = (256, 1024, 3072), n: int = 100, seed: int = 0, n_distract: int = 8,
                   ice_keep: float = 1.0, categories: Optional[Sequence[str]] = None) -> Dict[str, float]:
    res = {}
    for cat in categories or CATEGORIES:
        for gap in (gaps if cat not in ("increment", "track") else gaps[:1]):
            rng = np.random.default_rng([seed, CATEGORIES.index(cat), gap])    # same items for every arm
            seqs, spans, decide = build_items(tok, rng, cat, n, gap, filler, eos_id, n_distract)
            d = score_answers(model, seqs, spans, device, amp_dtype, ice_keep, details=True, decide=decide)
            for k in ("em", "dec_acc", "dec_acc_lookup_right", "dec_acc_lookup_wrong", "dec_acc_lookup_none",
                      "dec_lookup_right_frac", "dec_lookup_wrong_frac", "dec_follow_wrong_lookup"):
                res[f"coderecall/{cat}@{gap}/{k}"] = d[k]
            res[f"coderecall/{cat}@{gap}/items_dec"] = d["items_dec"]          # per item, for paired statistics
    return res


def summarize(res: Dict[str, float], metric: str = "dec_acc") -> Dict[str, float]:
    """Mean of a metric per category (over gaps)."""
    out = {}
    for cat in CATEGORIES:
        v = [x for k, x in res.items() if k.startswith(f"coderecall/{cat}@") and k.endswith(f"/{metric}")
             and not (isinstance(x, float) and np.isnan(x))]
        if v:
            out[cat] = float(np.mean(v))
    return out
