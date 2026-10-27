"""Evaluations from the proposal (Section 2.2, Metrics):
  (i)  held-out loss split by prose / code and by "copyable" tokens (longest match >= 4) vs the rest,
       and by what the lookup says about the next token: right / misleading / no match, with the
       "follow rate" P(argmax == s* | s* is wrong) -- how often the model is fooled by a wrong lookup
  (ii) exact-copy tasks: repeat-a-string, phonebook, MQAR (also beyond the training length)
  (iv) needle retrieval with TEMPLATED vs PARAPHRASED questions
All task scores are teacher-forced: answer-token accuracy and exact match (all answer tokens right).
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import torch

from .data import load_meta, model_ice_keys, open_tokens, split_batch, val_batches
from .ice_features import compute_ice_features


def _autocast(device, dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def uses_ice(model) -> bool:
    return model.uses_ice()


@torch.no_grad()
def eval_loss_splits(model, data_dir: str, seq_len: int, batch_size: int, n_batches: Optional[int],
                     device, amp_dtype, copy_min_len: int = 4, far: int = 512,
                     ice_keep: float = 1.0) -> Dict[str, float]:
    """Loss (and accuracy) per token group. Groups are defined by the exact lookup alone, so they are the same
    tokens for every arm (with or without ICE). ice_keep=0 evaluates an ICE model with ICE switched off."""
    model.eval()
    with_ice = uses_ice(model)
    res = {}
    for split, name in (("val_prose", "prose"), ("val_code", "code")):
        groups = ("all", "copyable", "other", "copy_far", "lk_right", "lk_wrong", "lk_none")
        tot = {g: [0.0, 0, 0] for g in groups}          # loss sum, count, correct
        follow = 0
        for batch in val_batches(data_dir, split, seq_len, batch_size, n_batches, model.cfg.ice_orders, with_ice,
                                 model.cfg.ice_v):
            x, y, ice, keep, mlen, st = split_batch(batch, device, with_stats=True)
            if keep is not None:
                keep = torch.full_like(keep, float(ice_keep))
            with _autocast(device, amp_dtype):
                logits, loss = model(x, y, ice=ice, ice_keep=keep, reduction="none")
            loss = loss.float()
            pred = logits.argmax(-1)
            s_star = st["lookup_s_star"]
            cp = mlen >= copy_min_len
            masks = dict(all=torch.ones_like(cp), copyable=cp, other=~cp, copy_far=cp & (st["ice_dist_star"] > far),
                         lk_right=s_star == y, lk_wrong=(s_star >= 0) & (s_star != y), lk_none=s_star < 0)
            for key, mask in masks.items():
                tot[key][0] += loss[mask].sum().item()
                tot[key][1] += int(mask.sum().item())
                tot[key][2] += int((pred == y)[mask].sum().item())
            follow += int((pred == s_star)[masks["lk_wrong"]].sum().item())
        for key, (s, n, c) in tot.items():
            res[f"{name}/{key}_loss"] = s / n if n else float("nan")      # nan: no such tokens (e.g. copy_far)
            res[f"{name}/{key}_acc"] = c / n if n else float("nan")
            res[f"{name}/{key}_frac"] = n / max(1, tot["all"][1])
        # how often the model's top prediction is the (wrong) lookup token
        res[f"{name}/follow_wrong_lookup"] = follow / tot["lk_wrong"][1] if tot["lk_wrong"][1] else float("nan")
    model.train()
    return res


# ---------------------------------------------------------------------------
# Synthetic tasks
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_answers(model, seqs: List[List[int]], answer_spans: List, device, amp_dtype, ice_keep: float = 1.0,
                  details: bool = False, decide: Optional[List[int]] = None):
    """seqs[i] = full token list; answer_spans[i] = (start, end) or a list of (start, end) spans of answer
    tokens in seqs[i]. Returns (token accuracy, exact-match rate over spans, mean answer log-prob), or with
    details=True a dict that also splits answer-token accuracy by what the exact lookup suggested for that token
    (right / wrong / none) and gives the rate of following a wrong lookup (argmax == the wrong s*).
    decide[i] (optional): position of the one token that decides item i (e.g. the first digit of a value: later
    digits can be copied once it is given). Adds dec_* metrics split by the lookup status of that token and the
    per-item 0/1 list items_dec (for paired comparisons between arms, which score identical items)."""
    model.eval()
    with_ice = uses_ice(model)
    acc_n = acc_d = em = n_spans = 0
    lp_sum = 0.0
    by = {s: [0, 0] for s in ("right", "wrong", "none")}   # answer tokens, correct
    dby = {s: [0, 0] for s in ("right", "wrong", "none")}  # deciding tokens, correct
    follow = dfollow = 0
    items_dec: List[int] = []
    for i_item, (seq, spans) in enumerate(zip(seqs, answer_spans)):
        spans = [spans] if isinstance(spans, tuple) else list(spans)
        x = torch.tensor(seq[:-1], device=device)[None]
        f = compute_ice_features(np.array(seq[:-1]), None, model.cfg.ice_orders, model.cfg.ice_v if with_ice else "")
        ice = ({k: torch.from_numpy(f[k]).to(device)[None] for k in model_ice_keys(model.cfg.ice_v)}
               if with_ice else None)
        keep = torch.full((1,), float(ice_keep), device=device)
        with _autocast(device, amp_dtype):
            logits = model(x, ice=ice, ice_keep=keep)
        logits = logits[0].float()
        for a, b in spans:
            tgt = torch.tensor(seq[a:b], device=device)
            rows = logits[a - 1: b - 1]                     # position t predicts token t+1
            pred = rows.argmax(-1)
            correct = pred == tgt
            acc_n += int(correct.sum())
            acc_d += len(tgt)
            em += int(correct.all())
            n_spans += 1
            lp_sum += torch.log_softmax(rows, -1).gather(-1, tgt[:, None]).mean().item()
            for i, t in enumerate(range(a, b)):
                s0 = int(f["ice_s_star"][t - 1])
                status = "none" if s0 < 0 else ("right" if s0 == seq[t] else "wrong")
                by[status][0] += 1
                by[status][1] += int(correct[i])
                follow += int(status == "wrong" and int(pred[i]) == s0)
        if decide is not None:
            t = decide[i_item]
            s0 = int(f["ice_s_star"][t - 1])
            status = "none" if s0 < 0 else ("right" if s0 == seq[t] else "wrong")
            p_t = int(logits[t - 1].argmax())
            ok = int(p_t == seq[t])
            dby[status][0] += 1
            dby[status][1] += ok
            dfollow += int(status == "wrong" and p_t == s0)
            items_dec.append(ok)
    model.train()
    acc, em_rate, lp = acc_n / max(1, acc_d), em / max(1, n_spans), lp_sum / max(1, n_spans)
    if not details:
        return acc, em_rate, lp
    out = dict(acc=acc, em=em_rate, logprob=lp, follow_wrong_lookup=follow / max(1, by["wrong"][0]))
    for st, (n_, c_) in by.items():
        out[f"lookup_{st}_frac"] = n_ / max(1, acc_d)
        out[f"acc_lookup_{st}"] = c_ / n_ if n_ else float("nan")
    if decide is not None:
        n_dec = max(1, len(items_dec))
        out["dec_acc"] = sum(items_dec) / n_dec
        out["items_dec"] = items_dec
        out["dec_follow_wrong_lookup"] = dfollow / dby["wrong"][0] if dby["wrong"][0] else float("nan")
        for st, (n_, c_) in dby.items():
            out[f"dec_lookup_{st}_frac"] = n_ / n_dec
            out[f"dec_acc_lookup_{st}"] = c_ / n_ if n_ else float("nan")
    return out


def task_repeat(rng, vocab_lo, vocab_hi, length, sep, prefix_len=16):
    s = rng.integers(vocab_lo, vocab_hi, size=length).tolist()
    pre = rng.integers(vocab_lo, vocab_hi, size=prefix_len).tolist()
    seq = pre + s + sep + s
    a = len(pre) + len(s) + len(sep) + 1         # skip the first copied token (not predictable)
    return seq, (a, len(seq))


def task_mqar(rng, vocab_lo, vocab_hi, n_pairs, n_queries):
    keys = rng.choice(np.arange(vocab_lo, vocab_hi), size=n_pairs, replace=False).tolist()
    vals = rng.integers(vocab_lo, vocab_hi, size=n_pairs).tolist()
    seq = []
    for k_, v_ in zip(keys, vals):
        seq += [k_, v_]
    q = rng.choice(n_pairs, size=n_queries, replace=False)
    spans = []
    for j in q:
        seq += [keys[j], vals[j]]
        spans.append(len(seq) - 1)
    return seq, spans


_NAMES = ["Alice", "Bruno", "Chen", "Dara", "Elif", "Farid", "Gita", "Hugo", "Ines", "Jonas", "Kenji", "Lena",
          "Mateo", "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wen", "Xavi",
          "Yara", "Zane"]
_SURN = ["Nguyen", "Smith", "Okafor", "Garcia", "Kowalski", "Tanaka", "Haddad", "Silva", "Novak", "Rossi",
         "Ivanova", "Kim", "Murphy", "Sato", "Dubois", "Moreau", "Larsen", "Costa", "Fischer", "Patel"]
_COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "silver", "golden"]
_OBJECTS = ["door", "box", "safe", "gate", "locker", "chest", "drawer", "vault", "cabinet", "case"]


def encode_with_answer(tok, prompt: str, answer: str):
    """Tokenize prompt+answer together; the answer span starts where the prompt's own tokenization stops
    matching (robust to tokenizers that merge a leading space into the answer)."""
    full = tok(prompt + answer, add_special_tokens=False)["input_ids"]
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = 0
    while a < min(len(p), len(full)) and p[a] == full[a]:
        a += 1
    a = min(a, len(p))
    while a < len(full) - 1 and not tok.decode([full[a]]).strip():      # a bare "▁" / space token is free
        a += 1
    return full, (a, len(full))


def build_text_tasks(tok, rng, n: int, context_len: int, filler: np.ndarray, eos_id: int):
    """Returns dict task_name -> (seqs, spans) for phonebook / needle-templated / needle-paraphrased."""
    out = {}
    # phonebook: fill the context with "Name Surname: 7 digits" lines, then ask for one entry
    seqs, spans = [], []
    line_tokens = max(8, len(tok("Alice Nguyen Kowalski: 1234567\n", add_special_tokens=False)["input_ids"]))
    n_lines = max(4, (context_len - 32) // line_tokens)
    for _ in range(n):
        book = {}
        while len(book) < n_lines:
            book[f"{rng.choice(_NAMES)} {rng.choice(_SURN)} {rng.choice(_SURN)}"] = \
                "".join(str(d) for d in rng.integers(0, 10, size=7))
        names = list(book)
        name = names[int(rng.integers(0, len(names)))]
        prompt = "Phone book:\n" + "".join(f"{k}: {v}\n" for k, v in book.items()) + f"\n{name}:"
        ids, span = encode_with_answer(tok, prompt, f" {book[name]}")
        seqs.append(ids)
        spans.append(span)
    out["phonebook"] = (seqs, spans)

    # needles: identical haystacks and needles (same seed), two question styles
    needle_seed = int(rng.integers(1 << 30))
    for style in ("templated", "paraphrased"):
        seqs, spans = [], []
        r2 = np.random.default_rng(needle_seed)
        for _ in range(n):
            color, obj = r2.choice(_COLORS), r2.choice(_OBJECTS)
            num = "".join(str(d) for d in r2.integers(1, 10, size=4))
            needle = tok(f" The secret code for the {color} {obj} is {num}. ", add_special_tokens=False)["input_ids"]
            if style == "templated":
                q = f"\nThe secret code for the {color} {obj} is"
            else:
                q = f"\nQuestion: which number opens the {obj} that is {color}? Answer: the number is"
            q_ids, (qa, qb) = encode_with_answer(tok, q, f" {num}")
            hay_len = max(16, context_len - len(needle) - len(q_ids))
            start = int(r2.integers(0, max(1, len(filler) - hay_len)))
            hay = [t for t in np.asarray(filler[start: start + hay_len]).tolist() if t != eos_id]
            depth = int(r2.integers(0, max(1, len(hay))))
            prefix = hay[:depth] + needle + hay[depth:]
            seqs.append(prefix + q_ids)
            spans.append((len(prefix) + qa, len(prefix) + qb))
        out[f"needle_{style}"] = (seqs, spans)
    return out


def run_synthetic_tasks(model, data_dir: str, device, amp_dtype, context_lens=(1024, 4096, 8192),
                        n: int = 50, seed: int = 0, ice_keep: float = 1.0) -> Dict[str, float]:
    meta = load_meta(data_dir)
    V = model.cfg.vocab_size
    lo, hi = max(10, V // 32), max(20, V - 100)
    rng = np.random.default_rng(seed)
    res = {}
    sep = [lo + 1, lo + 2, lo + 3]
    for L in context_lens:
        # repeat a random string of length ~L/2
        seqs, spans = zip(*[task_repeat(rng, lo, hi, max(8, L // 2 - 16), sep) for _ in range(n)])
        acc, em, lp = score_answers(model, list(seqs), list(spans), device, amp_dtype, ice_keep)
        res[f"repeat@{L}/acc"], res[f"repeat@{L}/em"] = acc, em
        # MQAR: single-token keys/values, n_pairs ~ L/4
        n_pairs = max(16, min((L - 32) // 2, hi - lo - 1))
        mq = [task_mqar(rng, lo, hi, n_pairs=n_pairs, n_queries=16) for _ in range(n)]
        acc, _, _ = score_answers(model, [s_ for s_, _ in mq], [[(p, p + 1) for p in pos] for _, pos in mq],
                                  device, amp_dtype, ice_keep)
        res[f"mqar@{L}/acc"] = acc
        # text tasks need the real tokenizer
        if meta.get("tokenizer"):
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(meta["tokenizer"])
            filler = open_tokens(data_dir, "val_prose")
            for name, (seqs, spans) in build_text_tasks(tok, rng, n, L, filler, meta["eos_id"]).items():
                acc, em, lp = score_answers(model, seqs, spans, device, amp_dtype, ice_keep)
                res[f"{name}@{L}/acc"], res[f"{name}@{L}/em"] = acc, em
    return res
