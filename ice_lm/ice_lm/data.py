"""Data: tokenize a prose + code mixture into flat uint16/uint32 token files, then serve packed
windows with ICE features computed on the fly (CPU, in DataLoader workers).

Files written by `prepare_data` into data_dir:
    train.bin, val_prose.bin, val_code.bin, meta.json
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .ice_features import V2_KEYS, V_KEYS, compute_ice_features, get_alpha_mask, segments_from_eos, set_alpha_mask

PROSE_DEFAULT = dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", field="text")
CODE_DEFAULT = dict(path="codeparrot/codeparrot-clean", name=None, field="content")
TOKENIZER_DEFAULT = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # Llama tokenizer, 32K vocab, ungated


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------
class _Writer:
    def __init__(self, path: str, dtype):
        self.f = open(path, "wb")
        self.dtype = dtype
        self.n = 0

    def write(self, ids):
        a = np.asarray(ids, dtype=self.dtype)
        self.f.write(a.tobytes())
        self.n += len(a)

    def close(self):
        self.f.close()


def _doc_stream(spec: dict, seed: int) -> Iterator[str]:
    from datasets import load_dataset
    kw = dict(split="train", streaming=True)
    ds = load_dataset(spec["path"], spec.get("name"), **kw)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for ex in ds:
        txt = ex.get(spec["field"])
        if txt:
            yield txt


def prepare_data(out_dir: str, train_tokens: int, val_tokens: int = 2_000_000, code_frac: float = 0.3,
                 tokenizer_name: str = TOKENIZER_DEFAULT, prose: Optional[dict] = None,
                 code: Optional[dict] = None, seed: int = 0, batch_docs: int = 256,
                 max_doc_chars: int = 200_000) -> dict:
    """Stream + tokenize. Validation sets (prose and code separately) are taken first."""
    from transformers import AutoTokenizer
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    eos = tok.eos_token_id
    vocab = len(tok)
    dtype = np.uint16 if vocab <= 65535 else np.uint32
    prose = prose or PROSE_DEFAULT
    code = code or CODE_DEFAULT
    streams = {"prose": _doc_stream(prose, seed), "code": _doc_stream(code, seed)}

    def batches(name):
        buf = []
        for txt in streams[name]:
            buf.append(txt[:max_doc_chars])
            if len(buf) == batch_docs:
                yield buf
                buf = []
        if buf:
            yield buf

    gens = {k: batches(k) for k in streams}

    def tokenize(texts):
        out = tok(texts, add_special_tokens=False)["input_ids"]
        return [ids + [eos] for ids in out]

    t0 = time.time()
    # validation sets
    for name in ("prose", "code"):
        w = _Writer(os.path.join(out_dir, f"val_{name}.bin"), dtype)
        while w.n < val_tokens:
            docs = next(gens[name], None)
            if docs is None:
                raise RuntimeError(f"{name} stream ended before the validation set was complete")
            for ids in tokenize(docs):
                w.write(ids)
        w.close()
        print(f"[data] val_{name}: {w.n:,} tokens")
    # training mixture, balanced by token counts
    w = _Writer(os.path.join(out_dir, "train.bin"), dtype)
    counts = {"prose": 0, "code": 0}
    exhausted = set()
    last = 0
    while w.n < train_tokens:
        name = "code" if counts["code"] < code_frac * max(1, w.n) else "prose"
        if name in exhausted:
            name = "prose" if name == "code" else "code"
        raw = next(gens[name], None)
        if raw is None:
            exhausted.add(name)
            print(f"[data] {name} stream exhausted; continuing with the other one (mix will drift)")
            if len(exhausted) == 2:
                print(f"[data] both streams exhausted at {w.n:,} tokens")
                break
            continue
        docs = tokenize(raw)
        for ids in docs:
            w.write(ids)
            counts[name] += len(ids)
        if w.n - last > 50_000_000:
            last = w.n
            el = time.time() - t0
            print(f"[data] train {w.n/1e6:,.0f}M / {train_tokens/1e6:,.0f}M tokens "
                  f"({w.n/el/1e6:.2f}M tok/s, code {counts['code']/max(1,w.n):.0%})", flush=True)
    w.close()
    meta = dict(tokenizer=tokenizer_name, vocab_size=vocab, eos_id=eos, dtype=np.dtype(dtype).name,
                train_tokens=w.n, code_frac_actual=counts["code"] / max(1, w.n), prose=prose, code=code,
                fingerprint=data_fingerprint(out_dir, dtype))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[data] done in {time.time()-t0:.0f}s: {meta}")
    return meta


def make_synthetic_data(out_dir: str, train_tokens: int = 3_000_000, val_tokens: int = 200_000,
                        vocab_size: int = 512, seed: int = 0) -> dict:
    """Offline copy-heavy toy corpus (for smoke tests): documents that re-use earlier spans.
    'code' docs repeat spans often; 'prose' docs rarely. ICE should help on this data."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    eos = vocab_size - 1

    def doc(copy_prob):
        n = int(rng.integers(200, 1200))
        out = []
        while len(out) < n:
            if len(out) > 20 and rng.random() < copy_prob:
                L = int(rng.integers(8, 40))
                s = int(rng.integers(0, max(1, len(out) - L)))
                out.extend(out[s:s + L])
            else:
                # locally predictable "words": small Markov-ish chunks
                out.extend(rng.integers(1, vocab_size - 1, size=int(rng.integers(3, 10))).tolist())
        return out[:n] + [eos]

    def write(path, n_tok, copy_prob_fn):
        w = _Writer(path, np.uint16)
        while w.n < n_tok:
            w.write(doc(copy_prob_fn()))
        w.close()
        return w.n

    write(os.path.join(out_dir, "val_prose.bin"), val_tokens, lambda: 0.05)
    write(os.path.join(out_dir, "val_code.bin"), val_tokens, lambda: 0.5)
    n = write(os.path.join(out_dir, "train.bin"), train_tokens, lambda: 0.5 if rng.random() < 0.3 else 0.05)
    meta = dict(tokenizer=None, vocab_size=vocab_size, eos_id=eos, dtype="uint16", train_tokens=n, synthetic=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def data_fingerprint(data_dir: str, dtype) -> str:
    """Cheap content fingerprint of the token files: runs (sessions) that re-downloaded the data can check that
    they trained on exactly the same tokens (streaming datasets are not pinned to a revision)."""
    parts = []
    for split in ("train", "val_prose", "val_code"):
        x = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=dtype, mode="r")
        n = min(len(x), 1 << 20)
        idx = np.arange(n, dtype=np.int64) * max(0, len(x) - 1) // max(1, n - 1)
        parts.append(f"{len(x)}:{int(np.asarray(x[idx], dtype=np.int64).sum() % (1 << 61))}")
    return "|".join(parts)


def load_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def open_tokens(data_dir: str, split: str) -> np.memmap:
    meta = load_meta(data_dir)
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.dtype(meta["dtype"]), mode="r")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
ICE_KEYS = ("ice_succ", "ice_len", "ice_dist", "ice_agree", "ice_rel",
            "ice_s_star", "ice_len_star", "ice_agree_star", "ice_rel_star")
EVAL_KEYS = ("ice_match_len", "ice_dist_star", "lookup_s_star")


def model_ice_keys(cand: str = "", v2: bool = False) -> tuple:
    """Feature keys an ICE model reads: ICE_KEYS, + the candidate slots V_KEYS for ICE-V / VR (cfg.ice_v), + the
    ICE-V2 slot features V2_KEYS (cfg.ice_v2)."""
    return ICE_KEYS + (V_KEYS + (V2_KEYS if v2 else ()) if cand else ())


def alpha_mask_from_tokenizer(tok, vocab_size: int) -> np.ndarray:
    """ICE-VR anchors: tokens whose text (without the word-start marker) is purely alphabetic, e.g. `port`,
    `timeout`, `_`-free subwords of identifiers (a vocabulary property, not a frequency threshold)."""
    mask = np.zeros(vocab_size, dtype=bool)
    for i, piece in enumerate(tok.convert_ids_to_tokens(list(range(min(vocab_size, len(tok)))))):
        piece = (piece or "").lstrip("\u2581\u0120")              # sentencepiece / byte-level BPE word start
        mask[i] = piece.isalpha()
    return mask


def alpha_mask_for_data(data_dir: str) -> np.ndarray:
    """The anchor mask of a prepared data folder (its tokenizer); a synthetic corpus has no text, so every token
    but EOS counts as alphabetic there."""
    meta = load_meta(data_dir)
    if not meta.get("tokenizer"):
        mask = np.ones(meta["vocab_size"], dtype=bool)
        mask[meta["eos_id"]] = False
        return mask
    from transformers import AutoTokenizer
    return alpha_mask_from_tokenizer(AutoTokenizer.from_pretrained(meta["tokenizer"]), meta["vocab_size"])


def make_example(window: np.ndarray, eos_id: Optional[int], orders, with_ice: bool,
                 cand: str = "", v2: bool = False) -> Dict[str, torch.Tensor]:
    """window: seq_len + 1 tokens -> input_ids, labels (+ ICE features for the inputs; cand: ICE-V / VR slots,
    v2: + the ICE-V2 slot features)."""
    w = window.astype(np.int64)
    x, y = w[:-1], w[1:]
    ex = dict(input_ids=torch.from_numpy(x), labels=torch.from_numpy(y))
    feats = compute_ice_features(x, segments_from_eos(x, eos_id), orders, cand if with_ice else "", v2)
    feats["lookup_s_star"] = feats["ice_s_star"]
    # lookup statistics are always returned (eval splits: copyable / lookup right / lookup misleading)
    for k in EVAL_KEYS:
        ex[k] = torch.from_numpy(feats[k])
    if with_ice:
        for k in model_ice_keys(cand, v2):
            ex[k] = torch.from_numpy(feats[k])
    return ex


class TrainWindows(IterableDataset):
    """Infinite stream of training windows (each non-overlapping window once per pass over the data).

    Samples are indexed globally: micro-batch m (counted over the whole run) of rank r holds the sample indices
    (m * world + r) * micro_batch + j, j < micro_batch. DataLoader worker w of nw produces this rank's local
    micro-batches w, w + nw, ... in order. Optimizer step s therefore always uses the sample indices
    [s*S, (s+1)*S), S = micro_batch * grad_accum * world: the data per step depends only on the seed and the global
    batch, not on micro-batch, GPU count or num_workers, and a resume at start_index = samples consumed continues
    exactly where the run stopped."""

    def __init__(self, data_dir: str, seq_len: int, orders, with_ice: bool, ice_dropout: float,
                 seed: int, rank: int = 0, world: int = 1, start_index: int = 0, micro_batch: int = 1,
                 cand: str = "", v2: bool = False):
        self.data_dir, self.seq_len, self.orders = data_dir, seq_len, tuple(orders)
        self.with_ice, self.ice_dropout, self.cand, self.v2 = with_ice, ice_dropout, cand, v2
        # ICE-VR: the anchor mask must be set in this process; carried to the DataLoader workers explicitly
        self.alpha = get_alpha_mask() if (with_ice and cand == "vr") else None
        self.seed, self.rank, self.world, self.start_index = seed, rank, world, start_index
        self.micro_batch = micro_batch
        self.eos = load_meta(data_dir)["eos_id"]

    def sample(self, data, i: int, cache: dict):
        """Sample index i -> window perm_e[i mod n_win] of pass e = i // n_win: every window once per pass, a new
        order (and shifted window boundaries) in later passes. The same index gives the same window in every arm."""
        L = self.seq_len
        n_win = max(1, (len(data) - 1) // L - 1)            # non-overlapping windows (one spare for the shift)
        epoch, k = divmod(i, n_win)
        if cache.get("epoch") != epoch:
            prng = np.random.default_rng([self.seed, 7919, epoch])
            cache.update(epoch=epoch, perm=prng.permutation(n_win),
                         shift=0 if epoch == 0 else int(prng.integers(0, L)))
        off = int(cache["perm"][k]) * L + cache["shift"]
        ex = make_example(np.asarray(data[off: off + L + 1]), self.eos, self.orders, self.with_ice, self.cand,
                          self.v2)
        ex["ice_keep"] = torch.tensor(float(np.random.default_rng([self.seed, i]).random() >= self.ice_dropout))
        return ex

    def __iter__(self):
        if self.alpha is not None:
            set_alpha_mask(self.alpha)
        data = open_tokens(self.data_dir, "train")
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        mb, cache = self.micro_batch, {}
        m = self.start_index // (mb * self.world) + wid       # this worker's first global micro-batch
        while True:
            base = (m * self.world + self.rank) * mb
            for j in range(mb):
                yield self.sample(data, base + j, cache)
            m += nw


def val_batches(data_dir: str, split: str, seq_len: int, batch_size: int, n_batches: Optional[int],
                orders, with_ice: bool, cand: str = "", v2: bool = False):
    """Sequential non-overlapping windows from a validation file."""
    data = open_tokens(data_dir, split)
    eos = load_meta(data_dir)["eos_id"]
    n_windows = (len(data) - 1) // seq_len
    if n_batches is not None:
        n_windows = min(n_windows, n_batches * batch_size)
    batch = []
    for j in range(n_windows):
        w = np.asarray(data[j * seq_len: j * seq_len + seq_len + 1])
        ex = make_example(w, eos, orders, with_ice, cand, v2)
        ex["ice_keep"] = torch.tensor(1.0)
        batch.append(ex)
        if len(batch) == batch_size:
            yield collate(batch)
            batch = []
    if batch:
        yield collate(batch)


def collate(examples):
    out = {}
    for k in examples[0]:
        out[k] = torch.stack([e[k] for e in examples])
    return out




def split_batch(batch: Dict[str, torch.Tensor], device, with_stats: bool = False):
    """-> input_ids, labels, ice dict (or None), ice_keep, match_len (all on device)
    (+ dict of lookup statistics for the eval splits when with_stats)."""
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    ice = {k: b[k] for k in ICE_KEYS + V_KEYS + V2_KEYS if k in b} if "ice_succ" in b else None
    out = (b["input_ids"], b["labels"], ice, b.get("ice_keep"), b.get("ice_match_len"))
    return out + ({k: b[k] for k in EVAL_KEYS if k in b},) if with_stats else out
