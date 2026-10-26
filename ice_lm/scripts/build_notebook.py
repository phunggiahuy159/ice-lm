#!/usr/bin/env python
"""Build a self-contained Kaggle notebook (notebooks/ice_kaggle.ipynb).

Every source file of the codebase is embedded as a %%writefile cell, so the notebook needs nothing
except internet access (for pip + datasets). Re-run this script after editing the code:
    python scripts/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [
    "ice_lm/__init__.py", "ice_lm/config.py", "ice_lm/ice_features.py", "ice_lm/gdn.py", "ice_lm/model.py",
    "ice_lm/data.py", "ice_lm/evals.py", "ice_lm/coderecall.py", "ice_lm/inference.py", "ice_lm/utils.py",
    "scripts/prepare_data.py", "scripts/day0_stats.py", "scripts/train.py", "scripts/evaluate.py",
    "scripts/benchmark.py", "scripts/plot_results.py",
    "tests/test_ice_lm.py",
]

def _lines(src):
    lines = src.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(src)}


cells = []

cells.append(md("""# In-Context Engram (ICE) — hybrid GDN language models on Kaggle

Implements the proposal *"In-Context Engram: An Attention-Free Exact Copy Path for Hybrid Language Models
with Less KV Cache"*: Gated DeltaNet + full-attention hybrids, the ICE lookup, arms A–F, and the evaluations.

**Kaggle settings:** *Accelerator* → **GPU T4 ×2** (or any GPU), *Internet* → **On**,
*Persistence* → **Files only** (so checkpoints survive between sessions).

**Run order:** 1 Config → 2 Setup → 3 Write code → 4 Tests → 5 Data → 6 Day 0 → 7 Train → 8 Evaluate →
9 Inference cost → 10 Figures.

### Be realistic about compute
| preset | params | tokens/s on 2×T4 (rough) | 12 h session ≈ |
|---|---|---|---|
| `smoke` | 1M | — (seconds) | pipeline check |
| `140m` | ~150M | ~7–20K (fla or not) | ~0.3–0.9B tokens |
| `0.6b` | ~0.6B | ~1.5–5K | ~0.06–0.2B tokens |

Kaggle gives ~30 GPU-hours/week. **Recommended main scale: `140m`** (default recipe: 400M tokens, batch 131K
tokens, lr 1e-3; ~1–2 sessions per arm). `0.6b` (default 300M tokens) is a sanity check at ~0.5 tokens/param.
Runs continue across sessions with **automatic resume** (see the last section); the training log prints the
measured tokens/s and an ETA after the first steps.

**Fair comparison:** every arm with the same `PRESET`, `TOTAL_TOKENS`, `SEQ_LEN`, `SEED`, number of GPUs and
GPU type (the micro-batch is picked from GPU memory), so all arms see the same data in the same order.
"""))

cells.append(md("## 1. Config"))
cells.append(code("""# ---- edit me ----------------------------------------------------------------
PRESET       = "0.6b"        # "smoke" | "140m" (recommended main scale) | "0.6b" (sanity check, slow on T4)
ARM          = "C"           # A B C F, controls B_pm C_notrust (+ D E C_nodrop C_in C_out)
SEED         = 1234          # paired seeds: use the same seed list for every arm (e.g. 1234, 1235)
TOTAL_TOKENS = None          # None = preset default (140m: 400M, 0.6b: 300M); same value for every arm
TRAIN_DATA_TOKENS = 420_000_000   # tokens to download/tokenize: >= TOTAL_TOKENS so no window repeats (~2 B/token on disk)
MAX_HOURS    = 11.0          # stop + checkpoint before Kaggle's session limit
GRAD_CKPT    = "auto"        # "auto" = on for 16 GB GPUs; try False for 140m (~25% faster) if it fits in memory
SEQ_LEN      = 4096
WORK         = "/kaggle/working"            # outputs (persisted with 'Save Version')
DATA_DIR     = f"{WORK}/data"
RUN_DIR      = f"{WORK}/runs/{PRESET}_{ARM}" + ("" if SEED == 1234 else f"_s{SEED}")
# To resume from a previous session: attach that notebook's output as an input and set this to the
# folder that contains ckpt_latest.pt, e.g. "/kaggle/input/<previous-version>/runs/0.6b_C"
RESUME_FROM  = None
# --------------------------------------------------------------------------------
import os, json, subprocess, time
SESSION_START = time.time()     # MAX_HOURS counts from here (setup + data prep included)
KAGGLE_LIMIT_H = 12.0          # Kaggle ends a session after 12 h
def hours_left():               # until MAX_HOURS: training stops (and checkpoints) here
    return MAX_HOURS - (time.time() - SESSION_START) / 3600
def session_left():             # until Kaggle ends the session
    return KAGGLE_LIMIT_H - (time.time() - SESSION_START) / 3600
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
print(PRESET, ARM, SEED, TOTAL_TOKENS, RUN_DIR)"""))

cells.append(md("## 2. Setup (GPU check + optional fast kernels)\n"
                "`flash-linear-attention` gives fast GDN kernels on recent GPUs. If it does not install or does not "
                "work on this GPU (common on T4/P100), the code automatically falls back to its exact pure-PyTorch kernel."))
cells.append(code("""!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU"
import torch
N_GPU = torch.cuda.device_count()
print("torch", torch.__version__, "| GPUs:", N_GPU,
      [torch.cuda.get_device_name(i) for i in range(N_GPU)])
!pip install -q --no-deps fla-core==0.4.2 flash-linear-attention==0.4.2 2>&1 | tail -2 || true
!pip install -q "datasets>=2.19" "transformers>=4.40" 2>&1 | tail -2 || true"""))

cells.append(md("## 3. Write the codebase\nEach cell below writes one file of the `ice_lm` package."))
cells.append(code("!mkdir -p ice_lm/ice_lm ice_lm/scripts ice_lm/tests"))
for rel in FILES:
    with open(os.path.join(ROOT, rel)) as f:
        src = f.read()
    cells.append(code(f"%%writefile ice_lm/{rel}\n{src}"))

cells.append(md("## 4. Tests + GDN backend self-test\nUnit tests run on CPU in about a minute. "
                "The second command shows which GDN kernel will be used on this GPU."))
cells.append(code("""!cd ice_lm && python -m pip install -q pytest && python -m pytest -q tests 2>&1 | tail -5
# in a separate process, so this kernel keeps no GPU memory during training
!cd ice_lm && python -c "import torch; from ice_lm.utils import pick_precision; from ice_lm.gdn import resolve_backend; d=torch.device('cuda'); amp,_=pick_precision('auto', d); print('amp dtype:', amp, '| GDN backend:', resolve_backend('auto', d, amp, 128)) if torch.cuda.is_available() else print('no GPU')\""""))

cells.append(md("### 4b. Smoke run (optional, ~2 min)\nTrains tiny arms B and C on an offline copy-heavy toy corpus. "
                "C (with ICE) should reach a clearly lower loss than B."))
cells.append(code("""!cd ice_lm && python scripts/prepare_data.py --synthetic --out {WORK}/data_synth
for arm in ["B", "C"]:
    smoke_cmd = (f"cd ice_lm && python scripts/train.py --preset smoke --arm {arm} --data_dir {WORK}/data_synth "
                 f"--out_dir {WORK}/smoke_runs/{arm} --total_tokens 600000 --eval_every 100 --log_every 50 --resume false")
    print("=== arm", arm); get_ipython().system(smoke_cmd + " 2>&1 | grep -E 'eval|finished|params'")
"""))

cells.append(md("## 5. Data\nStreams FineWeb-Edu (prose) and codeparrot-clean (Python), 70/30 by tokens, with the "
                "TinyLlama (Llama, 32K) tokenizer. Skipped if the data already exists."))
cells.append(code("""if not os.path.exists(f"{DATA_DIR}/meta.json"):
    !cd ice_lm && python scripts/prepare_data.py --out {DATA_DIR} --train_tokens {TRAIN_DATA_TOKENS}
print(json.load(open(f"{DATA_DIR}/meta.json")))"""))

cells.append(md("## 6. Day 0 — the ceiling of ICE (no model)\nCoverage = how often an earlier match exists; "
                "precision = how often its next token is right. If long matches are rare and imprecise, stop here."))
cells.append(code("!cd ice_lm && python scripts/day0_stats.py --data_dir {DATA_DIR} --seq_len {SEQ_LEN} --windows 200 --out {WORK}/day0.json"))

cells.append(md("## 7. Train one arm\nUses all GPUs via `torchrun`. Checkpoints every 30 min and at `MAX_HOURS`; "
                "re-running this cell (or a new session with `RESUME_FROM`) continues from the checkpoint.\n\n"
                "Memory: with 16 GB T4s, the `0.6b` preset uses micro-batch 1 + activation checkpointing, a chunked loss "
                "and Adam states sharded over the GPUs (ZeRO-1); `140m` uses micro-batch 2. Resume with the same "
                "number of GPUs to keep the optimizer state."))
cells.append(code("""import glob, shutil
os.makedirs(RUN_DIR, exist_ok=True)
if RESUME_FROM and not os.path.exists(f"{RUN_DIR}/ckpt_latest.pt"):
    # model checkpoint + per-GPU optimizer shards (ckpt_latest.pt.optim<rank>of<world>) + log
    for f in glob.glob(f"{RESUME_FROM}/ckpt_latest.pt*") + glob.glob(f"{RESUME_FROM}/log.jsonl") \
            + glob.glob(f"{RESUME_FROM}/config.json") + glob.glob(f"{RESUME_FROM}/model_final.pt"):
        shutil.copy(f, RUN_DIR)
    print("copied checkpoint from", RESUME_FROM, os.listdir(RUN_DIR))
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
micro = {"0.6b": 1 if gpu_mem < 40 else 4, "140m": 2 if gpu_mem < 40 else 8, "smoke": 8}[PRESET]
ckpt = str(GRAD_CKPT).lower() if GRAD_CKPT != "auto" else ("true" if (PRESET == "0.6b" or gpu_mem < 40) else "false")
launcher = f"torchrun --standalone --nproc_per_node={max(1, N_GPU)}" if N_GPU > 1 else "python"
cmd = (f"cd ice_lm && {launcher} scripts/train.py --preset {PRESET} --arm {ARM} --data_dir {DATA_DIR} "
       f"--out_dir {RUN_DIR} --seed {SEED} --seq_len {SEQ_LEN} --micro_batch {micro} "
       + (f"--total_tokens {TOTAL_TOKENS} " if TOTAL_TOKENS else "") +
       f"--grad_checkpointing {ckpt} --max_hours {hours_left():.2f} --num_workers 3")
print(cmd)
!{cmd}"""))

cells.append(md("## 8. Evaluate\n"
                "* held-out loss split by prose/code × copyable/other, and by what the exact lookup says about the "
                "next token (right / misleading / none) + how often the model follows a misleading lookup\n"
                "* **CodeRecall** (identifiers and config values in code): copy with the same / different syntax, "
                "conflicting keys, later overrides, and cases where copying is wrong (`increment`)\n"
                "* repeat / MQAR / phonebook / needles (templated vs paraphrased)\n"
                "* every ICE arm is also scored with ICE switched off (`C_off`): it should not be worse than B."))
cells.append(code("""import glob
# every run of this preset (all arms and seeds); add folders from attached inputs to compare runs from other sessions
runs = sorted(d for d in glob.glob(f"{WORK}/runs/{PRESET}_*")
              if os.path.exists(f"{d}/model_final.pt") or os.path.exists(f"{d}/ckpt_latest.pt"))
print(runs)
if not runs:
    print("no trained checkpoint yet - nothing to evaluate")
elif session_left() < 1.0:      # the full evaluation takes ~0.5-1 h per arm on a T4
    print(f"only {session_left():.1f} h left in this session: run the evaluation in a separate session")
else:
    !cd ice_lm && python scripts/evaluate.py --runs {" ".join(runs)} --data_dir {DATA_DIR} --n_task 30 --val_batches 500 --out {WORK}/results_{PRESET}.json"""))

cells.append(md("## 9. Inference cost: speed and memory, including the ICE lookup\n"
                "Prefill and decode speed, KV cache, GDN state and ICE table bytes per arm, plus the memory model "
                "`M(batch, T) = W + batch·(fixed + per_token·T)` and how many 32K-token sequences fit in 16/24/80 GB. "
                "Random weights are fine for speed/memory; pass `--runs` to use trained ones."))
cells.append(code("""if session_left() < 0.5:
    print("not enough session time left for the benchmark; run it in a separate session")
else:
    !cd ice_lm && python scripts/benchmark.py --preset {PRESET} --arms A B C F --contexts 4096 16384 32768 --decode_steps 64 --data_dir {DATA_DIR} --out {WORK}/benchmark_{PRESET}.json"""))

cells.append(md("## 10. Figures for the presentation\nFrontier (code loss vs cache bytes/token), CodeRecall per category "
                "(ordered from *lookup right* to *lookup wrong*), and misleading-lookup harm. Seeds of the same arm are pooled."))
cells.append(code("""from IPython.display import Image, display
if os.path.exists(f"{WORK}/results_{PRESET}.json"):
    bench = f"--benchmark {WORK}/benchmark_{PRESET}.json" if os.path.exists(f"{WORK}/benchmark_{PRESET}.json") else ""
    !cd ice_lm && python scripts/plot_results.py --results {WORK}/results_{PRESET}.json {bench} --out {WORK}/figs
    for name in ("frontier", "coderecall", "harm"):
        if os.path.exists(f"{WORK}/figs/{name}.png"):
            display(Image(f"{WORK}/figs/{name}.png"))
else:
    print("run the evaluation first")"""))


cells.append(md("""## Continuing across Kaggle sessions
Training stops cleanly at `MAX_HOURS` and saves `ckpt_latest.pt`. Two ways to continue (Kaggle's menu names may
differ slightly):

* **Interactive, with persistence:** in the session settings set *Persistence* → *Files only*. `/kaggle/working`
  (data, checkpoints, logs) is then kept between sessions; just run cells 1–3 and 7 again.
* **Unattended (recommended for long runs):** *Save Version* → *Save & Run All*. Kaggle runs the whole notebook in
  the background (up to its time limit) and stores `/kaggle/working` as the version's output. For the next run:
  *Add Input* → that notebook's output, set `RESUME_FROM = "/kaggle/input/<output>/runs/<PRESET>_<ARM>"` and
  point `DATA_DIR` at the attached `data` folder (so the data is not downloaded again), then save another version.

Either way, training continues from the last checkpoint with the same learning-rate schedule and data order.

**Fair comparison:** run every arm with the same `PRESET`, `TOTAL_TOKENS`, `SEQ_LEN`, `SEED`, number of GPUs
and GPU type, so they see the same data in the same order."""))

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}}}
os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
out = os.path.join(ROOT, "notebooks", "ice_kaggle.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out} ({len(cells)} cells)")
