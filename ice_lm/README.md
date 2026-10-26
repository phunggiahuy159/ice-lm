# ice_lm — In-Context Engram for hybrid language models

Code for the proposal in `../proposal.pdf` (how ICE works, with formulas: `../ice_explained.pdf`).
What changed after the reviewers' feedback, and why: `../feedback_response.md`.

A hybrid language model (Gated DeltaNet layers + a few full-attention layers) gets an **attention-free exact
copy path**: a hash-table lookup over the current context finds where the last *n* tokens appeared before and
hands the model the token that came next. The model learns how much to trust it (a gate inside the network and a
copy bonus on the logits), helped by two trust signals: whether the two most recent earlier occurrences agreed on
the next token, and how often this lookup has been right so far in the document. The question: can ICE replace
some full-attention layers (and their KV cache), and where does an exact lookup stop helping?

## Layout

```
ice_lm/
  config.py        presets (smoke / 140m / 0.6b) and arms (A-F + ablations)
  ice_features.py  the exact n-gram lookup + trust signals (vectorized NumPy, runs in DataLoader workers)
  gdn.py           Gated DeltaNet heads; fla Triton kernel with automatic fallback to an exact PyTorch kernel
  model.py         hybrid mixer (GDN heads / attention heads / both), ICE input + output modules, the LM,
                   inference cache (KV / GDN state / conv state) for prefill + decode
  inference.py     incremental ICE index for decoding (same features as training) + its memory accounting
  data.py          dataset download + tokenization to .bin, packed windows with ICE features, ICE-dropout
  evals.py         loss splits (prose/code x copyable/other x lookup right/misleading/none), follow rate,
                   repeat / MQAR / phonebook, templated vs paraphrased needles
  coderecall.py    CodeRecall: identifiers / config values in code (copy, conflicts, overrides, copy-is-wrong)
  utils.py         DDP, ZeRO-1 optimizer, precision, LR schedule, checkpoints
scripts/
  prepare_data.py  python scripts/prepare_data.py --out data --train_tokens 4.2e8   (--synthetic for an offline toy corpus)
  day0_stats.py    the "Day 0" ceiling: how often a match exists and how often its successor is right
  train.py         train one arm (single GPU or torchrun); auto-resumes from out_dir/ckpt_latest.pt
  evaluate.py      evaluate several runs (+ ICE switched off) and print comparison tables with paired CIs
  benchmark.py     inference cost per arm: prefill / decode speed, KV + GDN state + ICE table memory
  plot_results.py  figures (frontier / CodeRecall / harm) from evaluate.py + benchmark.py output
  build_notebook.py  rebuild notebooks/ice_kaggle.ipynb from the source files
tests/test_ice_lm.py  correctness tests (run: python -m pytest -q tests)
notebooks/ice_kaggle.ipynb  self-contained Kaggle notebook (embeds the package, scripts and tests)
modal_app.py         optional: tests / data / training / evaluation on Modal (CPU checks need no GPU)
```

## Arms

| arm | attention | ICE |
|---|---|---|
| A | layer-level 3:1 | – |
| B | layer-level 7:1 | – |
| C | layer-level 7:1 | input + output + trust signals, dropout 0.2 |
| D | head-level hybrid, KV budget of B | – |
| E | head-level hybrid, KV budget of B | input + output + trust signals, dropout 0.2 |
| F | layer-level 15:1 | input + output + trust signals, dropout 0.2 |
| B_pm | layer-level 7:1 | – (wider MLP: as many parameters as C; capacity control) |
| C_in / C_out / C_nodrop | ablations of C | input only / output only / no dropout |
| C_notrust | ablation of C | the original ICE, without the trust signals |

Head-level arms (D, E) spread B's KV heads over more layers, one KV head (plus its query group) per layer next to
GDN heads. This is a from-scratch analogue of HydraHead's allocation, not HydraHead's head-selection procedure.

## Presets

| preset | d_model | layers | heads x dim | KV heads | MLP | params (32K vocab) | default recipe |
|---|---|---|---|---|---|---|---|
| smoke | 128 | 4 | 4 x 32 | 2 | 256 | ~1M | 2M tokens |
| **140m** (main scale) | 768 | 16 | 12 x 64 | 4 | 2048 | 142-150M | 400M tokens, batch 131K tokens (~3K steps), lr 1e-3, warmup 20M |
| 0.6b | 1280 | 24 | 10 x 128 | 2 | 3840 | 567-594M | 300M tokens, batch 262K, lr 6e-4, warmup 20M (sanity check only: ~0.5 tokens/param) |

Layouts and KV cache per token (bf16) for the two real presets (`G` = GDN layer, `A` = full attention,
`h` = GDN heads + one attention KV group):

| arm | 140m layout | KV B/token | 0.6b layout | KV B/token |
|---|---|---|---|---|
| A | `GGGAGGGAGGGAGGGA` | 4096 | `GGGA` x 6 | 6144 |
| B, C | `GGGGGGGA` x 2 | 2048 | `GGGGGGGA` x 3 | 3072 |
| D, E | `Gh` x 8 | 2048 | `GGGh` x 6 | 3072 |
| F | `GGGGGGGGGGGGGGGA` | 1024 | 2 attention layers (24 layers round 15:1 to 11:1) | 2048 |

ICE adds ~3.6M parameters (140m) / ~10M (0.6b), mostly the six projections W_n, and ~100 bytes per token
of token-ID memory at inference.

Any field can be overridden on the command line, e.g. `--total_tokens 3e8 --micro_batch 1 --lr 3e-4`.

## Quick start (any machine with a GPU; on Kaggle use the notebook)

```bash
pip install torch numpy transformers datasets   # optional, fast kernels: pip install fla-core==0.4.2 flash-linear-attention==0.4.2
python scripts/prepare_data.py --out data --train_tokens 4.2e8
python scripts/day0_stats.py --data_dir data
torchrun --nproc_per_node=2 scripts/train.py --preset 140m --arm C --data_dir data --out_dir runs/140m_C
python scripts/evaluate.py --runs runs/140m_A runs/140m_B runs/140m_C --data_dir data --val_batches 500
python scripts/benchmark.py --preset 140m --arms A B C F --contexts 4096 16384 32768 --data_dir data \
    --out benchmark_140m.json
python scripts/plot_results.py --results results.json --benchmark benchmark_140m.json --out figs
```

## What the evaluation reports

* **Held-out loss** for prose and code, split into copyable / other tokens, tokens whose exact match is more than
  512 tokens back (`copy_far`), and by what the lookup says about the next token: right (`lk_right`), misleading
  (`lk_wrong`), no match (`lk_none`). These groups depend only on the text, so they are the same tokens for every
  arm. `follow_wrong_lookup` = P(top-1 prediction = the wrong lookup token); `follow_excess` = minus arm B's rate.
* **ICE switched off** (`C_off`): the same model with keep = 0. If it is worse than B, the model became lazy.
* **CodeRecall** (`ice_lm/coderecall.py`): accuracy on the deciding digit of a recalled value, per category,
  split by the lookup's status, with paired bootstrap 95% CIs against B.
* **Inference cost** (`scripts/benchmark.py`): prefill and decode speed, measured KV / GDN-state bytes, ICE table
  bytes (compact layout), ICE CPU time per token, and how many 32K-token sequences fit in 16/24/80 GB.

Offline smoke test (CPU is fine):

```bash
python scripts/prepare_data.py --synthetic --out data_synth
python scripts/train.py --preset smoke --arm C --data_dir data_synth --out_dir runs/smoke_C
```

## Running on Modal (optional: e.g. a 0.6B confirmation run on H100)

`modal_app.py` runs everything on Modal with a persistent volume `ice-lm` for data, checkpoints and results.
GPU functions need a payment method on the Modal workspace; with `ICE_CPU_ONLY=1` the CPU entrypoints
(`tests_cpu`, `smoke_cpu`, `ddp_cpu`, `notebook_cpu`) work without one.

```bash
pip install modal
modal token set --token-id <id> --token-secret <secret>    # keep these out of files and git
cd ice_lm
modal run modal_app.py::tests                           # unit tests + which GDN kernel the GPU uses (T4, cents)
modal run modal_app.py::smoke                           # tiny B vs C on a toy corpus: C should win clearly
modal run modal_app.py::prepare --train-tokens 4.2e8    # CPU-only download + tokenize into the volume
modal run modal_app.py::day0                            # Day 0 statistics -> /vol/day0.json
modal run --detach modal_app.py::train --preset 0.6b --arm A --gpu H100:8
modal run --detach modal_app.py::train --preset 0.6b --arm B --gpu H100:8
modal run --detach modal_app.py::train --preset 0.6b --arm C --gpu H100:8
modal run modal_app.py::evaluate --runs 0.6b_A,0.6b_B,0.6b_C
modal run --detach modal_app.py::evaluate_when_done      # waits for 140m_B/C/C_notrust, then evaluates + plots
modal volume get ice-lm results_0.6b.json .
```

* **W&B logging** (optional): `modal secret create wandb WANDB_API_KEY=<key>` once; every training run then logs
  to the W&B project `ice-lm` (training loss, lr, grad norm, tokens/s, the periodic held-out splits). A resumed run
  continues the same W&B run (id in `<run>/wandb_id.txt`). Without the secret, use `ICE_WANDB=0 modal run ...`;
  outside Modal, set `WANDB_API_KEY` to enable it.

* `--detach` keeps a run going after the terminal closes; the same command resumes from the volume's checkpoint
  (Modal also retries a crashed container up to 2 times). Each call stops cleanly before Modal's 24 h limit;
  run it again to continue.
* Micro-batch and activation checkpointing are picked from GPU memory (80 GB: 0.6b micro-batch 4, no checkpointing).
* Measured on one H100 (140m, micro-batch 8, fla kernels): B 89K tok/s, C 80K tok/s, so **140m, 400M tokens** ≈
  1.3–1.4 H100-hours per arm. RTX 3090: B 22.8K, C 21.1K tok/s (micro-batch 2). 0.6b is roughly 4× slower per
  token. The training log prints the measured tokens/s and an ETA.

## Running on Kaggle

`notebooks/ice_kaggle.ipynb` is self-contained (it writes all source files with `%%writefile`). Upload it, set
*Accelerator* = GPU T4 x2 and *Internet* = on, edit the config cell, and run the cells in order. Kaggle's free
GPUs are slow for 0.6B (1–5K tokens/s), so the main experiments use the `140m` preset (1–2 sessions per arm);
0.6B is a sanity check. Rebuild the notebook after code changes: `python scripts/build_notebook.py`.

## Notes

* **GDN kernel.** `gdn_backend=auto` tests the flash-linear-attention kernel against the PyTorch reference on the
  actual GPU and dtype, and falls back to the PyTorch kernel if fla is missing, fails, or disagrees (e.g. on T4).
  The PyTorch kernel is exact, but slower.
* **Precision.** bf16 on Ampere and newer GPUs; fp16 with a gradient scaler on T4/P100; fp32 on CPU.
* **Same data for every arm.** Training windows are non-overlapping chunks visited in a seeded permutation (each
  once per pass); sample i always maps to the same window, and optimizer step s always uses samples
  [s*S, (s+1)*S) (S = global batch / seq_len), whatever the micro-batch, GPU count or DataLoader workers. A resumed
  run keeps its saved schedule settings; a finished run (model_final.pt) is never retrained by accident.
  Prepare at least as many tokens as you train on (`--train_tokens`), or windows repeat.
* **Data fingerprint.** `meta.json` records a fingerprint of the token files; `evaluate.py` warns if the runs
  being compared were trained on different data (e.g. a re-download in another session).
* **ICE features** are computed per window from token IDs only; documents in a packed window never match each other.
* **Decode path** (`allocate_cache` / `forward_cached`) is tested to give the same logits as the full forward pass
  for every arm, including chunked prefill and batches; the incremental ICE index gives exactly the training features.
* Not included yet: RepoBench/LCC code completion, free-running generation, and the retrieval-head formation
  analysis (proposal metrics iii and v).
