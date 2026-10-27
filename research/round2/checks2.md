# Checks 2: F1, F2 and the ICE-K probe (run 6), 27 Oct 2026

Scripts (new, in `ice_lm/scripts/`): `rule_oracle2.py` (F1, plus the helpers the other two import),
`token_scorer.py` (F2) and `icek_probe.py` (run 6).

**Compute note.** The Modal workspace hit its spend limit at 07:08, so nothing here ran on Modal.
- F1 and F2 ran on local CPU, on `val_code.bin` downloaded before the stop.
- `train.bin` is not available locally. F2's training set is therefore the val_code windows **after** the 300
  held-out windows (258 windows, 1.06M tokens, indices 300..557), so training and test windows are disjoint. The
  script's default (`--train_split train`) uses the training stream's code windows instead.
- The G_VR checkpoint is not local either, so **the ICE-K probe did not run**. It passed a CPU plumbing smoke test
  on a random tiny model. Its command and cost are in §3.

All numbers below use the anchor mask G_VR was trained with (`data.alpha_mask_for_data`).
- With `--mask coverage` (rule_oracle.py's mask, 150 windows), the old numbers come back exactly: s\* 0.508,
  longest 0.426, s\*-wrong oracle 0.333.
- With the training mask, the 300-window numbers are s\* 0.525, longest 0.444, s\*-wrong oracle 0.342.
- CodeRecall is identical under both masks.

## 1. F1: relational rules (`rule_oracle2.py`)
Only ambiguous positions count: the slots hold ≥ 2 distinct tokens. Held-out code is 300 val_code windows. CodeRecall
uses exactly run_coderecall's items at the deciding token, 100 per gap.

| set | n | s\* | longest | **longest_occ** | **newest** | newest_occ | newest_anc | vote | lenvote | oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| code, ambiguous | 1,087,034 | 0.525 | 0.444 | **0.530** | 0.109 | 0.382 | 0.036 | 0.502 | 0.491 | 0.673 |
| code, s\* wrong | 424,974 | 0 | 0.017 | 0.011 | 0.047 | 0.066 | 0.021 | 0.064 | 0.056 | 0.342 |
| copy_diff | 300 | 0.030 | 0.000 | 0.030 | 0.087 | 0.030 | **0.313** | 0.120 | 0.127 | 0.913 |
| conflict_g2 | 300 | 0.000 | 0.913 | **0.913** | 0.093 | 0.093 | 0.100 | 0.057 | 0.090 | 0.937 |
| conflict_g5 | 300 | 0.000 | 0.847 | **0.847** | 0.090 | 0.090 | 0.057 | 0.040 | 0.020 | 0.873 |
| override_diff | 300 | 0.000 | 0.000 | 0.000 | 0.027 | 0.027 | 0.093 | 0.183 | 0.003 | 0.997 |
| focus 4 pooled | 1,200 | 0.007 | 0.440 | 0.448 | 0.074 | 0.060 | 0.141 | 0.100 | 0.060 | 0.930 |

(copy_same 1.00, conflict_g1 1.00 and override_same 1.00 under longest_occ. increment and track are 0 for every rule.)

- **Longest over occurrence slots only** fixes the anchor artefact of the old `longest` rule.
  - On real code it now edges past s\*: 0.530 vs 0.525, +0.08 over the all-slot rule.
  - It keeps the 0.91 / 0.85 on conflict_g2/g5.
  - It is still worthless where s\* is wrong: 0.011.
- **"Latest write" is not a rule.** Newest distinct token scores 0.109 on code and 0.027 on override_diff.
  - The newest source is usually a filler or span token of a recent anchor.
  - Restricted to occurrence slots it is 0.38 on code. Restricted to anchors it is 0.31 on copy_diff and 0.09 on
    override_diff.
- **No fixed rule beats 0.07 on s\*-wrong code.** The ceiling is 0.342.

**Per-group slot precision vs Δ / n / r.** Valid slots at ambiguous held-out code positions; full tables are in the JSON.

| group | all | Δ ≤ −2 | Δ = 0 (tie) | Δ = +1 | Δ ≥ +3 | no competitor | n = 1 | n = 6 | r = 0 (newest) | r = 3 |
|---|---|---|---|---|---|---|---|---|---|---|
| last | 0.620 | 0.07–0.22 | 0.31 | 0.50 | 0.70–0.89 | 0.88 | 0.09 | 0.80 | 0.45 | 0.67 |
| second | 0.576 | 0.07–0.22 | 0.28 | 0.61 | 0.83–0.94 | 0.94 | 0.05 | 0.85 | 0.71 | 0.59 |
| chain | 0.396 | 0.16–0.36 | 0.19 | 0.71 | 0.91–0.98 | 0.92 | 0.04 | 0.79 | 0.65 | 0.35 |
| anchor | 0.111 | 0.01–0.02 | 0.04 | 0.12 | 0.19–0.43 | 0.50 | 0.01 | 0.66 | 0.10 | 0.13 |

- **On all ambiguous code**, Δ and n are strongly monotone in every group. Precision moves 0.07 → 0.89 with Δ and
  0.05 → 0.85 with n.
  - This is mostly the "consensus vs contested" split the existing per-slot L already partly sees.
  - r is non-monotone. Newest is best only for second and chain slots.
- **Where s\* is wrong, the features almost stop separating.**
  - Last slots with Δ < 0 reach 0.19–0.26. Δ ≥ 0 is 0 by construction: that slot is s\*.
  - Second slots: 0.15 at Δ = 0 and 0.18–0.25 at Δ < 0.
  - Chain slots: 0.12 at Δ = 0, and 0.48 only at Δ = +8 (3,110 slots).
  - Anchor slots: 0.02–0.06 everywhere.
  - n is flat to mildly rising (second 0.09 → 0.20 for n 1 → 5). The newest token gives second slots 0.23 vs 0.14.
  - These are the positions where V2's R/A/N would have to help, and the signal there is small.

## 2. F2: token-only listwise scorer (`token_scorer.py`)
**Not finished:** the agent stopped at a usage limit before writing F2. Kaggle job `fit_GVRbb_v2` (V2 with all features) plus the round-3 no-fit probe (`research/round3/results_cpu3.md`) answer the same question on the real model.

## 3. ICE-K probe, run 6 (`icek_probe.py`): NOT RUN (Modal budget)
**What it does.**
- G_VR's backbone and its trained ICEPointer are loaded frozen.
- Two scorers are refitted on identical batches with identical loss and steps, both initialised from G_VR's pointer:
  - **ctrl:** the pointer without keys.
  - **keys:** the pointer plus s_k += τ·(q_t·k_{i_k−1})/√16, with q = W_q·RMSNorm(h_t) and
    k = Q8(tanh(W_k·RMSNorm(h))).
  - Keys use int8 fake-quant (straight-through), W_q = 0 at init, and 24,577 parameters.
- Key causality: i_k − 1 ≤ t − 1 is asserted. Source positions come from a copy of `_candidates` that is checked to
  give bit-identical tok/len/dist.
- **Loss:** mixture NLL + 0.25 × listwise slot CE, over code positions of training-stream code windows (seed 7).
- **Held out:** 200 val_code windows, scored three ways: G_VR as is (ref), ctrl and keys. It reports:
  - paired NLL with a window bootstrap;
  - token-level slot rank-1 on ambiguous s\*-wrong positions;
  - CodeRecall focus-4 rank-1 and deciding-token accuracy, 50 per gap, never trained on.
- **Self-check:** the probe's ref mixture NLL must equal `model(x, y, reduction="none")` on window 0. This held
  exactly in the smoke test.
- **Go rule:** keys − ctrl ≥ +0.10 rank-1 on s\*-wrong code and ≥ 0.005 nats paired loss (`res["go"]`).

**Command, once the budget is back.** Read-only on /vol/runs; writes /vol/diag/icek_probe.json:
```
cd <frozen copy> && MODAL_PROFILE=run3 <scratchpad>/mvenv/bin/modal run modal_app.py::run --gpu --cmd \
 "python scripts/icek_probe.py --base_run /vol/runs/t200_G_VR --data_dir /vol/data --out /vol/diag/icek_probe.json \
  --tokens 1.2e7 --micro_batch 8 --val_windows 200 --n_cr 50 --num_workers 7"
```
**Expected cost:** one H100 for about 8–12 min (≈ $0.6–0.8).
- Training is 367 steps × 32K window tokens. The frozen forward plus chunked logits take about 0.15 s per step, and
  7 CPU workers build the features.
- Held-out takes about 2 min and CodeRecall about 3 min.
- It is within the ≤ $1.5 / 20 min cap. Sanity checks: `check_model_vs_probe` must be equal, and ref NLL ≈ G_VR's
  code loss (1.69).

**Prior from F1/F2 (not a result).** Token-only features give little on s\*-wrong code: see §2. The probe asks
whether h at i_k − 1 carries what they lack.
