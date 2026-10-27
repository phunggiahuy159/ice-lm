# ICE project handoff (26 Oct 2026, ~22:40 UTC+7)

Repo: https://github.com/huypg-zenai/ice-lm (private, branch `main`). Local copy: `/Users/phunghuy/code/gstar/proposal_ice`.
Runs, models and data live on Modal (see "Infrastructure"); this repo holds code, docs, logs and small result files.

## 0. Round-2 status (27 Oct, 07:50 UTC+7): blocked on the Modal spend limit

* **Done in round 2:**
  * the judge's plan, `research/round2/decision2.md`;
  * seed 2: G_VR beats B on 2 seeds (code −0.040 / −0.045), so seed 3 is skipped (`results_t200/README.md`);
  * the rule oracle: "longest match wins" fails on real code;
  * the G_MQA arm added to `config.py` (15 GDN + 1 MQA layer, 256 B/token);
  * `scripts/fit_scorer.py`.
* **At 07:08 workspace tien-phat140205 hit its billing-cycle spend limit.** Every new call now fails with
  `ResourceExhaustedError: workspace billing cycle spend limit reached`. The limit killed both running jobs:
  * the fitter gate: the refit finished and is saved (`/vol/runs/fit_GVRbb_vr/model_final.pt`), but its evaluation
    did not run;
  * G_MQA: killed at step 40 of 1526 (~$0.15 lost).

  Logs: `research/round2/logs/`. Nothing else was run, and no other workspace was used.
* **To resume once the budget is raised (profile run3):**
  1. Evaluate the gate:
     `spawn.py _gpu_cmd '["python scripts/evaluate.py --runs /vol/runs/fit_GVRbb_vr --data_dir /vol/data --val_batches 500 --n_task 50 --out /vol/results_fit_gate.json"]'`.
     It must reproduce 1.690 ± 0.005.
  2. Retrain G_MQA from scratch: `spawn.py _train '["140m","G_MQA",2e8,1,"","t200_G_MQA"]'`.
  3. Run the ICE-V2 ladder (runs 2–5 and 8 of decision2.md) after the implementer's tests pass on Modal CPU.
     Notes: `research/round2/impl_v2_notes.md`.
  4. Remaining budget for this round: ≈ $29.

### Round 3 (27 Oct, ~13:00): what improved, what is still weak
**Improved (all on CPU or free; Modal is blocked by its spend limit until 1 Nov):**
- **Honest cost:** the measured H100 benchmark is in the repo (`results_t200/benchmark_t200.*`).
  - The compact index was built for real: 82–90 B/token (`research/round3/index_memory.json`).
  - Memory per sequence: 2.8× less than B at 4K, where quality is measured; 11× at 32K.
- **The insight:** 88% of what attention adds to a recurrent model sits on tokens an exact lookup already predicts
  (45% null).
- **Conflicts understood:** where the lookup's best guess is wrong, the backbone's own probability ranks the right
  candidate first 0.645 vs the verifier's 0.406 (fresh windows, pre-registered). The fix is a context-dependent mix:
  V2, run 2 on Kaggle.
- **Realistic code use case** (`scripts/stale_edit.py`): with an old version of a file in context, ICE predicts the
  unchanged code much better than B (0.39 vs 0.89), but copies the old name at changed places 65% vs 13%.
- **Synthetic recall artefact found:** B's 0 was rare tokens; on real text B repeats about 60% vs G_VR 97%.
- **Fit leakage fixed:** `fit_scorer --unseen`.
- **WSD + decay branches** in `train.py`, for the 3× longer runs. The cosine default is unchanged, and a branch
  matches a standalone run bit for bit.
- **Slides/notes** follow decision3: every claim is supported, and the shrinking lead is disclosed.

**Still weak / open:**
1. The G_VR − B lead shrinks with training (−0.117 → −0.039). The 600M WSD runs decide it (decision3 M2, about $23
   on Modal after 1 Nov).
2. There is no equal-memory attention baseline yet (G_MQA, M1, about $3.5).
3. Conflicts and stale values are unsolved: ICE copies the old occurrence.
4. The frozen-backbone fits (over-reliance answer), V2, line completion and the T4 re-evaluation are waiting for
   the user's Kaggle run (`ice_lm/notebooks/ice_eval_kaggle.ipynb`, decision3 §3).
5. Only 140M, 200M tokens, one tokenizer. The real pretrained model (Mamba2 vs Pythia) moves to the next round.

**Next session:** fetch `kaggle_results/latest` from the Modal volume (or the zip the user downloads) and check
SUMMARY.md against decision3 §3. After 1 Nov, run M1 + M2 on Modal (profile run3; `train.py --schedule wsd
--branch_tokens 2e8,4e8`, then branch runs with `--branch_from`).

### Update (27 Oct, ~10:30)
* **ICE-V2 is implemented** (commit 34de81d): 169 tests pass on local CPU, and the old arms are bit-identical.
  Notes: `research/round2/impl_v2_notes.md`.
* **Kaggle notebook `ice_lm/notebooks/ice_eval_kaggle.ipynb`:** the user runs it on 2×T4. It holds every blocked
  GPU job: the re-evaluation of B/G/G_VR, the fitter gate, `recall_check.py`, `line_completion.py` and the ICE-V2
  ladder, followed by a SUMMARY.md with the gate and decision2 prediction checks.
  - The code is embedded from a commit.
  - The inputs come from the Modal volume, via Kaggle secrets, or from an attached dataset.
  - Results are uploaded to `kaggle_results/latest` on the volume.
  - It was dry-run end to end on CPU with tiny models.
  - Rebuild it with `python ice_lm/notebooks/build_eval_notebook.py` after committing code.
* **Modal downloads of large files can time out,** and a re-download over a partial file left a right-sized but
  corrupt checkpoint. Delete the partial file before retrying, and check `zipfile.ZipFile(p).testzip()`. The
  notebook does both.
* **The synthetic-recall "B = 0" is mostly a rare-token artefact:** see the results README, last section.
* **Slides:** `slides/ice_gstar.html` + `slides/talk_notes.md`.
* **Round 3:** advisors are writing `research/round3/proposal_{conflicts,evidence,directions}.md`.

## 1. TL;DR

* **ICE** = an exact in-context n-gram lookup on token IDs (CPU hash table, ~41 B/token, vs 2,048 B/token KV for
  the 7:1 hybrid B), injected into a Gated DeltaNet (GDN) / attention hybrid LM.
* **At 7:1, ICE's gain vanishes with training.** B's attention learns to copy by itself: code loss gap C−B went
  0.27 → 0.01 over 360M tokens. The trust signals (agree / rel) did nothing (C ≈ C_notrust).
* **New method, ICE-V**: the lookup proposes candidates and a backbone that never sees the lookup decides, through a
  log-space pointer-sentinel mixture. **ICE-VR** adds richer candidates. Both are tested in a model with **no
  attention at all** (G).
* **Final result (200M tokens, 500-window eval, `results_t200/README.md`):** G_VR (no attention, ~69 B/token table)
  1.690 code / 3.324 prose vs B (7:1 hybrid, 2,048 B/token KV) 1.730 / 3.384 and G (no attention) 1.899 / 3.444.
  G_V: 1.703. The verifier removes most of the misleading-lookup harm (lk_wrong: G_C 3.408, G_V 3.251, G_VR 3.211,
  G 3.215). It does **not** fix stale values / "which object" conflicts (CodeRecall ≤ B for every ICE arm).
  ICE-off is still worse than G (laziness reduced, not removed). VR's beyond-exact candidates help loss only.
  One seed.
* All results are committed; the W&B runs t200_* are uploaded.

## 2. Background

* Proposal: `proposal.tex/.pdf`, explainer `ice_explained.*`. Reviewer feedback (score 5/10) in `feedback.md`:
  (1) misleading lookups / conflicting continuations, (2) over-reliance on copying, (3) real memory + speed incl. the
  table, (4) one concrete use case. Response and all earlier results: `feedback_response.md`.
* Model: 140M preset (16 layers, d=768), TinyLlama tokenizer (32K vocab), 4K training length, FineWeb-Edu + 30% code
  (codeparrot). Arms in `ice_lm/ice_lm/config.py` (`ARMS`):
  * A (3:1), B (7:1), C (7:1 + ICE incl. trust), C_notrust, F (15:1 + ICE), B_pm, C_in/C_out/C_nodrop
  * **G** (`layout="gdn"`, no attention), **G_C** (G + ICE), **G_V** (G + ICE-V), **G_VR** (G + ICE-VR),
    G_V_in (V + input-side injection), G_V_nover (no verifier term), C_V / C_VR (7:1 variants, not run)
* Evaluation (`scripts/evaluate.py`): held-out loss splits (all / copyable / copy_far / lk_right / lk_wrong / lk_none),
  follow rate = P(top-1 = wrong lookup token), ICE-off control (`*_off`), CodeRecall (deciding-digit accuracy per
  category × gap 256/1K/3K, paired bootstrap CIs vs B), synthetic tasks. `scripts/benchmark.py`: memory + speed.

## 3. What was done this session (in order)

1. Code hardened over several review rounds; the Kaggle notebook builds (`scripts/build_notebook.py`).
2. **RTX 3090 (RunPod, now deleted): B vs C, 140M, 80M tokens.** Full eval + benchmark (`results_3090/`).
3. **Modal H100, workspace `yooo`:** B, C and C_notrust (400M tokens); then G, G_C and A. The workspace hit its
   spend limit at ~21:25 local. Only **B** finished, and its model download was corrupted, as was C's checkpoint.
   **No usable H100 models from these runs.** The full curves are kept in `results_h100/runs/*/log.jsonl`,
   `results_h100/logs/`, and W&B project `ice-lm` (runs 140m_*).
4. W&B logging added (`init_wandb` in `ice_lm/utils.py`; Modal secret `wandb`), plus `scripts/wandb_backfill.py`.
5. **Literature review** (3 Sonnet agents) → **3 advisor proposals → 3 critics → rebuttals → judge**
   (all in `research/`). Decision: `research/decision.md`.
6. ICE-V / ICE-VR implemented (`research/impl_notes.md`): 125 tests pass, and the 13 old arms are bit-identical.
7. Free checks: D2 (training efficiency), D3 (coverage gate), D1 (long context): `research/checks_results.md`.
8. **Modal H100, workspace `tien-phat140205` (profile `run3`): G, G_C, G_V, G_VR and B at 200M tokens** (running /
   finishing at hand-off).
9. Slides (2-slide GStar update): https://claude.ai/artifact/PWbf5FG4LhLbPQgFHZ5ZoL (uses 3090 numbers; needs
   updating, see §8).
10. Code pushed to GitHub (`huypg-zenai/ice-lm`, private).

## 4. Results

**Day 0 (no training, real code).** The longest-match successor is right on 46.7% of tokens: 76.8% when the two
latest occurrences agree vs 20.3% when they disagree.

**3090, 80M tokens (B vs C), full eval** (`results_3090/results/results_140m.json`):

| Metric | B | C | C_off |
|---|---|---|---|
| Code loss | 2.885 | 2.231 | 3.299 |
| Follow wrong | 0.150 | 0.407 | — |

* CodeRecall vs B: copy_same +0.44, conflict_g1 +0.66, override_same +0.23, override_diff −0.06.
* Cost: ICE 41 B/token vs KV 2,048; decode −7%.
* This was early training. The large gain does not survive longer training (next table).

**H100, 400M-token runs (training-time eval, 8 val windows, matched steps; models lost):**

| Step (tokens) | B code | C code | C_notrust code | Follow wrong B / C |
|---|---|---|---|---|
| 750 (98M) | 2.425 | 2.157 | 2.166 | 0.196 / 0.356 |
| 1500 (197M) | 1.799 | 1.774 | 1.783 | — |
| 2750 (360M) | 1.576 | 1.566 | 1.560 | 0.211 / 0.240 |

* D2: C reaches B's loss with 40% fewer tokens early, but only 3–5% fewer after ~130M tokens. Not claimable at
  full training.

**200M-token runs, workspace run3 (training-time eval, matched steps), the current main experiment:**

| Step | G (0 attn) | G_C (ICE) | G_V (ICE-V) | G_VR (ICE-VR) | B (7:1) |
|---|---|---|---|---|---|
| 750, code all | 2.453 | 2.136 | 2.048 | 2.028 | 2.327 |
| 750, lk_wrong | 3.770 | **3.969** | 3.798 | **3.727** | 3.740 |
| 1250, code all | 2.046 | 1.899 | 1.835 | **1.818** | 1.876 |
| 1500, code all | 1.986 | (pending) | **1.794** | (pending) | 1.816 |
| 1500, copyable | 1.449 | — | **1.079** | — | 1.138 |
| 1500, lk_wrong | 3.295 | — | 3.339 | — | **3.245** |
| 1500, prose all | 3.377 | — | **3.265** | — | 3.316 |
| 1500, follow wrong | 0.203 | — | 0.340 | — | 0.234 |

Read-out so far:
* **ICE-V removes the old ICE's harm on misleading lookups.** At step 750, G_C's lk_wrong is 3.969 vs 3.770 for G;
  G_V's is 3.798.
* G_V and G_VR beat G_C everywhere.
* G_V (no attention) is level with or slightly better than B on all/copyable/prose code loss at 197M tokens.
* ICE-VR adds a small further gain at step 1250 (−0.017).
* Caveats:
  * one seed;
  * training-time evals use only 8 windows (the final eval uses 500);
  * the follow rate stays high for V (top-1 lands on the wrong lookup ~34%) although the loss there is fine;
    the model hedges rather than refuses.

**Free checks (`research/checks_results.md`, on the 3090 models):**
* D3 coverage gate **passed**. The correct token is in VR's candidates for override_diff 100%, copy_diff 91%,
  conflict_g5 87% (V: 7 / 5 / 18%). conflict_g5 is near-vacuous (~9 digit candidates).
* D1 long context: on 61 code files > 8K tokens, B degrades < 0.10 beyond 8K, so drop the long-context claim (for
  80M models). CodeRecall at 6K–24K gaps: B ≈ 0, C 0.36–0.77.

## 5. Method: ICE-V / ICE-VR (`research/decision.md`, `research/impl_notes.md`)

* **Candidates:**
  * V: 12 slots, (order ∈ {1,2,3,4,6,8}) × (last, second occurrence), each with successor token, distance
    bucket, exact match length, source id. ~45 B/token.
  * VR: 34 slots = V + deep suffix chain (occurrences 3–8 of the longest order, per-occurrence match length) +
    identifier-anchor spans. Every slot with source index > t is masked (asserted). ~69 B/token. Needs
    `set_alpha_mask`.
* **Scoring:**
  * The backbone logits z never see ICE.
  * s_k = b + L[len] + D[dist] + γ·clip(log p_LM(c_k)) for each candidate.
  * Sentinel s_0 = wᵀRMSNorm(h) + β·max log p_LM.
  * log π = log_softmax over slots.
  * p(y) = π_0·p_LM(y) + Σ_{k: c_k=y} π_k, computed in log space inside the chunked loss.
  * The auxiliary 0.5·CE(z, y) keeps the backbone usable alone. ICE-dropout is 0 for V arms.
* **Eval paths:** one `_mix` serves all eval paths. ice_keep=0 gives exactly log_softmax(z).
* **Code:** `ice_lm/ice_lm/ice_features.py` (candidates), `model.py` (ICEPointer, `_mix`), `inference.py`
  (ICEIndex incremental candidates), `data.py`, `evals.py`, `config.py`. Tests in `ice_lm/tests/test_ice_lm.py`
  (125).

## 6. Collected results (done 26 Oct, 23:15)

* On Modal workspace **tien-phat140205** (local profile `run3`), app `ice-lm`, volume `ice-lm`:
  * runs `runs/t200_{B,G,G_C,G_V,G_VR}`; B and G_V were finished at hand-off, the others within ~10 min;
  * `evaluate_when_done` (detached) then writes `results_t200.json` and `figs_results_t200/`
    (frontier.png, coderecall.png, harm.png). Expected ~23:30 local.
* Collect: the commands are in `STATUS.md` (download, then W&B backfill of the 5 runs, which trained before the
  secret existed).
* If the evaluation failed:
  `MODAL_PROFILE=run3 modal run modal_app.py::evaluate --runs t200_B,t200_G,t200_G_C,t200_G_V,t200_G_VR --n-task 50 --val-batches 500 --out results_t200.json`
* Then: put the numbers into `feedback_response.md` and the slides, commit `results_t200/` (small JSON/PNG/logs
  only), push.

## 7. Infrastructure and accounts

| Modal profile | Workspace | State |
|---|---|---|
| `huypg-zenai` | huypg-zenai | No GPU billing (CPU checks only) |
| `yooo` | yooo | **Disabled (spend limit)**; its volume holds old checkpoints (unreadable while disabled) |
| `run3` | tien-phat140205 | **Active.** Shared with other projects (volumes arm-*, verl-*, …): do not touch them. Has secrets `wandb` (ours) and huggingface-* (not ours) |

* Modal CLI: `/private/tmp/claude-501/-Users-phunghuy-code-gstar/a1fcb2a4-ae79-48f6-b6e8-350e5f8c37f0/scratchpad/mvenv/bin/modal`
  (a temp venv; `pip install modal` elsewhere if it is gone). Always run from `ice_lm/`. Set `ICE_WANDB=0` only if
  the workspace has no `wandb` secret (it is baked into the image so the container matches).
* `modal_app.py` entrypoints:
  * `tests_cpu` (with `ICE_CPU_ONLY=1`), `prepare --train-tokens 4.2e8`;
  * `train --preset 140m --arm X --gpu H100 [--total-tokens 2e8 --name t200_X]` (detached; resumes from checkpoint);
  * `evaluate`, `evaluate_when_done`, `run --cmd "..." [--gpu]`.
* Measured speed (H100, 140m): B 89K, G 85K, A 93K, C/G_C 77–80K, G_V/G_VR 81K tok/s. 400M tokens ≈ 1.3–1.4 h;
  200M ≈ 0.7 h.
* W&B: project `ice-lm` at https://forge.coreweave.com/wandb/phunghuysoict-hust/ice-lm (test project
  `ice-lm-selftest` can be deleted).
* fla kernels pinned: `fla-core==0.4.2 flash-linear-attention==0.4.2` (newer ones break with torch 2.6's triton).
* GitHub: `gh` is installed and logged in as huypg-zenai; `proposal_ice/` tracks `origin/main`.
* **Rotate:** all credentials pasted in chat (3 Modal tokens, the W&B key, the RunPod key). Never write tokens into
  files.

## 8. Next steps (priority order)

1. Collect the final 200M eval (§6). Confirm on 500 val windows + CodeRecall + ICE-off control:
   * G_V vs B on code loss;
   * G_V vs G_C on lk_wrong and follow;
   * G_VR vs G_V on override_diff, copy_diff, conflict_g5 (pre-registered: ≥ +0.10 with CI excluding 0 on at least
     one).
2. Update the slides (slide 2) and `feedback_response.md`. New story: "an exact lookup at ~2% of a hybrid's cache
   lets a pure recurrent model match the hybrid, if a lookup-blind model verifies the candidates".
3. Benchmark the G arms (`scripts/benchmark.py`, with arms G, G_V, B): bytes/token + tok/s at 4K/32K.
4. If budget allows:
   * a second seed of G_V and B (the claim rests on ~0.02 nats);
   * A at 200M as the upper reference;
   * G_V_nover (does the verifier term matter?);
   * G_V_in if the final eval shows V lost much of G_C's copy gain (it did not so far).
5. Long-context (D1) rerun on the G arms with `scripts/long_context_eval.py --clamp_dist`.
6. Write-up: closest prior work to cite:
   * Pointer Sentinel 1609.07843;
   * adaptive kNN-LM 2109.04212;
   * Akyürek n-gram heads 2401.12973;
   * Infini-gram 2401.17377;
   * HOLA 2607.02303 (closest: bounded exact KV for delta-rule models);
   * Engram 2601.07372;
   * LLMA 2304.04487.

## 9. Known pitfalls

* Evals print every 250 steps. Count lines × 250 for the step (an earlier report mislabeled them as ×200).
* Training-time eval = 8 windows: noisy at the 0.01 level; use the final eval for claims.
* Files downloaded while a workspace hits its spend limit can be silently truncated: check with
  `python -c "import zipfile; print(zipfile.ZipFile(p).testzip())"` (None = OK).
* `modal app stop` needs `-y` non-interactively. Detached local entrypoints keep only the last-triggered function
  alive.
* ICE features reset at EOS: concatenating documents does not test long-context ICE.
