# Kaggle results (2× T4, fp32, 27–28 Oct 2026): analysis

Notebook `ice_lm/notebooks/ice_eval_kaggle.ipynb`. All 19 jobs finished.
- `SUMMARY.md`: the pre-registered checks.
- `all_eval_tables.txt`: every metric.
- `line_*.json`: line completion, per item.
- Plan and predictions: `research/round3/decision3.md` §3.
- The CI computations for this file were done ad hoc, not by the notebook (window bootstrap, 2,000 resamples).

## Names
| Short name | Run | What it is |
|---|---|---|
| G | t200_G | 16 GDN layers, no attention, no ICE |
| B | t200_B | 7:1 hybrid (2 attention layers) |
| G_VR | t200_G_VR | G trained jointly with ICE-VR (the main model so far) |
| G+VR refit | fit_GVRbb_vr | G_VR's backbone frozen, VR scorer refitted (the gate) |
| **G+V2** | fit_Gbb_v2 | **plain G backbone, frozen; only the 2,873-parameter V2 scorer is fitted** |
| **G_VR+V2** | fit_GVRbb_v2 (seed 7), `_s8` (seed 8) | G_VR's backbone frozen, V2 scorer fitted |
| **B+V2** | fit_Bbb_v2 | the hybrid's backbone frozen, V2 scorer fitted |
| G_VR+V2+syn | fit_GVRbb_v2syn | as G_VR+V2, with 2% synthetic conflict/override windows |

Fits: 20M tokens, `--unseen`, i.e. only training windows none of the backbones saw. Evals: 500 held-out windows of 4K tokens per split.

## 1. The T4 reproduces the H100 exactly
B, G and G_VR match `results_t200.json` within 0.0001 on code and prose loss, and their differences match within 0.0000. All numbers below are directly comparable with the H100 results.

## 2. Held-out loss (lower is better)
| | Cache | Code | Prose | Copy > 512 back | Wrong-lookup tokens | Follow wrong |
|---|---|---|---|---|---|---|
| G | 0 B | 1.899 | 3.444 | 1.680 | 3.215 | 0.208 |
| B (hybrid) | 2,048 B | 1.730 | 3.384 | 1.132 | **3.171** | 0.239 |
| G_VR | 65–90 B | 1.690 | **3.324** | 0.933 | 3.211 | 0.340 |
| G+VR refit | 65–90 B | 1.683 | 3.324 | 0.898 | 3.207 | 0.329 |
| **G+V2** (frozen G) | 65–90 B | **1.710** | 3.376 | 0.909 | 3.271 | 0.314 |
| **G_VR+V2** (s7 / s8) | 65–90 B | **1.680 / 1.681** | **3.324** | 0.895 | 3.208 | 0.319 |
| **B+V2** (frozen hybrid) | 2,048 + 65–90 B | **1.665** | 3.356 | **0.863** | 3.191 | 0.294 |

- **Over-reliance is answered.** G+V2 freezes the plain recurrent model and fits only a 2,873-parameter scorer. It beats the hybrid on code by 0.021 (pre-registered verdict: "beats", rule ≤ −0.02) and on prose by 0.008. With ICE off it is G exactly, by construction, so it has no reliance at all.
  - Joint training (G_VR) is a further 0.020 better on code, so co-training helps but is not needed to beat the hybrid.
- **ICE also helps the hybrid.** B+V2 is 0.065 better than B on code and 0.028 better on prose. This is the best code loss of any model.
  - The pre-registered range was −0.005 to −0.02, so the check reads FAIL. It failed in the favourable direction: the prediction was wrong, and the "attention-free only" fallback does not apply.
  - The old jointly trained ICE on the 7:1 hybrid (C) had shrunk to about −0.01; the post-hoc V2 verifier does not shrink like that.
- **V2 vs VR on loss is small.** G_VR+V2 (1.6803) vs the VR refit (1.6833) is −0.003. Most of run 2's −0.009 over G_VR comes from refitting the scorer on a frozen backbone (the gate, −0.0064), not from the V2 features.
  - The gate's FAIL (outside ±0.005) is in the favourable direction: the refit is better than the jointly trained scorer, not broken.
  - Seed 8 is within 0.0002 of seed 7.
- **Remaining cost on wrong-lookup tokens.** Against its own backbone, G+V2 is +0.056 worse there: the frozen backbone cannot adapt. B+V2 is +0.020 worse than B.
  - Follow-wrong rates stay at 0.29–0.33, against 0.24 for the hybrid.

## 3. Line completion (greedy, exact match of the whole line)
There are 1,200 lines from 400 held-out 4K code windows, the same lines for every model.

| | All | Repeated lines (228) | Reuse lines (721) | Novel lines (251) | Edit similarity, all |
|---|---|---|---|---|---|
| G | 9.4% | 26.3% | 4.3% | 8.8% | 0.53 |
| B | 11.8% | 40.8% | 4.3% | 7.2% | 0.56 |
| G_VR | 13.3% | 51.3% | 3.2% | 8.0% | 0.58 |
| **G+V2** | **15.7%** | **63.6%** | 3.1% | 8.4% | 0.58 |

Differences, with window-bootstrap 95% CIs:

| | All | Repeated | Novel | Reuse |
|---|---|---|---|---|
| G_VR − B | +1.5 pts [+0.2, +2.9] | +10.5 [+4.4, +16.9] | +0.8 [−1.3, +3.0] | −1.1 [−1.9, −0.4] |
| G+V2 − B | +3.8 pts [+2.2, +5.6] | +22.8 [+16.3, +29.7] | +1.2 [−1.5, +3.8] | −1.2 [−2.4, −0.1] |

- All pre-registered checks passed: all ≥ +1, repeated ≥ +5, novel within ±2.
- The gain is on repeated lines. On "reuse" lines, which reuse earlier tokens but not the whole line, ICE is about 1 point worse: a small but real cost of copying.

## 4. Conflicts ("which object"): fixed by V2, without task supervision
CodeRecall deciding-digit accuracy, mean over gaps:

| | conflict_g1 | conflict_g2 | conflict_g5 | override_same | override_diff | copy_same |
|---|---|---|---|---|---|---|
| B | 0.09 | 0.08 | 0.10 | 0.33 | 0.09 | 0.14 |
| G_VR | 0.32 | **0.00** | **0.00** | **0.42** | 0.01 | 0.31 |
| G+VR refit | 0.42 | 0.14 | 0.12 | 0.40 | 0.01 | 0.35 |
| G+V2 | 0.97 | 0.67 | 0.77 | 0.15 | 0.05 | 0.22 |
| G_VR+V2 (s7 / s8) | **0.99** | **0.81 / 0.87** | **0.83 / 0.84** | 0.15 | 0.01 | 0.32 |
| B+V2 | 0.99 | 0.70 | 0.82 | 0.37 | 0.09 | 0.23 |

- **"Which object" conflicts are solved.** g2 and g5 are the cases where the most recent occurrence belongs to another object. They go from 0.00 to 0.81–0.87 (vs B: +0.74 to +0.79, CI excludes 0), with a scorer fitted on natural code only.
  - It generalises: held-out perturbed conflict templates (positional arguments, swapped fields) score 0.93–1.00, against 0.05–0.25 for B and G_VR.
  - Mechanism: the V2 features compare candidates. The right occurrence matches a longer prefix, including the object's name, than the most recent wrong one.
- **New trade-off: override_same drops from 0.42 to 0.15.** Here a value is reassigned with the same syntax, so the latest write is right. V2 prefers the longest match over the latest write. B+V2 keeps 0.37. Report this next to the conflict gain.
- **Stale values (override_diff) are not solved by anything** (0.01–0.09).
  - Run 5 adds 2% synthetic override items, and even the *training* override template stays at 0.03. The scorer cannot represent "the latest write in another syntax wins": a feature limit, not a data limit.
  - Pre-registered override_diff ≥ +0.3: FAIL.
  - Run 5 otherwise equals run 2, so the synthetic data adds nothing.

## 5. Retrieval tasks (exact match of the whole answer, 50 items each)
| | Phone-book 1K / 4K | Needle 1K / 4K | Paraphrased needle 4K (token accuracy) |
|---|---|---|---|
| G | 0% / 0% | 0% / 0% | 0.01 |
| B | 0% / 0% | 16% / 2% | 0.19 |
| G_VR | 82% / 82% | 92% / 66% | 0.56 |
| G+V2 | 76% / 76% | 94% / 96% | 0.33 |
| G_VR+V2 | 92% / 86% | 100% / 98% | 0.59 |
| B+V2 | 42% / 56% | 86% / 82% | 0.54 |

- V2 lifts the needle at 4K from 66% to 96–98%.
- These G_VR / B numbers differ by 1 item from the H100 run (needle 4K: 64% → 66%, B 0% → 2%).
- The paraphrased needle stays near 0% exact match for every model: an exact lookup cannot help with reworded questions.

## 6. Recall check (n = 50 per cell; `recall_check.json`)
Repeating real prose verbatim, token accuracy: B 0.41–0.63, G 0.38–0.49, G_VR 0.93–0.97.
- MQAR, frequent tokens: B 0.07–0.16, G 0.00–0.06, G_VR 0.53–0.65.
- Pre-registered checks pass: B 0.55, G_VR 0.96.

## 7. What this changes
- **New best claims, all pre-registered or CI-backed:**
  1. A frozen recurrent model plus a 2.9K-parameter scorer beats the 7:1 hybrid (code −0.021, prose −0.008, line exact match +3.8 pts), with zero reliance.
  2. V2 solves "which object" conflicts (0 → 0.8+) without task supervision.
  3. ICE also improves the hybrid (−0.065 code), so it complements attention as well as replacing it.
- **New caveats:**
  - override_same regresses with V2 (0.42 → 0.15).
  - Stale values remain unsolved, even with supervision.
  - Reuse lines lose about 1 point of exact match.
  - Most of V2's loss gain over G_VR is the refit.
