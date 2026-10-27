# 200M-token runs: can an exact lookup stand in for attention?

140M preset, 200M training tokens per arm, one seed, same data order, Modal H100 (workspace tien-phat140205).

| Arm | Model | Cache per token |
|---|---|---|
| G | 16 GDN layers, no attention | 0 B |
| G_C | G + ICE (original: input injection + logit bonus + trust signals) | ~41 B (table) |
| G_V | G + ICE-V (a lookup-blind backbone verifies 12 candidates; log-space pointer-sentinel) | ~45 B (table) |
| G_VR | G + ICE-VR (V + deep suffix chain + identifier-anchor spans, 34 candidates) | ~69 B (table) |
| B | 7:1 GDN/attention hybrid, the reference | 2,048 B (KV) |

`*_off` = the same model evaluated with ICE switched off. Files:
* `results_t200.json`: everything;
* `tables.txt`: all printed tables, incl. synthetic tasks;
* `figs_results_t200/`: `frontier.png`, `coderecall.png`, `harm.png`;
* `runs/*/log.jsonl`: training curves, held-out eval every 250 steps (8 windows).

## Final evaluation (held-out, 500 windows of 4K tokens per split)

| Loss (nats) | B | G | G_C | G_V | G_VR | G_C_off | G_V_off | G_VR_off |
|---|---|---|---|---|---|---|---|---|
| code, all | 1.730 | 1.899 | 1.762 | 1.703 | **1.690** | 2.328 | 2.037 | 2.042 |
| code, copyable | 1.061 | 1.374 | 1.033 | 0.996 | **0.989** | 2.104 | 1.675 | 1.675 |
| code, copy > 512 back | 1.132 | 1.680 | 0.966 | **0.926** | 0.933 | 2.265 | 1.921 | 1.942 |
| code, lookup wrong | **3.172** | 3.215 | 3.408 | 3.251 | 3.211 | 3.345 | 3.150 | 3.160 |
| code, no match | 2.535 | 2.560 | 2.579 | 2.512 | **2.506** | 2.631 | 2.512 | 2.511 |
| prose, all | 3.384 | 3.444 | 3.379 | 3.339 | **3.324** | 3.560 | 3.454 | 3.453 |
| follow wrong lookup, code | 0.239 | 0.208 | 0.315 | 0.340 | 0.340 | 0.158 | 0.164 | 0.164 |

CodeRecall, deciding-digit accuracy minus B, paired bootstrap 95% CI, items pooled over gaps 256/1K/3K:

| Category | G | G_C | G_V | G_VR |
|---|---|---|---|---|
| copy_same | −0.06 | **+0.34** [+0.29, +0.40] | +0.15 [+0.10, +0.20] | +0.17 [+0.12, +0.22] |
| conflict_g1 | −0.01 | **+0.90** [+0.86, +0.93] | +0.16 [+0.10, +0.22] | +0.23 [+0.16, +0.29] |
| override_same | −0.24 | −0.04 | **+0.15** [+0.09, +0.21] | +0.09 [+0.03, +0.15] |
| copy_diff | −0.01 | −0.03 | −0.03 | −0.03 |
| conflict_g2 | +0.02 | −0.05 | −0.04 | −0.07 |
| conflict_g5 | −0.02 | −0.10 | −0.10 | −0.10 |
| override_diff | −0.03 | −0.07 | −0.08 | −0.09 |
| increment | +0.01 | +0.86 | 0.00 | 0.00 |

## What it shows

1. **A pure recurrent model with an exact lookup beats the attention hybrid on held-out loss.** G_VR is 0.040 better
   than B on code (1.690 vs 1.730) and 0.060 better on prose, with no KV cache: ~69 B/token of table vs
   2,048 B/token of KV. Over the G→B gap (0.169 on code), G_C closes 81%, G_V 116%, G_VR 124%.
2. **The verifier removes most of the harm from misleading lookups.** Where the lookup is wrong, the old ICE (G_C) is
   0.19 worse than G; G_V is 0.04 worse, and G_VR equals G. It does not copy less often: top-1 = the wrong
   lookup 34% of the time vs 24% for B. The model hedges instead of refusing.
3. **The verifier gives up some exact-copy power.** On CodeRecall, G_C is far ahead where the lookup is right
   (conflict_g1 +0.90, copy_same +0.34); G_V/G_VR gain +0.15 to +0.23. G_V/G_VR are better on override_same
   (latest value).
4. **Nothing fixes the cases where the lookup is wrong in a structured way.** Stale values (override_diff),
   "which object" (conflict_g2/g5) and other wordings (copy_diff) are all ≤ B for every ICE arm. VR's richer
   candidates contain the answer (coverage gate: override_diff 100%, copy_diff 91%) but the model does not pick it.
   Pre-registered test G_VR − G_V ≥ +0.10 on one of them: **failed** (all within ±0.01). VR helps only on loss
   (−0.013 code).
5. **Laziness is reduced, not removed.** With ICE switched off, G_V is 0.14 worse than G on code (G_C: 0.43 worse).
   The backbone still leaves copying to the lookup, even though it never sees it.
6. G_C's +0.86 on `increment` (where the exact lookup is wrong on the last digit) is unexplained. Its input-side
   features may expose the digit sequence; to check before citing.

Pre-registered predictions (`research/decision.md`):

| Comparison | Prediction | Outcome |
|---|---|---|
| G_C vs G | code −≥0.05 | **pass** (−0.137) |
| G_C vs G | copy_far −≥0.30 | **pass** (−0.71) |
| G_C vs G | copy_same / conflict_g1 ≥ +0.30 | **pass** (+0.40, +0.91) |
| G_V vs G_C | lk_wrong ≤ −0.10 | **pass** (−0.16) |
| G_V vs G_C | keeps ≥ 75% of the copyable gain | **pass** (111%) |
| G_V vs G_C | follow-wrong ≤ −0.05 | **fail** (+0.03) |
| G_V vs G_C | ICE-off ≤ G + 0.03 | **fail** (+0.14) |
| G_VR vs G_V | ≥ +0.10 on a structured-conflict category | **fail** |


## Seed 2 (B and G_VR, same recipe, `--seed 2`; `seed2/results_t200_seed2.json`)

| Metric | B s1 | B s2 | G_VR s1 | G_VR s2 | G_VR − B s1 | G_VR − B s2 |
|---|---|---|---|---|---|---|
| code, all | 1.730 | 1.734 | 1.690 | 1.689 | −0.040 | −0.045 |
| code, copyable | 1.061 | 1.064 | 0.989 | 0.989 | −0.071 | −0.076 |
| code, copy > 512 back | 1.132 | 1.135 | 0.933 | 0.933 | −0.199 | −0.203 |
| code, lookup wrong | 3.172 | 3.173 | 3.211 | 3.211 | +0.039 | +0.037 |
| follow wrong lookup | 0.239 | 0.236 | 0.340 | 0.342 | +0.10 | +0.11 |
| prose, all | 3.384 | 3.383 | 3.324 | 3.323 | −0.060 | −0.061 |

Seed-to-seed spread ≈ 0.004 on code loss, so the G_VR − B gap (−0.040 / −0.045) is ~10σ. By the pre-registered rule
(research/round2/decision2.md, F0) this counts as "beats B (2 seeds)", and seed 3 is not needed. CodeRecall is
noisier across seeds (e.g. G_VR conflict_g1 0.31 vs 0.18; override_same 0.42 vs 0.56).

Caveats (seed 1 section above): 200M tokens; 140M parameters.

## Cost, measured (H100, bf16, batch 1; `benchmark_t200.json`, `benchmark_t200.log`)
| | B | G | G_VR |
|---|---|---|---|
| decode tok/s at 4K / 32K | 87.2 / 86.4 | 84.2 / 85.2 | 80.1 / 78.4 |
| ICE CPU time per decode step | – | – | 346–360 µs (Python index) |
| cache per sequence at 4K | 10.8 MiB | 3.2 MiB | 3.7 MiB (2.9× less than B) |
| cache per sequence at 32K | 66.8 MiB | 3.2 MiB | 5.5 MiB (12× less) |
| growth per token | 2,048 B (KV) | 0 | 64.7 B (table) |

- The ICE table bytes are measured from the index's contents (distinct n-grams, tokens, chain pointers) under the
  compact layout of `inference.ICEIndex.compact_bytes`. That layout is what a C/GPU index needs; it is not the
  size of the Python dicts used here.
- Quality is measured at 4K, where the memory saving is 2.9×. The 12× is at 32K.
- **Measured index** (`scripts/index_memory.py`, `research/round3/index_memory.json`). The compact layout was built
  as real numpy arrays (power-of-two open-addressing tables) and every lookup was checked against the Python index:
  - 82.0 / 82.0 / 90.0 B/token at 4K / 16K / 32K, plus a 250 KiB fixed order-1 table. The count above (64.7) assumes
    an ideal load factor of 0.5.
  - With the measured index: 3.8 vs 10.8 MiB per sequence at 4K (2.8×), 6.3 vs 66.8 MiB at 32K (11×).
  - The Python dict index used in this repo takes 520–640 B/token (tracemalloc).

## Synthetic recall: the hybrid's 0.00 is mostly an artefact (27 Oct, CPU check)
`tables.txt` reports B at about 0 on repeat and MQAR. Those tasks draw tokens uniformly from about 31K vocabulary
ids, mostly rare tokens whose embeddings barely trained. Attention has to match them through the embeddings, while
ICE matches the ids directly. `scripts/recall_check.py` re-scores the same tasks with frequent tokens and with
real text. The CPU run on the real t200 checkpoints used n = 10 per cell, fp32
(`research/round3/recall_check_cpu_n10.json`):

| repeat, token accuracy | B | G_VR |
|---|---|---|
| random tokens, full vocabulary (as in evaluate.py) | 0.04–0.05 | 0.98–1.00 |
| random frequent tokens (top 5K) | 0.44–0.62 | 0.98–1.00 |
| real prose, repeated verbatim | 0.58–0.62 | 0.94–0.97 |
| **MQAR**, frequent tokens | 0.09–0.19 | 0.61–0.73 |

- **Do not cite the full-vocab synthetic numbers as "0 vs 0.9".** The fair statement is that B repeats real text
  about 60% of the time and ICE about 97%.
- **ICE is weaker on frequent-token MQAR than on full-vocab MQAR,** because spurious matches appear. This is the
  same conflict weakness as in CodeRecall.
- The Kaggle notebook (`ice_lm/notebooks/ice_eval_kaggle.ipynb`) reruns this check with n = 50.
