# Round 3 proposal: evidence (27 Oct 2026)

Short answer: the 4K loss result holds. Two cheap-to-check attacks can still take it down: "your baseline is undertrained" and "1/12 is measured where quality is not". The current Kaggle job list answers neither.

## 1. Audit

**Solid**
- **Held-out loss, 2 seeds:** G_VR − B = −0.040 / −0.045 (code), −0.060 / −0.061 (prose). Same recipe for every arm.
- **The gain is copying:** copy > 512 is 0.93 vs 1.13 nats; verbatim repeat 94–100% vs 44–62% (CPU, n = 10).
- **The verifier removes the old harm:** lookup-wrong loss is G_VR 3.211 = G 3.215, vs G_C 3.408.

**Fragile**
1. **The gap shrinks with training.** This is free to see in `runs/*/log.jsonl` (8 in-training windows):

   | step (tokens) | 1000 (131M) | 1250 (164M) | 1500 (197M) |
   |---|---|---|---|
   | G_VR − B, code | −0.117 | −0.058 | −0.039 |
   | G − B, code | +0.175 | +0.170 | +0.170 |
   | B gain per 250 steps | | −0.134 | −0.060 |
   | G_VR gain per 250 steps | | −0.075 | −0.041 |

   B and G learn at the same late rate; G_VR more slowly. ICE is a head start that fades. G_VR at step 1250 (1.818) equals B at 1500 (1.816), so ICE is now worth about 20% of the tokens. The runs use 1.4 tokens per parameter.

   **A reviewer who opens the curves attacks this first:** "your hybrid's attention is still learning to copy". B's 60% on verbatim repeat supports them; a well-trained hybrid repeats at about 100%.
2. **"1/12 of the memory" is measured at 32K, but quality is measured at 4K.** At 4K the per-sequence cache is about 3.7 MiB (G_VR) vs 10.8 MiB (B), or **1/2.9**. That is 16 × 0.19 MiB GDN state + 65 B × 4K + 0.25 MiB order-1 table, against 2.8 MiB + 2,048 B × 4K.
   - The 64.7 B/token is a **modelled** compact layout (`benchmark.py` docstring). The Python index that ran is not measured, and the reviewer asked for "actual total memory … including the lookup table".
   - Weights are 280 MB in bf16, so the saving matters only for batch × length serving. Say so.
3. **Loss is not behaviour.** Top-1 is +0.4 pt (66.5 vs 66.1%) and there is no generation metric yet. A −0.04 that is mostly calibration on copyable tokens is the next target.
4. **"10σ" comes from two seeds.** A spread from n = 2 is not a σ. Say "both seeds, gaps 0.040 / 0.045" and add a window-level paired bootstrap CI.
5. **No equal-memory attention.** At 4K, G_MQA *is* the equal-memory arm (15 × 0.19 + 256 B × 4K ≈ 4.0 MiB vs 3.7), which is cleaner than any 32K comparison.
6. **Conflicts, corrections, wrong lookups** are the cases the feedback named, and ICE loses on all of them: follow-wrong 34% vs 24%, every structured CodeRecall category ≤ B.
7. **Synthetic recall** is no longer ours: B's "0.00" was a rare-token artefact, and G_VR is worse on frequent-token MQAR (0.61–0.73) than on full-vocab. Cite only natural-text repeat.

**Attack order I expect:** (1) undertrained B / shrinking curve, (2) 1/12 at an unmeasured length, (3) equal-memory attention, (4) loss ≠ generation, (5) conflicts and over-reliance, (6) toy scale.

## 2. Missing experiments, ranked by credibility per compute

| # | Experiment | Where / time | Pre-registered prediction | Weakens the claim if |
|---|---|---|---|---|
| E1 | **Disclose the curve.** Plot G_VR − B vs tokens and state tokens-to-match. | Mac, 15 min (done above) | — | (already observed: the gap is shrinking) |
| E2 | **Measured memory.** Cache at 4K / 16K / 32K from the formula plus `benchmark.json`. Measure the Python index (tracemalloc / RSS) and a numpy-array compact index at 32K. | Mac CPU, 1–2 h | Compact index measured ≤ 100 B/token. Python index ≥ 500 B/token (disclose it). | Compact > 200 B/token: the headline becomes ≈ 1/6 at 32K |
| E3 | **Frozen G + fitted scorer** (run 3, `fit_Gbb_v2`). ICE off = G exactly, which answers over-reliance. | Kaggle, ~1.5 GPU-h (already in the notebook) | Code ≤ B(T4) + 0.005 → "matches the hybrid with zero reliance" | > B + 0.02: the win needs co-training, and the reliance critique stands |
| E4 | **Stale-version test on real code** (a new eval-only script, ~2–3 h of code). v1 = a held-out file with 10% of identifiers renamed and literals changed; context = v1 then the original v2. Score v2 by unchanged / changed spans, plus the follow-stale rate. It is the reviewer's "correction later in context" on a realistic edit workload. | Mac CPU (B and G_VR are local), 200 windows ≈ 30–40 min; G on Kaggle | Unchanged spans: G_VR < B by ≥ 0.10. Changed spans: G_VR ≤ B + 0.20 and ≤ G + 0.05. Follow-stale: G_VR > B. | Changed spans > G + 0.10 (the lookup does harm under edits), or the window total > B |
| E5 | **Line completion, powered** (generation exact match). | Kaggle, ~0.5 h per arm (estimate) | G_VR − B exact match: repeat ≥ +5 pt, reuse ≥ +1, novel ±2, all ≥ +1 | all ≤ 0, or novel ≤ −3 (copying hurts generation) |
| E6 | **G_MQA at 200M**, the equal-memory arm at 4K. | Modal 1 Nov, 40 min H100, ~$3. Not Kaggle: ~1 day of T4 quota. | Code 1.74–1.78 | G_MQA ≤ G_VR − 0.01: the equal-bytes headline is dead |
| E7 | **B and G_VR at 600M tokens** (3×). This answers attack 1. | Modal, ~2 h H100 each in parallel, ~$15 | G_VR − B ∈ [−0.03, −0.01] | ≥ −0.005: the claim becomes "matches B; ICE = 1/3 of the cache + data efficiency" |
| E8 | **Mamba2-370M vs Pythia-410M bolt-on** (frozen, scorer only). Pythia is *fully trained* attention, so it also answers attack 1. | Kaggle; ~1 day of code (no `bolt_on.py`); 4–6 GPU-h. First check on CPU that the HF Mamba2 checkpoint loads and that the torch SSD fallback is fast enough. | Premise: copy > 512 gap ≥ 0.10. ICE closes ≥ 50% of it; prose harm ≤ +0.005; ICE on Pythia ≤ 1/3 of its gain on Mamba2. | Closes < 25%, or the premise gap is < 0.05 |
| E9 | **8–32K evaluation** (`long_context_eval.py`). | Kaggle, ~0.5 h per arm | B's copy-far loss rises ≥ 0.10 beyond 4K; G_VR's rises ≤ 0.03 | G_VR copyable +0.05 at 16–32K (an earlier 18-document hint showed +0.09) |
| E10 | **G_SWA at equal memory at 4K** (W ≈ 700). | Modal; ~4 h of code + $3 | G_VR beats it by ≥ 0.02 and by ≥ 0.3 on copy > 700 | Only if E6 is within 0.01 |
| E11 | Seed 3; a bigger B | — | — | Low value: seeds agree; E8 covers scale |

E9 ranks low: val has only 61 documents > 8K (18 > 16K, 2 > 32K), and B (RoPE θ = 10⁴, trained at 4K) extrapolates badly for reasons unrelated to memory, so a win there is confounded. Run it, but **claim memory at 4K** (1/2.9) and quote 32K only as per-token growth: 65 vs 2,048 B (1/32).

## 3. Kaggle notebook: concrete changes

The current order is: G_VR (+line) → gate (+line) → B (+line) → recall → V2 on G_VR → G → `fit_Gbb_vr` → `fit_Gbb_v2` → `fit_Bbb_v2` → syn → s8 → q.

1. **Line completion only where it is read.** `eval_chain` adds `line_*` to all 11 chains; the gate's and most fits' are waste. Keep it for `t200_B`, `t200_G`, `t200_G_VR`, `fit_Gbb_v2`, `fit_GVRbb_v2`. Raise `LINE_WINDOWS` 100 → 400: at 300 lines the ~70-line "repeat" subset has SE ±6 pt, too wide for a +5 pt claim. Add a paired bootstrap CI per kind.
2. **Move `fit_Gbb_v2` / `fit_Gbb_vr` (run 3) to right after the gate and B**, ahead of `fit_GVRbb_v2`: it is the over-reliance answer and E8's template. Add the check `fit_Gbb_v2 − t200_B (T4) ≤ +0.005`; the current check is an absolute "1.705–1.735 (H100 scale)" from another GPU.
3. **Add a `bench` chain:** `benchmark.py --runs B G G_VR --contexts 4096 16384 32768` (~20 min): measured T4 decode tok/s and peak memory, as the feedback asks.
4. **Add a `longctx` chain** (E9) for B, G_VR and `fit_Gbb_v2`, placed after `fit_Bbb_v2`.
5. **Demote `recall`** to the end: the CPU check settled it. `fit_GVRbb_v2_s8` / `v2q` stay last.
6. **Do not train G_MQA on Kaggle** (~24 of 30 weekly GPU-h vs 40 min on Modal in four days). Pre-stage Modal now: G_MQA, then B / G_VR at 600M.
7. Add E4 (stale-version) once the script exists; run it on the Mac first.

## 4. Honest 2-slide story (current evidence; brackets are filled by Kaggle)

**Slide 1: An exact lookup buys back attention's copying with a near-zero cache**
- On held-out data, pure GDN + ICE-VR beats the 7:1 hybrid in both seeds: code 1.690 vs 1.730, prose 3.324 vs 3.384.
- The gain is copying: on copies > 512 back, 0.93 vs 1.13 nats and top-1 80.8% vs 76.2%; verbatim repeat of real text 94–97% vs 58–62%; aggregate top-1 only +0.4 pt.
- Cache grows at 65 vs 2,048 B/token: 3.7 vs 10.8 MiB per sequence at the evaluated 4K (1/2.9), 5.5 vs 66.8 MiB at 32K; decode 9% slower (Python index).
- Caveat: the lead shrinks with training (−0.12 at 131M tokens, −0.04 at 200M), and is now worth about 20% of B's tokens. [600M result]
- [Line completion exact match: repeat / reuse / novel, vs B, with CI.]

**Slide 2: Where copying is the wrong answer, and what that costs**
- It follows wrong lookups more than attention (top-1 = wrong lookup 34% vs 24%) but hedges: wrong-lookup loss 3.21 vs B 3.17, equal to no lookup (3.22).
- On conflicting values and corrections it is ≤ the hybrid in every category (override_diff −0.09, conflict_g5 −0.10); the right value is a candidate 91–100% of the time, but no fixed rule picks it (≤ 0.07 vs oracle 0.34).
- Over-reliance: co-trained, ICE off costs 0.14 nats vs G. [Fitted post hoc on a frozen G: code X vs B 1.730, with ICE off = G exactly.]
- [Stale-version edit test: changed spans vs B and G.]
- Not yet shown: an equal-memory attention layer (G_MQA, 1 Nov), quality beyond 4K, and a real pretrained model.

Drop "1/12" from the title until E9 (or a 32K-trained comparison) exists. Use "1/32 of the per-token cache" plus the 4K figure.
