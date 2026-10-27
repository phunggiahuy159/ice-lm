# Decision, round 3 (judge, 27 Oct 2026)

## 1. Verdicts
- **Shrinking gap: real, not extrapolatable.** G_VR − B on code is −0.117, −0.058 and −0.039 at 131M, 164M and 197M tokens. The LR anneals to 0.1×, so three points fit anything from a crossing at 265M to −0.03. Claim "equal training tokens", disclose the shrink, settle it with WSD (§5); drop "10σ".
- **Conflict-probe ΔNLLs: none cited.** The test overlaps the exploration windows, six variants were tried, the fits sit inside the headline windows and are code-only, and all gains are below 0.02. Kept, pending §4.4: on S, ℓ alone ranks the right candidate first 0.645 of the time, the pointer 0.41.
- **Decomposition: citable for G→B.** It is exact and not circular: 88% (code) and 90% (prose) of attention's gain falls on lookup-right tokens, against a 45% null from ordinary training. The gap is 0.169, so one G seed suffices. Not citable: "ICE recovers 133%" (tautological), and "the gain is copying" for prose (44%).
- **"Division of labour": fact yes, mechanism no.** No-match code is 2.511 / 2.510 vs G 2.560 and B 2.535, but the aux CE weights those tokens 1.5× and G_C is worse there. Q&A only.
- **Memory: lead with 2.9× at 4K** (3.7 vs 10.8 MiB) and 65 vs 2,048 B/token. Quote 12× (32K, no quality measured) as bytes only.
  - The 65 B is counted from index contents, not measured (§4.3). "9% slower" traces to `benchmark_t200.json` (78.4 vs 86.4 tok/s at 32K).
- **ICE-K vs V2: V2 only** (0 B, run 2). ICE-K gets no GPU: its gain is in-sample and sub-threshold, "random beats trained" is a spec mismatch, and 16 B/token makes it top-K int8 attention, which needs int8-KV baselines. Revisit only if run 2 passes, requiring ΔNLL ≤ −0.005 vs V2 and follow-wrong ≤ V2 − 0.01.

## 2. Slide edits now
**Slide 1**
- Box 3: "A wrong candidate is down-weighted, not ignored. Only 2.7K parameters."
- "Real cost?" box: "Total memory, lookup table included, and decode speed (H100)"
- Footer: "140M parameters · 200M training tokens (short: 1.4 per parameter) · Modal H100 · TinyLlama tokenizer"

**Slide 2**
- **Title:** "Without attention, ICE beats the hybrid at equal training, with a third of its memory"
- **Under the table:** "On code, 88% of what attention adds to the recurrent model is on tokens the lookup already predicts."
- **Memory tile:** "**2.9×** less memory at 4K tokens. 3.7 vs 10.8 MiB per sequence, where quality is measured. Grows 65 vs 2,048 bytes per token (12× less at 32K). Decoding 9% slower (Python index)."
- **Seeds tile:** "**2 seeds**: same win, but shrinking. Code −0.040 / −0.045. The lead fell from 0.12 to 0.04 over the last third of training; a 3× longer run is next."
- **Harm tile:** "**0.19 → 0** loss harm from wrong lookups, against no lookup. The hybrid is still 0.04 better on these tokens."
- **Still weak:** "Conflicts (an updated variable, two similar objects): the hybrid guesses at chance, and ICE does worse because it confidently copies the wrong occurrence. It follows a wrong lookup 34% of the time (hybrid 24%). ICE off: 0.14 worse than recurrent-only."
- **Next:** "3× longer training · same-memory attention layer · line-completion exact match · a real pretrained model."
- **Only if the §3 rules pass:** "Added to a frozen recurrent model, the verifier alone gets code X vs the hybrid's Y; ICE off gives back the original exactly." "Line completion, exact match: X% vs Y%."

**Script swaps** (`talk_notes.md`; still about 2 min)
- "can be ignored" → "gets down-weighted".
- After the 81% sentence, add: "In fact, 88% of what attention adds is on tokens this lookup already gets right."
- "12 times less memory per sequence at 32K tokens" → "a third of the hybrid's memory at the 4K length we tested".
- "Seed noise is a tenth of the gap." → "The lead shrank during training, from 0.12 to 0.04, so a 3× longer run is next."
- "it still picks the wrong one. And it still leans on the lookup." → "it copies the outdated value."
- "build a verifier that compares candidates" → "train 3× longer".

**Notes:** drop "seed spread 0.004"; add the curve ("≥ 17% fewer tokens to match"). New Q&A:
- Undertrained hybrid? "Possibly (60% verbatim repeat); the 3× run answers it."
- Is 65 B measured? "Counted from index contents; the Python index is larger." Update after §4.3.
- ICE off? "Beats both baselines on no-match tokens; possibly loss weighting."
- Synthetic recall? "A rare-token artefact; real text: 60% vs 97%."

## 3. Kaggle notebook
**Leakage: decided.** The fits sample all of `train.bin`, about 48% of which the backbones saw. Rebuild the bundle (`1523257` predates `--unseen`) and add `--unseen` to `FIT`. All three backbones share seed 1234 and 200M tokens, so samples ≥ 48,640 are unseen by all; seeds 7 and 8 take blocks 0 and 1.

**Settings**
- `LINE_WINDOWS = 400`, line completion only for B, G, G_VR and `fit_Gbb_v2`.
- `FIT_TOKENS = 2e7`, `VAL_BATCHES = 500`, fp32, `MAX_HOURS = 11`.
- Save per-item line EM for window-bootstrap CIs.
- Job 1 over 75 min: `N_CODERECALL = 50`, drop jobs 11–13.

| # | Job | Prediction | Rule |
|---|---|---|---|
| 1–3 | eval G_VR, B, G (**T4 vs H100**) | Each within 0.006 of `results_t200.json`; deltas within 0.003 | Deltas off: T4 vs T4 only. Arm off by > 0.02: bug |
| 4–6 | line B, G_VR, G | G_VR − B: all ≥ +1 pt, repeat ≥ +5, novel ±2 | Slide only if the CI excludes 0. All ≤ 0 or novel ≤ −3: disclose |
| 7 | gate eval | ±0.005 vs T4 G_VR | Fail: fix the fitter |
| 8 | **run 3** fit_Gbb_v2 → eval → line | Code − B(T4) ∈ [−0.025, +0.005] | ≤ −0.02 "beats"; ±0.01 "matches"; > +0.01: needs co-training |
| 9 | **run 2** fit_GVRbb_v2 → eval | Code −0.003 to −0.010, prose ±0.003, follow ≤ 0.32, conflict_g2/g5 ≥ +0.3 | Code > −0.003 or prose > +0.003: stop V2 |
| 10 | run 2, seed 8 | Within 0.003 of seed 7 | Otherwise report the range |
| 11 | **run 4** fit_Bbb_v2 → eval | −0.005 to −0.02 | < 0.005: "attention-free only" |
| 12 | recall_check, n = 50 | B real text 0.55–0.65, G_VR ≥ 0.94 | A cell moves > 0.10: update notes |
| 13 | run 5 v2syn | As in decision2 | Off the slides |

**Dropped:** `fit_Gbb_vr`, `v2q`, extra line jobs, the T4 benchmark, `longctx` (RoPE-confounded), G_MQA and ICE-K. The run-3 "H100 scale" check becomes relative.

## 4. CPU work now (Mac)
First copy the checkpoints and `ldata/` out of `/private/tmp`.

| # | Task | Time | Prediction / rule |
|---|---|---|---|
| 1 | Rebuild the notebook, dry-run with tiny fits, hand it over | 1.5 h | Tests pass, every check prints |
| 2 | Slide and notes edits | 45 min | — |
| 3 | **Memory:** numpy compact index at 4K/16K/32K (`nbytes`), plus tracemalloc of the Python index | 1.5 h | Compact 60–80 B/token, Python ≥ 500. Over 100: quote the measured value; over 200: retitle |
| 4 | **No-fit `zprobe`** on val_code windows 100–299 | 45 min | Rank-1 on S: pointer 0.36–0.45, ℓ 0.58–0.70. Gap ≥ 0.15: cite. Gap < 0.08: drop |
| 5 | **Stale-version edit test**, B and G_VR: a v1 with 10% of identifiers and literals changed, then the original | 3 h + 40 min | Unchanged: G_VR < B by ≥ 0.10. Changed: ≤ B + 0.20, else slide 2 says "copies the old version on realistic edits" |
| 6 | **WSD + decay branches** in `train.py`, cosine bit-identical, by 31 Oct | 2 h | Late: M2 becomes full-cosine |
| 7 | README: 45% null, no "10σ", ICE-off for both seeds | 30 min | — |

**Cut:** fitted conflict refits, rarity filter, drafts, truncated state, length eval; Mamba2/Pythia moves to next round.

## 5. Modal after 1 Nov (≤ $30; 200M tokens ≈ 40 H100-min ≈ $2.6)

| # | Run | $ | Prediction | Kill / decision |
|---|---|---|---|---|
| M1 | **G_MQA**, 200M cosine, + eval | 3.5 | Code 1.74–1.78 | ≤ 1.680: the equal-bytes headline is dead |
| M2 | **WSD B and G_VR to 600M**; evaluated decay branches (last 20%) end at 200M, 400M and 600M | 23 | Gap at 200M −0.03 to −0.05, 400M −0.01 to −0.035, 600M −0.025 to +0.005; B repeat ≥ 0.75 | At 600M: ≤ −0.02 "beats", (−0.02, +0.01) "matches", ≥ +0.01 fallback. 400M ≥ +0.01: skip the 600M decay |
| M3 | If M1 + M2 ≤ $26: G, CE ×1.5 on no-match tokens | 3 | No-match code ≤ 2.53 | Met: "reweighting" |

Run M1 and M2 in parallel under a spend cap; 600M is 1.43 epochs (disclose). **Cut:** seed 3, G_SWA, G_VRK, G seed 2.

## 6. Claims
**If M1/M2 work:**
> Up to 600M tokens, an attention-free model with an exact lookup and a 2.7K-parameter verifier beats both the hybrid and an attention layer with 4× its bytes. Most of what attention adds to a
> recurrent model is exact copying, which the lookup supplies. Stale values remain unsolved by both.

**Fallback:**
> At short training it matches or beats the hybrid with a third of the memory, because 88% of what attention adds is
> exact copying. With more tokens the hybrid catches up (600M gap: X): ICE substitutes for copying, not for attention.
