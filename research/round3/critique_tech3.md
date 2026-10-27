# Round 3 technical critique (27 Oct 2026)

All numbers recomputed on CPU from the JSON, logs, probes and scripts.

## 1. The shrinking gap cannot be extrapolated, and a 600M run repeats data. **Major**
- **Confirmed:** −0.117 / −0.058 / −0.039, on val_code windows 0–7 (harder than the full set: B 1.816 vs 1.730).
- **The LR is cosine to 0.1× peak, not to 0:** 3.9e-4, 1.9e-4 and 1.0e-4 at steps 1000, 1250 and 1500. Late gains are mostly annealing.
- **Where the 1250→1500 shrink comes from (+0.019):**
  - +0.013 on lookup-right tokens: B's attention is still learning to copy.
  - +0.006 on lookup-wrong tokens: B is learning to refuse.
- **Any forecast fits these three points:**
  - linear: crosses zero at ≈ 265M tokens;
  - local power law (exponent −2.2): −0.003 at 600M;
  - geometric deceleration: ≈ −0.03.
  The evidence advisor's [−0.03, −0.01] does not follow.
- **"20% of tokens" is a lower bound** (G_VR mid-anneal vs B annealed).
- `train.bin` has 420M tokens, so a 600M run is 1.43 epochs.

**Fix:**
- Train one WSD run per arm (B and G_VR) to 600M, with decay branches at 200M, 400M and 600M, each scored on the 500 windows. That gives three schedule-matched points for about 1.3× the cost of one 600M run.
- Cheaper first step: full-cosine B and G_VR at 100M (≈ 40 H100-min). With the 200M pair, fit α in gap ∝ tokens^−α and pre-register the 600M value from it.
- Use fresh tokens or disclose the epochs.

## 2. The conflict-probe gains are not credible yet. **Major**
- **The test set was used for exploration.** `zprobe.py` found the cos/rp16 signal on windows 0–39. `zfit2.py` tests on windows 0–47, an 83% overlap.
- **Multiple comparisons:** six variants, same 48 windows, best reported.
- **The CIs are too narrow:** no fit-seed variance; the 48 windows are contiguous and atypical (ref 1.788 vs 1.690).
- **The M2a "pre-registration" brackets the in-sample result** (−0.009 to −0.014 around −0.011), so it tests nothing.
- **The fits touch the headline set.** val_code has only 558 windows, and the headline eval is windows 0–499. The fit windows 300–459 lie inside it, and E1 (fit 300–557) would contaminate 300–499.
- **Domain confound:**
  - The pointer was co-trained on 70% prose, but the refit is code-only and never tested on prose.
  - 124 table parameters can learn code-specific priors. The "g" control (+0.0003) rules out only a global recalibration.

**Fix:** fit on unseen mixed `train.bin` windows and test on the 500 code **and** prose windows. Use ≥ 2 fit seeds and paired variant-vs-variant CIs. Freeze the predictions first.

## 3. "Random keys beat trained keys" is an implementation mismatch. **Major**
- **The trained arm is not the M2b spec:**
  - W_k is a fresh Gaussian, not R16.
  - The score is an un-normalised bilinear q·k/4 with W_q = 0 at lr 1e-3.
  - rp16 instead gets a normalised cosine with a per-group weight at lr 1e-2.
- **The "overfit" evidence is not a fit-set loss.** `fit_nll_end` is the loss on the last 4-window minibatch, including the listwise term.
- No leakage found (h_{src−1} is causal). Undertrained vs overfit is undetermined.

**Fix:** initialise W_k = R16 in cosine form, so that trained ≡ rp at step 0. Log the full fit and test loss.

## 4. 81 B/token barely moves memory but changes the method. **Minor**
- **Memory:** 32K 5.5 → ≈ 6.0 MiB (≈ 1/11); 4K +0.06 MiB.
- **Method:**
  - State keys need a backbone forward at write time. That kills directions §4, a table over text the backbone never read.
  - ICE-K becomes top-K int8 attention, so the fair baselines are int8/int4 KV in B (1,024 / 512 B/token) and G_MQA.

**Fix:** say so; add a quantised-KV B row.

## 5. The decomposition is exact and meaningful, but "the gain is copying" is false for prose. **Major**
- **Arithmetic checked:** Σ frac × Δloss reproduces every total; fractions are identical across arms.
- **It is not circular.** As a null, G's own training gain (steps 1000→1500) puts 45% on lookup-right tokens, matching their 46% base rate. Attention puts 86–88% there. On prose: 17% vs 93%.
- **"ICE recovers 133%" is near-tautological,** because the bucket means "s\* is right". The informative bucket is lookup-wrong, where G_VR is +0.014 vs B.
- **Prose B→G_VR (−0.060):** 44% is on lookup-right, 37% on no-match and 19% on lookup-wrong. "The gain is copying" holds for code only.

**Fix:** report the null, add the rarity filter, split the prose claim.

## 6. "Division of labour" is probably real, but likely loss reweighting. **Major**
- **It replicates on no-match code loss:** G_V 2.512 and G_VR 2.511 / 2.510 (seeds 1 and 2), vs G 2.560. B's seed spread there is 0.003.
- **A simpler mechanism:** the V arms train on mixture NLL + 0.5·CE(backbone). On no-match tokens the mixture is exactly the backbone, so those tokens get weight 1.5, against ≤ ~1 elsewhere.
- **Against "freed capacity":** G_C, whose lookup also copies, is *worse* than G on no-match (2.579).
- G has one seed.

**Fix:** train G seed 2 (≈ $3), plus a G with CE reweighted by bucket and no ICE. If the reweighted G reaches ≈ 2.51, claim reweighting, not capacity.

## 7. Seed noise depends on the metric. **Minor**
- ICE-off code: 2.042 vs 2.071; ICE-off lookup-right: 1.042 vs 1.105. That is ≈ 8× the 0.004 headline spread, so reliance is 0.14–0.17.

**Fix:** drop "10σ" and report ICE-off with both seeds.

## 8. Memory claims. **Major**
- **1/12 pairs a byte ratio with unmeasured quality.** There is no 32K loss for any arm: B uses RoPE θ = 10⁴ and was trained at 4K, and a hint at 16–32K showed copyable +0.09 for G_VR.
- **At 4K the ratio is about 1/3, and mostly reflects the backbone choice:**
  - Both arms pay the fp32 GDN state.
  - ICE is ≈ 0.5 of G_VR's 3.5–3.9 MiB.
  - With a bf16 state the ratio would be ≈ 1/4.6.
- **ICE bytes are quoted three ways:** 64.7, 65 and 69 B/token. All are modelled; the index has never been measured.

**Fix:** measure (the evidence advisor's E2) and state the precision of every term. Lead with 1/32 per token and ≈ 1/3 at 4K.

## 9. Kaggle notebook. **Major**
1. **T4 fp32 vs H100 bf16.** The run-3 check is an absolute "1.705–1.735 (H100 scale)".
   - Add a precision gate: T4 − H100 for B, G and G_VR on the same 500 windows must agree within 0.003 across arms. Otherwise, never mix T4 and H100 numbers.
2. **The fit windows were seen in training.** The `fit_scorer` seed-7 permutation spans all of `train.bin`, and the base runs saw ≈ 48% of it. ℓ is over-confident there, so the scorer over-trusts the backbone. Restrict the fits to unseen windows.
3. **The plans disagree:**
   - The notebook fits 20M mixed tokens; the conflicts proposal's E2 assumes 12M code tokens.
   - The run-2 predictions differ (−0.003 to −0.010 vs −0.004 to −0.008).
   - Freeze one version.
4. **Line completion:** 300 items (≈ 70 repeats), clustered: bootstrap by window. Headline reuse/novel; repeat is near-trivial for ICE.
5. **Runtime risk:** fla on T4 may fall back to the PyTorch kernel. `MAX_HOURS` can then silently skip runs 3–4. Time one eval job first, and move `fit_Gbb_v2` up.
