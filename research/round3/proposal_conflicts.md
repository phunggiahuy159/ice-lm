# Proposal: conflicts and wrong lookups (round 3, 27 Oct 2026)

**New CPU evidence.** Two diagnostics on the frozen G_VR checkpoint, Mac CPU, about 15 minutes, not pre-registered. Scripts and outputs are in `round3/conflict_probe/`.
- `zprobe.py`: 40 held-out val_code windows, no training.
- `zfit2.py`: a post-hoc refit on val_code windows 300–459 (0.65M tokens), tested on windows 0–47 (197K tokens),
  with paired window-bootstrap CIs. The test subset is harder than the 500-window set (ref 1.788 vs 1.690), so read
  the deltas, not the levels.
- Sanity check: follow-wrong for ref is 0.331, against the 0.34 reported.
- **S** = s\* wrong, ≥ 2 distinct candidates and y in the set. S is 12.5% of code tokens.

## 1. Why the scorer fails

**(a) On S, the lookup's own evidence points away from the answer, by construction.** s\* is the longest and most
recent match. So whenever s\* is wrong, the correct occurrence slot ties s\*'s length 58% of the time, is shorter 38%
of the time, and is longer only 5% (F1 tables, 289K hit slots).
- Token-only scorers are therefore at chance.
- F2's best listwise fit is a convex optimum. It scores 0.169 rank-1 given y in the set, against 0.137 for a uniform
  pick, and identifier overlap scores 0.126.
- "No rule beats 0.07" is a property of the evidence, not of the tuning.

**(b) The backbone knows, and the verifier throws it away.** Token-level rank-1 among the distinct candidates:

| scorer | rank-1 on S | rank-1 on all ambiguous, y in set |
|---|---|---|
| uniform | 0.137 | 0.154 |
| G_VR pointer (b + L + D + γℓ) | 0.406 | **0.857** |
| backbone ℓ alone (0 B) | **0.645** | 0.822 |
| cos(h_t, h_{i_k−1}), final layer, untrained | 0.447 | 0.835 |
| same, 16-d random int8 projection | 0.398 | 0.762 |

- The learned priors are right on consensus positions and wrong on conflicts.
- With γ ≤ 0.61 (−0.17 on o8_last), the scorer cannot tell which regime it is in.
- Unconditionally, ℓ alone gets 0.22 of s\*-wrong positions right, the pointer 0.14, and the oracle 0.34.

**(c) Selection is worth top-1, not loss.** The sentinel already passes p_bb(y): NLL on S is 1.43, against 4.14 when
y is not in the set. Refitting 0-byte V2 features moves rank-1 on S from 0.41 to 0.49, but NLL on S moves by only
−0.015.

**(d) CodeRecall's conflicts are a different problem.**
- **B is at chance** (0.07–0.10) on override_diff, conflict_g2/g5 and copy_diff. Even B's backbone has a flat ℓ
  there: P[ℓ(y) > ℓ(s\*)] = 0.41–0.49.
- On override_diff and conflict_g2/g5, ICE drops *below* chance (0.000–0.007) because it follows s\*: follow
  rate 0.66–1.0, π0 0.14–0.54.
- conflict_g2/g5 can be solved by Δ (longest_occ 0.91/0.85). override_diff and copy_diff need refusal, not
  selection.

## 2. Mechanisms

All refits run on the frozen G_VR and start from its trained pointer. Rows below M1 add to M1. ΔNLL is against
G_VR as is.

| variant | bytes/token | added parameters | ΔNLL code [95% CI] | rank-1 on S | follow-wrong | top-1, s\* wrong |
|---|---|---|---|---|---|---|
| ref (G_VR) | 65 | 0 | 0 | 0.412 | 0.331 | 0.343 |
| g: extra γ per group + sentinel offset | 65 | 5 | +0.0003 | 0.449 | 0.327 | 0.344 |
| **M1** v2: + R/A/N/Q tables | 65 | +124 | **−0.0056** [−0.0069, −0.0043] | 0.494 | 0.319 | 0.347 |
| v2gate: + γ(h_t) | 65 | +3,072 | +0.0005 | 0.521 | 0.312 | 0.348 |
| **M2a** v2rp: + w_g·cos(rp16(h_t), rp16(h_{i_k−1})) | **81** | +4 | **−0.0110** [−0.0124, −0.0095] | 0.500 | 0.314 | 0.349 |
| **M2b** v2keys: + q_t·k_{i_k−1}, 16-d int8, trained | **81** | +24,576 | −0.0089 [−0.0109, −0.0068] | **0.531** | **0.301** | 0.353 |
| v2cos: + raw full-state cosine (reference only) | ~3,000 (two fp16 states) | +8 | −0.0156 [−0.0172, −0.0139] | 0.500 | 0.312 | 0.351 |

What the table says:
- **State match is the strongest new signal, and it works mostly through copy calibration, not selection.**
  - Over V2, rp16 adds −0.0054 nats, but rank-1 on S stays flat at 0.50.
  - The gain on S (−0.021 × 12.5% ≈ −0.003) is under half of it. The rest falls where s\* is right.
- **Trained keys overfit at 0.65M tokens.** On the same last fit batch their loss is 1.799 against rp's 1.828,
  yet they test worse.
  Selection is best with them (0.531, follow 0.301), so they need the 12M-token T4 run.
- **Gated γ is killed:** it overfits and gains nothing.

### M1: the V2-nat refit (0 B)
The spec is decision2 §2, unchanged: it is run 2 of the notebook.

**My pre-registration (300 windows):**
- code −0.004 to −0.008;
- rank-1 on S 0.46–0.52;
- follow-wrong 0.31–0.32, so the gap to B's 0.24 is not closed;
- top-1 +0.1 to +0.4 points;
- conflict_g2/g5 ≥ +0.3 over G_VR. Natural code teaches R: chain slots with Δ ≥ +3 have precision 0.91–0.98.
- override_diff and copy_diff within ±0.05.

**Kill:** ΔNLL > −0.003.

### M2: source-state keys (ICE-K), 16 B/token
- **Spec.**
  - k_i = Q8(tanh(W_k·RMSNorm(h_i))), int8, written when token i is processed.
  - s_k += τ·(W_q·RMSNorm(h_t))·k_{i_k−1}/4.
  - Two forms:
    - **(a) rp16:** W_k is a fixed seeded Gaussian, and the score is w_g·cos. 4 parameters, nothing to train in the
      key.
    - **(b) trained:** W_k is initialised from (a), W_q = 0. 24.6K parameters.
- **Constraints.**
  - The backbone never sees ICE: h is read, as w·h already is for the sentinel. Old arms are untouched.
  - The table grows from 65 to 81 B/token, still 3.2× below G_MQA's 256.
- **Compute.** About 25K MACs per token for the projections, plus 34 × 16 for the scores.
- **Pre-registration (T4, 12M train-code tokens, 300 test windows):**
  - (a) code −0.009 to −0.014 vs G_VR;
  - (b) ≤ (a) − 0.002;
  - rank-1 on S +0.02 to +0.05 over M1;
  - follow-wrong ≤ 0.30.
  - override_diff will **not** improve and may get worse: the query `config["timeout"] =` resembles the *stale*
    write's state, not `config.update(...)`'s.
- **Kill.**
  - (b) − M1 > −0.004, or (a) − M1 > −0.003: keys are not worth 16 B.
  - If (a) ≈ (b), report "the backbone state is already the key".
- **Change the go rule.** Run 6 asks for "+0.10 unconditional rank-1 on s\*-wrong code". That means 0.24 or more,
  which is beyond ℓ alone (0.22), so it is unreachable for any backbone-based selector. Replace it with "ΔNLL ≤
  −0.005 vs M1 **and** follow-wrong ≤ M1 − 0.01".

### M3: rejected or folded in
- **Scope or lexical candidates/features.** Identifier overlap is *below* chance on S (0.126 vs 0.137). It is
  confounded by recency, because the most recent occurrence shares the most context. Scope ids (1 B/token) would
  carry the same confound. At most an optional F2 variant (CPU, ~5 min).
- **Listwise training on mined natural conflicts.** F2 is a convex optimum, so reweighting cannot add information
  to token features. With ℓ or keys in the features, it is only a loss weight, already present as λ = 0.25.
- **Re-ranking with the backbone's own prediction.** This is γℓ plus Q, i.e. M1. The gap to ℓ alone (0.53 vs 0.645)
  is a regime-detection problem, and gating γ on h failed.
- **A backbone query attending over the K candidates' stored states** is M2 with dk = 16. Full states give only
  −0.005 more, for about 190× the bytes.

## 3. Experiments, ranked by expected value per GPU-hour

| # | experiment | where / time | expected | EV / GPU-h |
|---|---|---|---|---|
| E1 | Replicate `zfit2.py` at scale (test windows streamed): fit on val_code 300–557, test on 0–299; dk ∈ {8, 16}, rp vs trained | Mac CPU, ~35 min | firm CIs; picks dk | ∞ (0 GPU) |
| E2 | Notebook run 2 (M1) **+** M2a and M2b in `icek_probe.py`: ctrl = V2, plus an rp arm and Wk initialised from rp; 12M train-code tokens; CodeRecall and line-completion EM | T4, ~2 h (3 × 25 min forward plus evaluation) | −0.009 to −0.014 | ≈ 0.005 nats/h |
| E3 | Best of M2 on frozen G and frozen B | T4, ~1 h | G: largest gain. B: ≤ 0.004, because attention already does state matching | complementarity claim |
| E4 | G_VRK joint, only if E2 passes | H100 after 1 Nov, ~$3 | must beat post hoc by ≥ 0.005 | low |

E2 needs about 1 h of code: the rp arm, Wk init, a V2 control, `zfit2.py`'s top-1/follow metrics and the new go rule.

## 4. Insight: what an exact lookup can and cannot be

1. **It proposes and cannot dispose.** The lookup's evidence (match length, recency) is maximal at s\* by definition.
   Whenever s\* is wrong, that evidence is neutral or adverse for the right candidate in 95% of cases. Selection must
   come from outside the table: the model's prediction (ℓ, 0 B) or the model's state at the source (keys, 16 B).
2. **"The table stores, the backbone addresses."** ICE-K is attention restricted to the K exact-match candidates,
   with 16-byte int8 keys. That places ICE on a clean axis:
   - an n-gram cache matches tokens over all positions;
   - attention matches states over all positions;
   - ICE-K matches tokens, then states, over K candidates.
3. **Report three numbers, not one.**
   - **Coverage:** y in the set, 0.34 of s\*-wrong positions.
   - **Selection:** rank-1 given coverage, 0.41 → 0.50–0.53.
   - **Calibration:** the loss gain, which comes mostly from state match acting as *trust*.
4. **Stale values are a different axis.** Similarity retrieval (keys, and attention too) prefers the
   same-syntax *old* write. Neither the 140M hybrid nor ICE resolves "which version". The honest claim is that ICE
   turns chance into below-chance by confident copying, and the fix is refusal (π0), not better lookup.
5. **Paper framing.** "Exact lookup supplies content. Choosing among exact matches needs context the lookup cannot
   store, and the model supplies it through 0–16 bytes per token. Version tracking is unsolved by both mechanisms at
   this scale."

**Caveats.** One checkpoint; fit and test windows both from val_code (no train.bin); 48 test windows; no CodeRecall
in the CPU runs; not pre-registered. E1 and E2 make the numbers citable.
