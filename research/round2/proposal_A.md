# Proposal A: ICE-V2, the smallest mechanistic fix (lens: measured failures)

## 0. Thesis
The diagnostics name the failures in the scorer and the gradient path, not in the candidates (VR coverage is 87–100%). So: keep VR's 34 slots and the 64.7 B/token table, and change only four things. (1) Add listwise slot features that are computed from the candidate set at query time. (2) Constrain γ ≥ 0. (3) Add a listwise slot CE. (4) Detach the backbone from the mixture and train it with CE at weight 1. Change (4) has a consequence that sets the plan: **the backbone's training becomes identical to G's, so ICE-V2 = frozen G + a post-hoc scorer.** No new 140M run is needed, and every G seed gives a V2 seed for about $0.7.

## 1. ICE-V2 (equations, shapes)
Notation:
- Candidates tok, ln, dist [B,T,K] as now (K = 34).
- New int8 features [B,T,K], each computed in torch from tok/ln plus the source positions:
  - **Δ_k = clip(len_k − max_{j valid, c_j≠c_k} len_j, −8, +8)**: the match margin over the best *competing* token. A 19th bucket means "no competitor".
  - **n_k = clip(#{j valid: c_j = c_k}, 1, 6)**: the cross-slot agreement count.
  - **ν_k = 1[i_k > i_{s*}]**: the slot's source is newer than the longest-match source. This is the "latest write" signal for override_diff.
- g(k) ∈ {last, second, chain, anchor} is the slot group. Relative embeddings are per group, which keeps the parameter count tiny: 4×(19+6+2) = 108 parameters.

Score:

  s_k = b[src_k] + L[src_k,len_k] + D[dist_k] + softplus(γ̃[src_k])·ℓ_k + R_g[Δ_k] + A_g[n_k] + N_g[ν_k]
  s_0 = wᵀRMSNorm(sg(h_t)) + c₀      (β·max ℓ dropped: inert, std 0.16)
  log π = log_softmax([s_0, s_1..K]), with invalid slots set to −∞

- ℓ = sg(log_softmax(z)), clipped at −20, as now.
- softplus(γ̃) ≥ 0 removes the measured −0.17 on o8_last. That negative γ gives the stale slot +0.5 nats and the correct slot −1.
- Init: γ̃ = softplus⁻¹(0.3), so γ starts at 0.3. The new embeddings start at 0, so the init equals today's scorer.

Loss per token, with the gradients routed:

  L = CE(z, y)                                                  → backbone only (weight 1, not 0.5)
    − log( π₀·sg(p_bb(y)) + Σ_{k: c_k=y} π_k )                  → pointer params only
    + λ·1[∃k: c_k=y]·( −log Σ_{k: c_k=y} softmax_{1..K}(s)_k )  → pointer params only (λ = 0.25)

Why each term:
- **Listwise term.** It trains the ranking *within* the copy, and that gradient is not diluted by π₀. Today, at copy_diff's π₀ = 0.93, the slot ranking gets almost no gradient. This term targets the measured split within the copy, where only 50–67% of the copy mass in the middle bins lands on y.
- **Relative features.** They make "longest match wins" a single weight, R_chain[Δ>0]. The flat absolute L table cannot express it, and that rule scores 97–98% on conflict_g2/g5.
- **ν.** It lets an anchor slot beat a stale exact match.
- **Detach.** It removes laziness by construction: ICE-off is exactly G.

**What I reject.**
- **Set-attention over the slots.** It adds expressivity, not training signal. If natural code does not reward "longest/newest wins", attention will not learn it either. Δ/n/ν already express those rules linearly.
- **Input injection (G_C_in).** It breaks "ICE-off = G". Its +0.86 on increment is a +1 template prior, not recall.
- **An h·E[c_k] content term.** It is a [B,T,K,d] gather, and ℓ already carries that information (anchor AUROC 0.98).

**Memory.** 0 extra cache bytes/token.
- Δ, n and ν are functions of the current candidate set. At decode time they are computed from positions the ICE index already holds: one K×K compare per step.
- In training only, the batch gains ice_v_pos as int32 [K,T]. That is 136 B/token of loader traffic, not cache.
- Total new parameters ≈ 110 + 34.

## 2. Files
- `ice_features.py`: `_candidates` also returns `ice_v_pos` (i_k, −1 = empty) and `s_star_pos`, but only when `cand="vr2"`. The old keys are unchanged, so the old arms stay bit-identical.
- `model.py`:
  - In `ICEPointer`, add a `v2` flag: the R/A/N embeddings, softplus γ, no β, and a `_rel_feats(tok, ln, pos)` method that reuses the existing [B,T,K,K] `same` mask.
  - `loss_sum` gets `detach_bb` and `list_w`.
  - In `ICELM._loss_sum`, the aux weight becomes 1 under v2.
  - `mix` is unchanged apart from the features.
- `config.py`: add `ice_v2`, `ice_v_detach`, `ice_v_list` (defaults off), and the arm `G_V2`.
- `ICEIndex` (incremental decode) computes the same features.
- New script `scripts/fit_pointer.py`:
  - freezes any checkpoint, runs the backbone forward without grad in bf16, and fits only `ice_ptr`;
  - then calls the existing `evaluate.py` and `diagnose_pointer.py` paths.
- Tests (`tests_cpu`):
  - brute-force Δ/n/ν against the torch version;
  - the leakage perturbation test;
  - keep=0 bit-equality;
  - logsumexp(out) = 0;
  - old-arm checkpoints reproduce their losses bit-for-bit;
  - with v2, the backbone's gradient equals the gradient of plain CE (to fp32 tolerance).

## 3. Cheap tests, in order (total ≈ $6)
1. **T0, CPU, $0, 1 h: does natural code support the rules?** On 300 val_code and train windows, from features only:
   - (a) where Δ_{s*} < 0 (a different token has a strictly longer match), P(longest-token = y) vs P(s* = y);
   - (b) on anchor slots with ν = 1, P(c_k = y) vs anchor precision overall (0.11).
2. **T1, $0.3, no fitting: an oracle-ish edit.** Add R_chain[Δ>0] = +4 and clip γ at 0 in the existing G_VR scorer, then rerun CodeRecall and the loss splits. This bounds the gain and shows its cost on real code.
3. **T2, 6 fits × ~10 min ≈ $5.** Each fit uses 20M train tokens and AdamW at lr 3e-3. The ablation ladder on frozen **G** is (i) the old VR scorer (this is the planned D4), (ii) + γ ≥ 0, (iii) + Δ/n/ν, (iv) + listwise term = **V2**. Then V2 on frozen **G_VR's backbone**, which separates the scorer fix from detaching. Then V2 on frozen **B**, which tests the 7:1 story.
4. **T3, $0.7.** Repeat on G seed 2 when it lands. That gives the seed CI.

## 4. Pre-registered predictions ("G+V2" = frozen G t200 + the fitted V2 scorer)

| # | Metric | Prediction | Kill / decision |
|---|---|---|---|
| P0 | T0(a): P(longest=y) − P(s*=y) on val_code | ≥ +0.10 | < 0: natural data rewards recency; drop the conflict_g claim (the feature cannot be learned without synthetic mixing) |
| P1 | conflict_g2 / g5, deciding digit, minus B | ≥ +0.50 on both (today −0.07 / −0.10) | < +0.20 on both: the relative features failed; stop V2 |
| P2 | override_diff minus B | ≥ +0.15 | < +0.05: drop ν |
| P3 | copy_diff minus B | ≤ +0.05 (not fixed; the sentinel reads h, and h has no clue) | none; reported as unsolved |
| P4 | conflict_g1 / copy_same accuracy | ≥ 0.55 / ≥ 0.35 (G_VR 0.39 / 0.29; still below G_C's 0.99) | below G_VR: the listwise term hurts the exact copy |
| P5 | ICE-off code loss | = G (1.899) by construction | any difference means a bug |
| P6 | code loss, G+V2 | 1.705–1.735: worse than G_VR (1.690), because the backbone no longer specialises; about B (1.730) | > 1.750: detaching costs too much → ship "V2-joint" (features + γ≥0 + listwise, no detach) and report laziness as the price |
| P7 | follow-wrong on code / lk_wrong NLL | ≤ 0.28 / ≤ G + 0.02 (3.256) | > 0.32: the near-tie over-copying is NLL-optimal hedging; report it and do not chase it |
| P8 | B+V2 vs B, code / copy>512 | −0.010 to −0.025 / ≤ −0.08 | < 0.005 overall: ICE is an attention-free-only tool; say so |
| P9 | V2 on the G_VR backbone vs G_VR, conflict_g2/g5 | ≥ +0.40 | if this passes and P1 fails, detaching (not the features) is what hurts |

`increment` and `track` are out of scope: a pointer cannot add 1.

## 5. What this buys for the six weaknesses
1. Over-copying: P7 (better discrimination only).
2. Laziness: P5 (removed exactly).
3. Structured conflicts: P1 and P2, with copy_diff conceded.
4. Exact copy: P4 (closes part of the gap).
5. The 7:1 story: P8 (a frozen-hybrid bolt-on, and the same script runs on Qwen3.5-0.8B).
6. Evidence: seeds are free via T3, and the bolt-on is now the method itself rather than an extra arm.
