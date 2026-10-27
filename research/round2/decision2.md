# Decision, round 2 (judge, 27 Oct 2026)

## 1. Verdict
**Kept from the consensus:**
- Frozen-backbone scorer fits come first (`scripts/fit_scorer.py` exists, with `--init_from`).
- Seeds; an equal-memory attention baseline; ICE-K only behind its probe.
- Qwen3.5 is cut.
- `increment` is never cited.

**Overruled:**
1. **"Longest match wins" is dead on real code.** On ambiguous VR positions of real code (`rule_oracle.py`), the rule scores 0.426 against s* 0.508. On s*-wrong positions every fixed rule is ≤ 0.071, while the oracle is 0.333.
   - The rule is expressible as one weight. Natural code will not teach it.
   - The headroom needs context-dependent scoring. One such feature already costs 0 bytes and was never tested as a rule: ℓ_k, the backbone's log-prob of each candidate (hit/miss AUROC 0.89–0.98).
   - Predictions for natural-code conflicts are now **null**.
2. **Laziness is specialisation.** The lookup is exact and always present, so an ICE-off penalty is like removing B's attention.
   - Detach is not *the* method. Detached V2 = frozen G + fitted scorer, which comes for free as an ablation.
   - Co-adaptation is worth about 0.02 nats.
   - So V2 is **both** fitted post hoc and (conditionally) trained jointly.
3. **The equal-bytes baseline is G_MQA, not G_SWA.**
   - G_SWA's bytes match only at 32K. At 4K it is a 15:1 hybrid, and it needs about 5 h of code.
   - G_MQA has 15 GDN layers + 1 full-attention layer with one KV head. That is 256 B/token, 4× ICE's 64.7, **at every length**.
   - It is a pure `ARMS` entry (`layout="layer", ratio=15, n_kv_heads=1`) and doubles as the missing 15:1 arm. If ICE beats 4× the bytes, the claim holds a fortiori.
4. **Seed 3 is conditional.** The gap is 0.04 and seed noise is ≈ 0.01, so two agreeing seeds give 4σ.
5. **The fitter is gated.** Refitting the old VR scorer on G_VR's backbone must reproduce 1.690 ± 0.005.

## 2. ICE-V2 spec
**Unchanged:** the VR slots (K = 34) and the table (64.7 B/token).

**Groups** g ∈ {last 6, second 6, chain 6, anchor 16}.

**New in `ICEIndex`:** it also returns the source position i_k, which it already holds (about 10 lines). ℓ_k = log_softmax(z_t)[c_k] is available at decode.

**Features.** All are int8, need one K×K compare per step, and add 0 cache bytes.

| Feature | Definition | Embedding |
|---|---|---|
| **Δ_k** | clip(len_k − max{len_j : j valid occurrence slot, c_j ≠ c_k}, −8, 8), plus a "no competitor" bucket. Occurrence slots only, because an anchor's len is the anchor's match length. Cap stays 32; ties are reported. | R_g |
| **n_k** | clip(#{j valid : c_j = c_k}, 1, 6) | A_g |
| **r_k** | Recency rank of m(c_k) = max{i_j : c_j = c_k} among the distinct tokens, newest = 0, clipped to 0..3. This is "latest write", generalised. | N_g |
| **q_k** | Rank of ℓ_k among the distinct tokens, 0..2. It is the non-linear use of ℓ that γℓ cannot express. | Q_g |

**Score:**

  s_k = b[k] + L[k,len_k] + D[dist_k] + softplus(γ̃_k)·sg(ℓ_k) + R_g[Δ_k] + A_g[n_k] + N_g[r_k] + Q_g[q_k]
  s_0 = wᵀRMSNorm(h_t) + c₀      (β dropped)

The new tables start at 0, so V2 starts as the current scorer. That adds about 160 parameters.

**Loss:**

  L = −log(π₀·p_bb(y) + Σ_{c_k=y} π_k) + 0.25·1[y in the set, ≥ 2 distinct tokens]·(−log Σ_{c_k=y} softmax_{1..K}(s)_k)

**Gradient paths:**
- **Post hoc:** only `ice_ptr` trains, on 20M tokens.
- **Joint (G_VR2):** G_VR's paths: p_bb(y) and h attached, aux CE 0.5.

**Variants:**
- **V2-nat:** fitted on natural code only. R/A/N stay only if F2 says go.
- **V2+syn:**
  - 2% of the fit windows are synthetic conflict and override items. Their template families are disjoint from CodeRecall's: other keys, ≥ 3 writes, swapped order.
  - They are evaluated on CodeRecall plus a held-out set of perturbed templates.
  - They are always labelled "task-supervised".

## 3. Runs (≤ $30)
**Free first:**
- **F0:** seed-2 evaluation (running).
  - Gap ≤ −0.025 and \|gap₁ − gap₂\| ≤ 0.015 → "beats B (2 seeds)".
  - Gap in (−0.025, 0) → seed 3 is mandatory.
  - Gap ≥ 0 → the claim is "matches B".
- **F1 (CPU):** `rule_oracle` with longest over occurrence slots only, plus newest distinct token, plus per-group precision vs Δ/n/r.
- **F2 (CPU):** a token-only listwise logistic, trained on train code and scored on held-out code. Keep R/A/N only if they improve log-loss by ≥ 0.01 nats.

| # | Run | $ | Prediction / kill |
|---|---|---|---|
| 1 | Old VR scorer refit on the G_VR backbone | 0.6 | **No-go:** off by > 0.005 → fix the fitter |
| 2 | V2-nat on the G_VR backbone, 2 fit seeds | 1.2 | Code −0.003 to −0.010 vs G_VR. Follow-wrong ≤ 0.31. conflict_g1 ≥ 0.50. Conflicts within ±0.10 (null). Rank-1 on s*-wrong real code: ≈ 0.07 → ≥ 0.12. |
| 3 | Old scorer and V2-nat on frozen G (the detached method) | 1.2 | 1.705–1.735. ICE-off = 1.899 exactly. |
| 4 | V2-nat on frozen B | 0.6 | −0.005 to −0.02. Under 0.005 → "attention-free only". |
| 5 | V2+syn on the G_VR backbone | 0.7 | conflict_g2/g5 ≥ +0.5 vs B; override_diff ≥ +0.3. **Kill:** code cost > 0.005, or < 50% of the gain kept on held-out templates. |
| 6 | ICE-K probe on frozen G_VR | 1.0 | **Go only if** it beats run 2 by ≥ +0.10 rank-1 on s*-wrong code **and** by ≥ 0.005 on paired loss. |
| 7 | **G_MQA** at 200M, plus evaluation and benchmark | 3.2 | Code 1.74–1.78, so G_VR leads by ≥ 0.04. **Kill the equal-bytes headline** if G_MQA ≤ G_VR − 0.01: the next step is a 64 B/token attention layer, and the claim becomes complementarity (run 8). |
| 8 | V2-nat on frozen G_MQA | 0.6 | −0.01 to −0.03 |
| 9 | 8–32K evaluation of G, G_VR, B, G_MQA and the best fit | 1.5 | G_VR's lead on copy > 2K holds. Labelled "length generalisation". |
| 10 | One joint run: G_VRK if run 6 says go; else G_VR2 if run 2 gains ≥ 0.008 on code or ≥ +0.10 on a category | 3.0 | Must beat run 2's post-hoc gain, else the post-hoc fit is the method. |
| 11 | Seed 3 of B and G_VR, under the F0 rule | 6.0 | All 3 seeds negative, mean ≤ −0.025 |
| 12 | Stretch: Mamba2-370M vs Pythia-410M, post hoc, loss splits only | 3.0 | Premise: copy > 512 gap ≥ 0.10. ICE closes ≥ 50% of it. |
| | Subtotal / reserve | 22.6 / 7.4 | |

**Implementation (about 10 h):**

| Task | Time |
|---|---|
| Features in torch and in `ICEIndex`, with tests: brute force, decode = batch, keep = 0 bitwise, old-arm bit-equality | 4 h |
| `fit_scorer` flags | 1.5 h |
| G_MQA arm | 0.5 h |
| ICE-K probe | 2 h |
| Synthetic templates | 1 h |
| F1/F2 | 1 h |

## 4. Cut
- **Qwen3.5:** about 1.5 days for an expected ≈ 0, and run 4 answers it.
- **A at 200M:** it only moves the upper reference.
- **G_SWA and W = 700:** superseded by G_MQA. Revisit only if G_MQA comes within 0.01.
- **A's P1/P2:** falsified.
- **Detach as the headline.**
- **Multiple-choice CodeRecall for NeoX.**
- **Loss claims under 0.02 from one seed.**

## 5. Claims
**If it works:**
> On 140M models (2–3 seeds), a pure GDN model with an exact index and a ~2K-parameter verifier beats a 7:1 hybrid (code −0.04, prose −0.06). It also beats one full-context attention layer with 4× the bytes per token, at a measured 64.7 B/token.
>
> Fitted post hoc on a frozen GDN, the verifier recovers X% of the gain, and ICE off is exactly the baseline. Joint training adds Y.
>
> Natural code does not teach occurrence selection. With task supervision, the same 0-byte features resolve conflicts on held-out templates at ≤ 0.003 loss.
>
> It adds ≤ 0.02 to hybrids and does not solve paraphrase.

**Fallback:**
> A backbone-verified exact lookup lets an attention-free model match a 7:1 hybrid at 1/12 of the cache, mostly through verbatim copying.
>
> Choosing among conflicting occurrences needs context the lookup does not store. Fixed rules and natural-code training do not supply it; 16-byte keys [did / did not]. A single attention layer is [better / worse] at 4× the bytes.
