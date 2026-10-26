# Decision: ICE-V/VR in the low-attention regime

## 1. Verdict
**ICE-V(R): the lookup proposes, a lookup-blind backbone decides.** It is a log-space pointer-sentinel over a symbolic candidate set (V: the exact successors; VR: plus a deep suffix chain and identifier-anchor spans). It is tested where ICE still matters, which is **0 attention layers**.

**Converged:** P1's lookup-blind verifier is the core; P3's A arm and the measured benchmark are needed; P2's coverage gate runs regardless.

**New data overrides the 7:1 framing.** At matched steps, C−B shrinks 0.27 → 0.025 (79M → 157M tokens), and the follow-wrong excess shrinks 0.16 → 0.04. Any 7:1 comparison at 400M will sit under the single-seed noise floor (about 0.01–0.02).

**I overrule the faculty critic** on F (15:1) as priority: one attention layer can probably still copy within 4K, repeating the vanishing gap. Also on Route-lite as "diagnostic only": the user wants beyond-exact, and VR shares V's scorer, so it is a one-variable test for ~0.5 day.

**I overrule P1 and P2:** no V or V_nover at 7:1; no q·E content term (V and VR scorers stay identical); max ℓ instead of entropy; a post-hoc pointer on a frozen backbone instead of the AUROC gate.

## 2. Mechanism
**Candidates.** Each candidate is a token ID c_k = x[i_k] with i_k ≤ t (asserted in the builder). Features are int8.

| Set | K | Slots | Extra B/token |
|---|---|---|---|
| **V** | 12 | Order j ∈ {1,2,3,4,6,8} × occurrence o ∈ {last, second}, with successor x[p+1], `ice_dist2`, and the exact match length of each occurrence (capped at 32) | 0 (`second` is already stored): ~41 B/token |
| **VR** | V ∪ 22 = 34 | (a) Chain slots 3–8 of the longest-matching order, from the per-order `prev` pointers. Each occurrence has its own match length, so the true-longest occurrence can beat the most recent one (conflict_g5). (b) Anchors: the 2 most recent alphabetic tokens (vocabulary mask, not rarity) in x[t−7..t]. For each, the latest earlier occurrence q under the order-1 and order-2 key, giving the span x[q+1..q+4]. Every slot with q+i > t is masked. | +24 (prev pointers): ~65 B/token |

B's KV is 2,048 B/token; A's is 4,096.

**Scoring.** z [B,T,V] are backbone logits; the backbone never sees ICE. Let ls = log_softmax(z) and ℓ = ls.detach().

- s_k = b[src_k] + L[src_k, len_k] + D[dist_k] + γ[src_k]·clip(ℓ[c_k], −20, 0)
- s_0 = wᵀRMSNorm(h_t) + β·max_v ℓ + c₀
- log π = log_softmax([s_0, s_1..K]), with invalid slots set to −∞.

**Loss, computed in log space inside the 1024-chunk `_loss_sum`:**
- −log p(y) = −logaddexp(log π₀ + ls[y], logsumexp_{k: c_k = y} log π_k)
- L = that NLL + 0.5·CE(z, y)
- ICE-dropout is 0. The auxiliary CE trains the ICE-off path instead.

**Eval paths.**
- The same `_mix` function builds the full log-mixture in `_logits`: out = log π₀ + ls, then out[c_k] = logaddexp(out[c_k], log π_k) applied sequentially. This handles duplicate candidates.
- `eval_loss_splits`, `score_answers`, CodeRecall and the benchmark all go through it.
- keep=0 masks every slot, so the output equals ls exactly. Unit test: bitwise equality, and logsumexp(out) = 0.

**Leakage tests.** Perturb x[200:] and assert bit-identical log-probs at positions ≤ 199; check the incremental `ICEIndex` against batch candidate features.

**Beyond-4K evaluation.** Clamp the distance bucket to the largest bucket seen in training (12). Buckets 13–17 are untrained.

## 3. Work plan
**Day 0, in parallel.**
- Finish B, C and C_notrust to 400M; every comparison is at the final checkpoint.
- Remove the `max(1, …)` attention floor in `layer_plan` and `_evenly_spaced` for `ratio=0` (5 lines, plus a test).
- **Launch Round 1 immediately; it needs no new code.**

Free checks:

| # | Check | Threshold and consequence |
|---|---|---|
| D1 | Copyable code loss at positions 4–8K, 8–16K and 16–32K on long held-out code, plus CodeRecall gaps 6K/12K/24K (600 items per category), for B, C and C_notrust | If B degrades beyond 8K by < 0.10 nats against its in-4K value, drop the long-context claim |
| D2 | Training efficiency: tokens until C reaches B's loss, read from the existing eval logs | Reported as a curve; no claim if the saving is < 10% |
| D3 | VR coverage gate (CPU): share of correct deciding tokens inside s\* / V / VR, per CodeRecall category and on real code | Train VR only if VR ≥ V + 30 points on at least 2 of {override_diff, copy_diff, conflict_g5} |
| D4 | Post-hoc pointer: fit V's ~1.3K-parameter scorer on frozen G (once Round 1 finishes) and on frozen B | Go for G_V if the copyable >512 gain is ≥ 0.20 nats with lk_wrong +≤ 0.05. If the gain is < 0.10, run G_V+in instead |

**Arms.** One seed each, 400M tokens, same data order. G = 16 GDN layers and 0 attention.

| Round | Arm | Purpose |
|---|---|---|
| R1 | **G** | Zero-KV floor |
| R1 | **G_C** | G plus current ICE: does an exact lookup replace attention? |
| R1 | **A** | Upper reference |
| R2 | **G_V** | Lookup-blind verifier |
| R2 | **G_VR** | Beyond-exact candidates; needs D3 |
| R3 (conditional, ≤ 3) | G_V+in | If D4 < 0.10 or G_V keeps < 50% of G_C's gain |
| R3 | G_V_nover (γ = β = 0) | If G_V passes |
| R3 | B15 / F pair | Only if budget remains |

**Pre-registered predictions.**

- **G_C vs G:** code-all −≥0.05; copyable >512 −≥0.30 (G_C ≤ midpoint(G, B) at 41 vs 2,048 B/token); CodeRecall copy_same, conflict_g1 ≥ +0.30. **Kill identity claim** if copyable gain < 0.10.
- **G_V vs G_C:** follow-wrong ≤ −0.05; lk_wrong ≤ −0.10; keeps ≥75% of G_C's copyable gain; C_off(G_V) ≤ G + 0.03 (C_off(G_C) expected ≫ G). **Kill** if follow-wrong improves < 0.03 and lk_wrong < 0.05.
- **G_VR vs G_V:** ≥ +0.10 deciding-digit (paired bootstrap CI excl. 0) on ≥1 of conflict_g5, override_diff, copy_diff; code loss and follow-wrong within +0.02. Otherwise report "beyond exact: negative", with D3 separating router from selection failure.
- Beyond 4K (D1 rerun on G, G_C, G_V and A): the ICE arms lose ≤ half as much as A and B.

**Measure** for every arm: bytes/token (KV + GDN state + table) and tokens/s at 4K/32K.

## 4. The "beyond exact" component
**Route-lite candidates** (deep suffix chain with per-occurrence match length, plus anchor spans) are tokens only and store no hidden vectors. They are trained in R2 as G_VR, gated by D3. Full ICE-Route with 64-d stored vectors is deferred to future work.

## 5. Novelty claim
"An exact token-ID lookup at ~41–65 B/token can stand in for attention layers in a recurrent LM, but only if trust comes from a backbone that never sees the lookup; statistics of the lookup itself (agree/rel) carry no usable signal. We score symbolic candidates beyond the most recent exact successor (true-longest occurrence, identifier anchors) with that blind backbone through a log-space pointer-sentinel mixture, so ICE-off is the backbone itself."

Closest prior work to cite:
Pointer Sentinel (1609.07843), adaptive kNN-LM (2109.04212), Akyürek n-gram heads (2401.12973), Infini-gram (2401.17377), HOLA (2607.02303); also pointer-generator, Engram, LLMA, RetrievalAttention.

## 6. What NOT to do
- **New 7:1 arms** (V and V_nover at 7:1, B_pm, C_in/C_out training runs): the effects are now below noise.
- **Speculative-decoding headline**: orthogonal to the reviewers' concerns, rewards blind copying, prior art (LLMA, PLD, ReplaySSM).
- **Full ICE-Route**: a stored per-token vector is compressed sparse KV, takes 3–4 days, and carries a leakage trap.
- **Run-length trust head**: its label 1[r≥1] = 1[s\*=y].
- **More trust buckets; synthetic corrupted lookups**: the missing piece is independent information, not more negative examples.
- **Equivalence claims at one seed** ("F ≈ A").

**Top risks:**
1. G_C's gain is input-side, so output-only G_V loses it (D4 detects; G_V+in is the fallback).
2. Verifier circularity: a blind backbone is weakest on far-back tokens (D4 is stratified on copyable >512).
3. Silent bugs: anchor slots past t, eval paths scoring p_LM, untrained distance buckets beyond 4K (asserts, perturbation test, keep=0 test, clamp).

## 7. Slide: next steps
1. At 7:1, attention learns to copy (the C−B gap shrinks 0.27 → 0.03), so we test ICE where attention is absent: 0-attention GDN ± ICE vs A/B, beyond 4K, in long-gap CodeRecall.
2. ICE-V: the lookup proposes candidates and a lookup-blind backbone decides (log-space pointer-sentinel), to cut wrong-copy following.
3. Beyond exact: true-longest occurrences plus identifier anchors at ~65 B/token (no KV), targeting stale values and "which object" conflicts.
