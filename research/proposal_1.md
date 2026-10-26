# Proposal 1: ICE-V, a backbone-verified candidate pointer

## Name
**ICE-V (Verified ICE)**: a pointer-sentinel mixture over the lookup's *candidate set*. The ICE-free backbone acts as the verifier.

## One-paragraph idea
We stop injecting the lookup into the residual stream and stop adding an unbounded logit bonus. ICE instead proposes up to 12 candidates: for each order, the successors of the two most recent occurrences. The backbone never sees ICE, and it scores each candidate with its own detached log-probability, as in speculative-decoding verification. A sentinel slot ("none of these, use the LM") competes with the candidates in a single softmax. The backbone is ICE-free, so its output *is* the C_off model, and we train it directly with an auxiliary loss on the same forward pass at no extra cost. The trust signal is now independent evidence (what the model predicts without the lookup), not another statistic about the lookup. agree/rel were statistics of that kind, and they did nothing (C ≈ C_notrust).

## Exact mechanism
**Features (`ice_features.py`, `inference.py`, `data.py`).** Add `ice_succ2[N,T]` = x[p2+1], the successor of the second-most-recent occurrence (−1 if none). `compute_ice_features` already computes it to derive `agree`. `ICEIndex` already stores `(last, second)` per key, so this costs **0 extra bytes/token**. Add the key to `ICE_KEYS`.

**Model (`model.py`, `config.py`).** New `ICEPointer` replaces `ICEOutput` when `ice_output="pointer"`. Set `ice_input=False` for the main arm. With z = backbone logits [B,T,V] and ℓ = log_softmax(z).detach():
- Candidates c_k for k = (order j, occurrence o), K = 2N = 12, invalid ones masked.
- Score: s_k = b[j,o] + L[j,len_j] + D[dist_j] + A[j,agree_j] + R[j,rel_j] + γ[j,o]·clip(ℓ[c_k], −20, 0).
- Sentinel: s_0 = wᵀRMSNorm(h_last) + β·H(softmax z).detach() + c.
- π = softmax([s_0, s_1..s_K]) (13-way). p_mix = π_0·softmax(z) + Σ_k π_k·onehot(c_k). Duplicate candidates sum, which gives a vote effect.
- `_logits` returns log p_mix (scatter_add on probabilities, then log). `evals.py` therefore works unchanged: argmax, CE, follow-rate. keep=0 masks every candidate, so p_mix = p_LM exactly.

**Loss.** L = CE(log p_mix, y) + α·CE(z, y), with α = 0.5 fixed in advance (not tuned). `_loss_sum` returns both terms per chunk. ICE-dropout stays at 0.2 so the comparison with C changes as little as possible.

New parameters: about d + 6·(11+18+4+5) + 24 ≈ 1.3K. That is negligible next to C's 6·d² input matrices. Roughly 150 lines of code in total.

## Why it should fix the measured failures
- **Follow-wrong / lookup-wrong loss.** In C the gate reads h_last, but h_last has already absorbed the injected successor, so it cannot independently judge the lookup. Gating on the LM's own uncertainty works for kNN-LM (Efficient kNN-LM, He et al. 2109.04212; Drozdov et al. 2210.15859). Speculative decoding rejects drafts the target model finds implausible (Leviathan et al. 2211.17192). Here we learn that acceptance rule jointly. A wrong copy now costs exactly −log(1−g) on top of the LM's loss, which is the pointer-sentinel property (Merity 1609.07843; See et al. 1704.04368).
- **Conflicts / overrides.** Two occurrences per order, across 6 orders, put competing values side by side as separate candidates rather than one s* plus a scalar `agree` flag (SuffixDecoding 2411.04975; Token Recycling 2408.08696). The backbone's ℓ can then choose between them.
- **C_off < B (laziness).** The greedy-modality literature (Wang et al. 1905.12681; Wu et al. 2202.05306) recommends a unimodal auxiliary loss. Our backbone is ICE-free by construction, so that loss is free.
- Why I do *not* lead with corrupted-lookup training (lit_trust's top pick): natural data already supplies "present but wrong" on more than half of the matched code positions. Synthetic corruption that cannot be told apart from real lookups only lowers the prior. The model is missing discriminating *information*, not negative examples.

## Memory / speed
- Table: unchanged, ~41 B/token (`second` is already stored).
- Activations: 12 gathers plus a 13-way softmax per token, under 0.1% of LM-head FLOPs.
- Training adds one fp32 [1024,V] exp/scatter per loss chunk, which is already checkpointed.
- The main arm also *removes* the 6·d² input projection.

## Experiment
**Step 0 (free, about 1 hour, gates the GPU spend).** On held-out code, fit a logistic regression for 1[s*=y] using (a) len/dist/agree/rel only, and (b) those plus the **existing B checkpoint's** log p(s*) and entropy. B is a proxy for the ICE-free backbone. If the AUROC gain from (b) over (a) is below 0.03, stop: there is no independent evidence to exploit. Also plot C's reliability diagram for p(s*), which tells us whether C is overconfident or simply information-limited.

**Arms** (400M tokens, same seed and data as B/C/C_notrust):
- **V**: pointer, output only, α = 0.5.
- **V_nover**: as V with γ = β = 0. Isolates the verifier.
- **V+in**: V plus C's input side. Tests whether the input side carries the gain and whether it brings back over-reliance.

**Metrics** (existing `eval_loss_splits` and CodeRecall): code all/copyable/lk_wrong loss, follow_wrong_lookup, C_off (ice_keep=0), CodeRecall per case, and prose loss.

**Pre-registered predictions for V vs C** (C → B values in brackets):
- follow-wrong ≤ 0.27 (0.356 → 0.196)
- lk_wrong loss ≤ 3.90 (4.014 → 3.786)
- copyable ≤ 1.40, which keeps ≥ 80% of C's gain (1.299 → 1.918)
- code all ≤ 2.20 (2.157)
- C_off code all ≤ 2.455 (B + 0.03)
- V_nover follow-wrong exceeds V's by ≥ 0.03.
- V+in: copyable better than V and follow-wrong worse than V.
- Hope, not a claim: CodeRecall override_diff and conflict_g5 ≥ 0 vs B.

**Kill criteria:**
- V follow-wrong > 0.32 **and** lk_wrong > 3.95: the mechanism does not fix over-reliance.
- V and V_nover within 0.01 on follow-wrong: verification adds nothing, and the result is just "pointer sentinel".
- V copyable > 1.50 with V+in not better: output-only verification costs too much.

Single seed: treat the C−C_notrust difference (about 0.01) as the noise floor.

## Novelty vs closest papers
- Pointer sentinel / neural cache (1609.07843, 1612.04426) use soft attention over stored hidden states (KV cost). ICE-V uses an exact token-ID table at ~41 B/token.
- Efficient kNN-LM (2109.04212) gates a fixed λ with LM entropy over a corpus datastore. ICE-V does *per-candidate* backbone verification inside an in-context candidate set, trained jointly.
- Prompt-lookup / REST (2311.08252) verify at inference without training.
- Engram (2601.07372) is a static table. Akyurek n-gram heads (2401.12973) are attention.
- The new claim: *an exact lookup is safe when the model that judges it never sees it*. In a GDN hybrid this makes C_off = backbone by construction.

## Biggest risk
C's gain may live mostly on the **input side**. C_in/C_out have never been run. If so, V loses copyable gains, and V+in brings back the over-reliance it was meant to remove. The step-0 gate and the V+in arm expose this within one round. A second risk: B-like backbones may simply not know when a far-back identifier is right. That is ICE's whole reason to exist, so ℓ may be uninformative exactly where copying matters.
