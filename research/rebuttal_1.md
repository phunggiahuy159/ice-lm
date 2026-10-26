# Rebuttal 1: ICE-V, revised

## Concessions
- **Novelty 3/10 is fair as written.** A pointer-sentinel mixture with an entropy-aware gate is textbook: See et al., Merity et al., adaptive kNN-LM (He et al., Drozdov et al.), LLMA 2304.04487, and the implicit-cache pointer LM 2009.13774. I withdraw "the gate" as the contribution.
- **"C_off = B by construction" was wrong.** Gradient still flows through π₀·softmax(z), so the backbone can still get lazy. C_off ≤ B+0.03 is a *test*, not a guarantee.
- **The verifier may be blind exactly where ICE matters.** That is the faculty's point about far-back identifiers. Step 0 is now reported on the **copyable >512 split**, not overall.
- **The baselines are at ~105M tokens.** Every comparison uses matched steps, and C is re-evaluated at 400M.
- **Without A, the identity claim is untestable.**

## Technical fixes
1. **Log-space training loss.** NLL = −logaddexp(log π₀ + ls[y], logsumexp_k(log π_k + 0/−∞ for [c_k = y])), where ls = log_softmax(z) and ls[y] is a gather. The full log-mixture is built only in `_logits` for eval: out = log π₀ + ls, then 12 sequential `out[c_k] = logaddexp(out[c_k], log π_k)`. Doing them sequentially handles duplicate candidates. Nothing underflows.
2. **Chunk memory.** Training holds only z and ls, about 2×524 MB at [4,1024,V] (≈ today's 1 GB). The same ls feeds the aux CE and the candidate features. I replace entropy with max(ls), a reduction over ls, so no softmax tensor is needed.
3. **Second-occurrence distance.** Add `ice_dist2` = bucket(t − p2). `ICEIndex` already stores `second`, so it costs 0 B/token.
4. **ICE-dropout.** In V the backbone never sees ICE, so dropout only starves the pointer. The α-weighted aux CE takes over its job (training the ICE-off path on every token). I set **dropout = 0 in V arms** and state this openly as a deliberate difference from C.
5. **Fallback if laziness persists.** V_det detaches z inside the mixture. The backbone then gets *exactly* B's loss, C_off = B holds truly by construction, and the pointer becomes a trained post-hoc verifier.

## Sharper novelty claim
The claim is a **measured separation principle**, not a mechanism: *trust in an exact lookup must come from a model that has not seen the lookup*. Three facts make it falsifiable:
- Lookup statistics already failed (C ≈ C_notrust).
- V vs V_nover tests whether lookup-blind verification works.
- V vs V+in tests whether letting the backbone see the lookup contaminates the verifier.

The setting is new: an exact in-context index as candidate generator, at 41 B/token, in a GDN hybrid. Honest novelty is ~4–5/10. The value is a clean causal finding plus a fix, not a new gate.

## Absorbed from the rivals
- **From P2: free CPU coverage analysis.** For each CodeRecall category, what fraction of deciding tokens lie in V's 12 candidates versus P2's 32 (chains plus anchors)? If anchors raise coverage substantially, I add **token-only anchor candidates** to V. That is P2's R_tok without stored vectors, and V's backbone-logprob scoring needs no per-token memory. Cases no candidate reaches become the "where copying is wrong" slide.
- **From P3: A and F arms, plus the measured benchmark** (bytes/token including the table, tokens/s at 4K/32K), without speculative decoding.

## Revised plan (≤6 arms)
- **Day 0 (CPU / existing checkpoints):**
  - step-0 AUROC gate on all code and on copyable >512;
  - coverage analysis;
  - C at 400M.
- **Round 1:** A, V, V_nover.
- **Round 2 (conditional):**
  - **V+in** if V copyable > 1.40;
  - **F_V** (15:1 + ICE-V) if V passes;
  - **V_det** if C_off > B+0.03.

The predictions and kill criteria are unchanged. One addition: if the AUROC gain on copyable >512 is < 0.02, I pre-declare that verification helps only near-range copies.

## The rivals' weakest points (fairly)
- **P2.** As written, slots q+1..q+4 can reach t+1. That is label leakage, and it would yield a spectacular, meaningless loss. More fundamentally, a learned 64-d vector per token read by softmax is a compressed sparse KV cache at 130–190 B/token. It gives up the "attention-free, token IDs only" identity and invites "why not top-32 sparse attention?" It also needs 3–4 days. Its coverage gate, though, is excellent, and I adopt it.
- **P3.** Speed does not address the reviewer's main concern, and greedy verification *rewards* over-reliance. Its run-length head reads h_last, which has already absorbed the injected successor. That is the same contamination that I argue made agree/rel inert, so its trust claim will likely reproduce C ≈ C_notrust. The A/F arms and the benchmark are its real contribution, and I take them.
