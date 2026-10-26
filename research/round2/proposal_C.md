# Proposal C: ICE-K, 16-byte write-time keys that let the index's candidates be told apart

## The diagnosis this targets
ICE-V rests on one premise: "the lookup proposes, a lookup-blind backbone decides". That premise fails exactly where it is needed most. On structured conflicts the backbone cannot recall the value, which is why the item is hard in the first place. So ℓ(correct) ≈ ℓ(wrong) (−3.57 vs −3.44 on override_diff), and the verifier has nothing to verify with.

The scorer also cannot route. In s_k = b[src] + L + D + γℓ, only the sentinel sees h_t. Every slot score is a context-free prior. The model cannot say "here I want the value after the identifier's latest write", or "the occurrence whose line had `name="dev"`".

The information that separates the candidates is the **context at each candidate's source position**: which server's line it was, whether it was a write, which key it belonged to. A GDN is good at summarising *local* context at the moment of writing, even though it cannot retrieve it 3K tokens later. Attention solves this with q·k over all positions at 2,048 B/token. ICE already supplies the 34 positions worth checking. All that is missing is a tiny key at each position.

**Method:** store a 16-byte quantized key per token, computed by the backbone at write time. Score only the ≤34 ICE candidates by q_t·k. There are no values, because the value is the candidate's token id, which the index already holds. This is sparse attention whose sparsity pattern comes from an exact index, with the value path removed.

## Mechanism (changes against G_VR)
- **Key**, written once at every position i: k_i = Q8(tanh(W_k·RMSNorm(h_i))) ∈ int8^16, with a straight-through fake-quant ×127 during training. h_i is the final hidden state at i, the state that predicted x[i+1].
- **Query**: q_t = W_q·RMSNorm(h_t).
- **Slot score.** For candidate c_k = x[i_k], the matching key is k_{i_k−1}. That key is the state that was "about to emit" the candidate, the same role q_t plays for x[t+1]; this is induction-head semantics.
  - s_k ← s_k^VR + τ·(q_t·k_{i_k−1})/(127·√16).
  - The sentinel is unchanged.
- **Init:** W_q = 0 and τ = 1. The arm starts exactly equal to G_VR.
- **Parameters:** 2 × 768 × 16 ≈ 24.6K.
- **Loss:** the existing mixture NLL + 0.5·CE, plus a listwise slot CE (λ = 0.2) on positions where some slot holds y and some other slot holds a different token. This uses the free label "which candidate was right", which the π0-diluted mixture gradient under-uses (diagnostics §2: in the 0–0.1 copy bin, y is in the set 38% of the time but receives 1.1% of the mass).
- **Backbone input is unchanged.** Keys are projections of h and the backbone never reads ICE, so keep = 0 still returns log_softmax(z) bitwise. Gradients from q·k do shape h (as w·h already does); that is intended.

**Code (~1 day):**
- `ice_features._candidates` already has `src` [K,T]. Under a new `cand="vrk"`, also return `ice_v_src = src`; old cand values are untouched.
- `ICEPointer` gets `use_keys`: a [B,T,16] key tensor and a gather at src−1 into [B,K,T,16], which is 17.8M floats for micro-batch 8 and fits in the 1024-chunk.
- `ICEIndex(cand="vrk")` keeps a [T,16] int8 buffer. `step(h_t)` appends k_t and returns the source positions.
- `config.G_VRK` is a new arm. The old arms stay bit-identical because they never touch the new branch; rerun the bitcmp harness to confirm.

## Bytes/token
- The VR table measures 68.7 B/token; keys add 16, for **~85 B/token**. B's KV is 2,048 B/token (24× more).
- At 32K tokens per sequence this adds 0.5 MiB, for 6.0 MiB against B's 66.8 MiB.
- Ablation: an 8-dim key (8 B/token). A sign-binary 64-bit key is the 8 B fallback.
- Decode cost is 34 dot products of length 16 per step, which is negligible next to the 0.35 ms Python index.

## Causality
- Every candidate satisfies i_k ≤ t (asserted), so the key index i_k−1 ≤ t−1.
- k_i is a function of h_i only, and h_i depends on x[≤i]. It is written once and never updated.
- Source positions are within the same document, because p and q are same-segment.
- **Tests:**
  - Extend the perturbation test: perturb x[200:] and require bit-identical log-probs at positions ≤ 199 for G_VRK, with an assertion that τ·q·k ≠ 0.
  - Cached decode must equal the full forward. This needs the same int8 rounding in both paths: fake-quant at train time equals real quant.
  - Incremental `ICEIndex` `ice_v_src` must equal the batch features.
  - keep = 0 must give bitwise log_softmax.

## Cheap pre-checks (before any training run)
**P0 (CPU, $0, ~1 h, coverage_gate.py style): the token-only ceiling.**
- On the 2,070 CodeRecall items and on val_code, measure two relational rules: "longest exact match wins" (known: 0.97–0.98 on conflicts) and "most recent source among anchor-derived candidates" (for override_diff).
- Then fit a listwise logistic scorer on train-split code over token-only features: slot type, length, distance, length − max length, recency rank, vote count.
- This is the 0-byte competitor. If it reaches slot rank-1 ≥ 0.8 on all four weak categories, keys are unnecessary. In that case adopt it instead and say so.

**P1 (H100, ~$1, ~20 min, diagnose_pointer.py style): the key probe on frozen G_VR.**
- Extract the final normed h on 300 *train-split* code windows and on the CodeRecall items.
- Fit only W_q, W_k and τ (int8 fake-quant) with the listwise slot CE, on top of the frozen trained pointer scores used as an offset. Train on real code only; no CodeRecall template is seen.
- Report correct-slot rank-1 per category (now 0.00 / 0.00 / 0.00 / 0.22 for override_diff / conflict_g2 / conflict_g5 / copy_diff) and the hit-vs-miss slot AUROC on val_code beyond ℓ.
- **Go** if rank-1 ≥ 0.5 on ≥ 2 of the 4 categories and the probe beats the P0 token-only scorer by ≥ 0.15 on at least one.
- **Kill** if rank-1 < 0.25 on all four: frozen h does not carry the source context, so joint training is a long shot. Fall back to the P0 scorer.

## Runs and budget (≈ $7)
1. P0 and P1 ($1).
2. G_VRK, 200M tokens ($2.8).
3. If P0 is within 0.15 of P1: G_VR+rel, the token-only scorer, as the matched control ($2.8).
4. Evals ($0.4).

## Predictions (G_VRK vs G_VR, paired bootstrap, deciding digit)
- conflict_g2 / g5: **≥ +0.30**. The answer is covered 94% / 87% of the time, and the key at the port position can encode the server name from the same line.
- copy_diff: ≥ +0.15. The key at the value position of `"db_port": 1234` should match a query asking for db_port's value. The sentinel should drop from 0.93.
- override_diff: ≥ +0.10, *low confidence*. The stale `config["timeout"] =` is the *more* similar context. Success needs the key to mark "write" and D/recency to break the tie.
- code loss −0.01 to −0.025, and lookup-wrong NLL ≤ −0.03.
- Top-1 = wrong s* falls from 0.34 to ≤ 0.29. This is weakness 1, which is fixed if the model can tell candidates apart rather than hedge.
- First structured-conflict win over B: conflict_g2/g5 above B with the CI excluding 0.

## The 7:1 question, reframed
ICE-K is not meant to be added to 7:1, where attention already copies. The claim becomes: "0 attention layers + 85 B/token beat 2,048 B/token, including on the conflicts". The fair test is the same-memory baseline, G plus one sliding-window attention layer. W ≈ 1,024 gives about the same 2.7 MB at 32K. Prediction: G_VRK beats it by ≥ 0.2 on CodeRecall at gaps ≥ 1K, and on code loss.

## Kill criteria
- G_VRK − G_VR < +0.10 on all four weak categories → "write-time keys" is a negative result.
- The gains need d_k ≥ 64 → the memory story degrades. Report the bytes-vs-accuracy curve at 8/16/64.
- Code loss is worse than G_VR by > 0.01, or τ·|q·k| ends near 0 → the keys are unused.
- The token-only control matches G_VRK within the CI → ship the 0-byte version and drop keys.

## Prior work and novelty
- kNN-LM / neural cache store d-float keys at ~2 KB/token. Memorizing Transformers, NSA/Quest and LoLA keep full KV and select by q·k.
- lit_hybrid_recall's "ICE-Attn" stores whole hidden states. "The Token Before the Value Is the Key" (2609.15545) motivates the choice of k_{i−1}.
- **New here:** the exact token-id index chooses *where* to look, a 16-byte write-time key decides *which*, and the value is the token id, so it costs nothing to store.
