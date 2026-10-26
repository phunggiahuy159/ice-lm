# Rebuttal 2: ICE-Route → ICE-Route-lite

## 1. The leak: a real risk
The proposal said "every slot c ≤ t". That is the right condition, but the span construction q+1..q+4 never enforces it. If the anchor sits at t and q = t−2, then c = t+1, which is the label.

It is worse than the critic says. The existing `test_causality_of_features` would **not** catch it: the integer t+1 does not depend on future tokens. The leak only shows up when the model gathers x[c].

**Fix:**
- In the builder: `cand[cand > t] = -1`, plus an assert.
- A new model-level test: perturb x[200:] and assert that the log-probs at positions ≤199 are bit-identical.

## 2. Reduced version (≈1–1.5 days): ICE-Route-lite, no stored vectors
The full version moves to stage 2. Lite keeps only the part that goes beyond exact recency: **where candidates come from**. It proposes ≤24 token IDs, each with c ≤ t.

- **Deep suffix chain (8 slots).** The 8 most recent occurrences of the longest-matching n-gram, found by chaining the existing `prev` arrays. Each one carries its *own* exact backward-match length, capped at 64.
  - This addresses conflict_g5 symbolically: the target server's line matches back through `name="staging"`, so its match is longer than the others'.
  - ICE throws this information away today by keeping only the most recent occurrence.
- **Anchor spans (16 slots).** Take the 2 most recent *alphabetic* tokens in x[t−7..t], decided by a boolean mask over the vocabulary rather than by rarity, so subwords like `time`/`out`/`port` count. For each, take the latest earlier occurrence, keyed at order 1 and order 2, and the 4 tokens after it.
- **Scoring and output (taken from P1).**
  - Pointer-sentinel score: s_k = q(h_last)·W_e E[c_k] + S_k + L[len_k] + D[dist_k] + γ·ℓ_det[c_k].
  - The mixture is computed in log space. The backbone is ICE-free, with an auxiliary CE (α = 0.5).
  - One function serves both `_logits` and `_loss_sum`, so `eval_loss_splits` and `score_answers` see the mixture.
  - `keep = 0` masks every slot.
  - The alphabetic mask is passed through a setter that raises if it is unset, which covers evals.py, inference.py and data.py.
- **Cost.** About 41 + 24 ≈ **65 B/token**, integers only.
- **Gate.** First measure CPU coverage of the correct value per CodeRecall category, for P1's 12 candidates versus lite's 24. If lite's margin on override_diff, copy_diff and conflict_g5 is below 30 points, I withdraw lite.
- **Test.** Lite equals ICE-V with a different candidate set, so **V vs Route-lite** isolates "beyond exact" as a single variable.
  - **Success:** point estimate ≥ +0.10 with the 95% CI excluding 0 on at least one target category, and code loss within +0.02. This replaces the earlier self-contradicting kill rule.
  - **Baseline timing:** compare at a matched step (≈105M tokens) or re-evaluate C at 400M.

## 3. The identity objection
For the full version I concede: a 64-d per-token store read by a softmax is compressed sparse attention. It would have to beat B plus a top-32 sparse-attention layer at the same bytes per token, and cite InfLLM, RetrievalAttention and Memorizing Transformers.

Lite stores no hidden state. Its softmax ranges over ≤24 index-proposed token IDs, scored against the vocabulary embedding. That is an output restriction, not attention. The identity becomes: *exact index, richer keys (occurrence depth, per-occurrence match length, skip-gram anchors), learned choice.* Stage 2 is justified only if coverage is high and lite still fails on conflict_g5.

## 4. What I absorb from the rivals
- **From P1:** the verifier, the ICE-free backbone with auxiliary CE, the log-space mixture, and the step-0 AUROC gate.
- **From P3 and the faculty critique:** the A (and F) arms, plus measured bytes per token and tokens/s.

## 5. The rivals' weakest points
- **P1:** its candidates are still successors of *the same suffix*.
  - For override_diff and copy_diff, the correct value cannot be among them by construction.
  - For conflict_g5, it is among them only when the target is one of the last two servers.
  - So P1 can *abstain* better on exactly the failures the faculty named, but cannot fix them. The coverage gate will measure this rather than leave it as my assertion.
  - Its verifier is least informed exactly where ICE matters.
- **P3:** lossless speed is orthogonal to the review and *rewards* over-reliance. It needs per-row `pos` and GDN rollback, and ReplaySSM and RAD already cover speculative decoding on SSMs.
