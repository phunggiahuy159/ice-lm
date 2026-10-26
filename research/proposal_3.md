# Proposal 3 (practical payoff / paper story)

## Name
**ICE-Draft: the lookup that restores recall also drafts the next span, with a learned "how far will the past repeat" head.**

## One-paragraph idea
ICE already knows *where* the past repeats (the matched position p*), so it can propose the whole span x[p*+1 .. p*+k], not just one successor. That span costs nothing to read: ICE already stores the token array. The hybrid then checks the span in one forward pass (speculative decoding with greedy verification, so the output is identical to the model's own decode). We add one trained piece, a **run-length head**. It predicts r = how many of the next k span tokens will be correct. The label is free (common prefix of x[t+1:] and x[p*+1:]). The head does two jobs. (a) Its prediction sets the draft length at inference, like SuffixDecoding's adaptive drafts but learned from the hidden state. (b) Its P(r≥1) feeds the ICE output-bonus gate, which gives the content-aware "is the lookup right" signal that the agree/rel buckets failed to provide. The story: one 41 B/token table buys recall, lossless speed and a trust signal, all in one mechanism.

## Exact mechanism (files)
- `ice_lm/ice_features.py`: add `ice_run[t] = min(K, LCP(x[t+1:], x[p*+1:]))`, K=8, kept inside the document. It is vectorised like the backwards match extension (≈30 lines).
- `ice_lm/data.py` / `config.py`: pass `ice_run`. New arms `C_run` and `F_run` with flag `ice_run_head=True`.
- `ice_lm/model.py` `ICEOutput`: run head `Linear(d, K+1)` on `h_last` plus a len* embedding. Add an auxiliary CE loss (weight 0.1, only at positions where s* exists). Add `w_r * logit P(r≥1)` to the bonus pre-activation. `forward` returns LM loss + aux loss. `InferenceCache.snapshot()/restore()` copies the GDN `S` and conv state (fixed size, a few MB) and resets `pos` for KV.
- `ice_lm/inference.py` `ICEIndex`: `draft(k)` returns `tokens[p*+1 : p*+1+k]`. Add an undo log in `step()` (table writes, hist appends) plus `rollback(n)`.
- New `ice_lm/specdec.py` (≈120 lines), one loop: draft k̂ = argmax of the run head (0 means plain decode) → snapshot → `forward_cached(T=k̂+1)` (the chunked fla path already takes `initial_state`) → accept the longest prefix that agrees with argmax → on partial accept, restore and re-run the accepted prefix → ICE rollback. Baseline mode: fixed-k, untrained drafter = Prompt Lookup Decoding (PLD).
- `scripts/benchmark.py`: `--spec {off,pld,ice}` with **real generation**, not teacher forcing. Report tokens/s, accepted tokens per forward, forwards per token, and peak memory at 4K/32K, batch 1 and 16.

## Headline figure and one-sentence claim
**Figure:** x = decode tokens/s on held-out code at 32K context (batch 1, real generation). y = CodeRecall accuracy (second panel: code loss). Each arm is a point. ICE arms have an arrow from plain decoding to ICE-Draft, and B has an arrow to B+PLD. Marker size = bytes/token. The target is the top-right corner at small marker size.
**Claim (to test, not a result):** "A 41 B/token exact in-context lookup, trained into a 7:1 hybrid, gives A-like code recall at about 1/4 of the KV memory, and its drafts give a lossless ≥1.5× code decode speedup that an untrained lookup drafter (PLD) on the same backbone does not match."

## Memory/speed cost
- Memory: the drafter adds 0 B/token, because the span comes from ICE's existing token array. The fixed per-sequence cost is one GDN+conv snapshot. The run head adds about 9·d parameters.
- Training: the aux head is negligible, and the label is computed on the CPU in the loader.
- Inference: a verify pass costs about the same as one decode step at batch 1, because the 140M model is launch-bound. Rejected tokens waste compute. A partial accept costs one extra short forward.
- Honest caveats: at large batch the model is compute-bound and the speedup shrinks, so report batch 16 too. The ICE index is Python (about 100 µs per step), so report model-only and end-to-end numbers.

## Experiment
Arms (3 × 1.3 H100-h, full 400M recipe, same seed): **A** (3:1, needed as the quality/memory reference if not already run at full recipe), **C_run**, **F_run** (15:1 + ICE + run head). B, C and C_notrust are reused. If A already exists, spend the third run on a second seed of C_run.
Evaluations (inference only, cheap): {A, B, B+PLD, C, C+ICE-Draft (fixed k), C_run+ICE-Draft (adaptive), F_run+ICE-Draft} on 200 held-out code prompts and 200 prose prompts, generating 256 tokens each. Also the existing loss splits (all, copyable, lookup-wrong), follow-wrong rate, CodeRecall incl. override_diff and conflict_g5, and C_off.
**Pre-registered predictions:**
1. On code, C_run+ICE-Draft gets ≥1.8 accepted tokens per forward and ≥1.5× tokens/s over plain decoding of the same model at batch 1, at 32K.
2. Training matters: C_run's accepted length is ≥1.15× that of B+PLD.
3. Prose slowdown is ≤5%, because the adaptive k̂ is mostly 0.
4. C_run's follow-wrong rate and lookup-wrong loss are lower than C's (0.356 / 4.014), and its code loss is within 0.02 of C's.
5. F_run's CodeRecall is within 2 points of A's at less than half of A's bytes/token.

**Kill criterion:** stop the "trained drafter" claim if any of these holds. (i) C_run's accepted length is ≤ B+PLD's: then training buys no speed, and the result is just PLD on a hybrid. (ii) The end-to-end speedup is <1.2× at batch 1. (iii) C_run's follow-wrong rate is ≥ C's. In case (iii), any acceptance gain is the model rubber-stamping the draft, and we must say so.

## Novelty vs closest papers
- **PLD / REST (2311.08252) / SuffixDecoding (2411.04975) / SAM-Decoding (2411.10666):** training-free lookup drafters for transformers. New here: the model is *trained* on the same lookup it drafts from, a learned run-length head sets the draft length, and the verifier is a GDN hybrid that needs recurrent-state rollback.
- **Medusa/EAGLE/MTP-style drafters:** these draft from the model's own state with extra parameters. ICE-Draft drafts from exact context at 0 B/token. (Not re-verified in this round of the literature review.)
- **NEST (2405.19325):** extends spans on retrieval confidence, using a dense kNN datastore. Ours is in-context and uses token IDs only.
- **Self-RAG / CRAG verifier ideas (lit_trust):** the run head is a free-label verifier. Its novelty is that a single label serves both trust and draft length.
- **Engram (2601.07372), Akyurek n-gram heads (2401.12973), HOLA (2607.02303):** recall-only. None of them reports a decode speedup.
- Speculative decoding for SSM/hybrid models exists in some form. I did not verify specific papers this round, so the student must search for it before claiming "first".

## Biggest risk
The acceptance gain could come from **over-reliance**: greedy verification accepts whatever the model's argmax is, and C already follows the lookup about 2× as often as B. If so, the speedup is real but comes from a less faithful model. The paper then becomes "fast because it copies blindly." The kill criterion (iii) and the follow-wrong / override_diff columns exist to catch this. A secondary risk is engineering: GDN rollback, the ICE undo log and Python index overhead could eat the gain at 140M. Build specdec on B+PLD first (no training, about half a day) to measure the speedup ceiling before spending any H100 time.
