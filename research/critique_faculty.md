# Faculty critique (the reviewer who gave 5/10)

What I asked for: (1) robustness to misleading lookups and conflicting continuations; (2) proof that the model does not just over-rely on copying; (3) **measured** memory and speed, including the table; (4) one realistic code use case with a clear saving and a small, understood tradeoff. The brief shows ICE helps where the lookup is right, but the model over-relies on it (follow-wrong 0.36 vs 0.20; C_off is worse than B). The gap to B also shrinks with training (0.65 → 0.27). The project's identity is "an attention-free exact copy path lets hybrids drop attention layers." **None of the three proposals makes that claim testable unless A (and ideally F) is run at the full recipe.** Only P3 includes A.

## P1: ICE-V (backbone-verified candidate pointer)
**Strengths.** It goes straight at concerns 1 and 2, the core of my feedback. The one-line story works in 2 minutes: "the judge never sees the evidence." C_off = backbone by construction, so the laziness result is fixed by design rather than hoped for. Putting the second occurrence in as its own candidate is a real answer to "same phrase, different continuation." The step-0 gate costs nothing (logistic regression with B's log p(s*)). The pre-registered effects are large: follow-wrong drops from 0.36 to ≤0.27. A single seed can show a change that size. About 150 lines, so it fits the time left. It sharpens the identity (still exact, still 41 B/token).
**Weaknesses.** It does not cover concern 3 beyond restating the table size. Override_diff and conflict_g5 are probably still lost, because the right value does not follow the same n-gram. The verifier is circular: the backbone is least informative exactly where ICE is needed (far-back identifiers). If C's gain lives on the input side, V loses copyable quality, and V+in brings the over-reliance back.
**Hardest Q&A:** "If B could tell whether a 2,000-token-back identifier is correct, it wouldn't need ICE. Show me verification accuracy on the *copyable >512* split, not overall."
**Predicted score:** works → 7 (7.5 with a memory/speed table); fails → 5.5–6 (an honest negative with a clean gate).

## P2: ICE-Route (index-routed pointer read)
**Strengths.** It is the only proposal that attacks the "which object's value" and stale-value failures (override_diff, conflict_g5), which my feedback named explicitly. The day-0 coverage gate is cheap and informative even if the method is never trained. Done well, it would be the most novel result.
**Weaknesses.** It dilutes the identity. A learned 64-d vector per token, read by a softmax over 32 gathered positions, is a small sparse-attention KV cache (130–190 B/token, 3–5× today). I would immediately ask why this isn't top-k sparse attention or landmark attention with extra steps. It has the most moving parts: rank thresholds, anchors, chains, two read sites, int8 storage. The selection rules it needs are rare in 400M tokens, as the proposal itself admits. Thresholds of +0.10–0.15 against an SE of about 0.04 on synthetic CodeRecall will likely come out ambiguous at one seed. Speed is promised, not measured. Too ambitious for 1–2 days.
**Hardest Q&A:** "You stored a per-token vector and attend over it. In what sense is this still attention-free, and how does it compare to giving B one sparse top-32 attention layer at the same bytes per token?"
**Predicted score:** works → 7.5–8; fails or half-finished → 4.5–5 (identity blurred, nothing clean to show).

## P3: ICE-Draft (lookup drafts spans + learned run-length head)
**Strengths.** It answers concern 3 and "a clear saving on a realistic workload" more literally than anything else: real generation, tokens/s, peak memory at 32K, batch 1 and 16. Speed numbers are nearly deterministic, so a single seed hurts less. It is the only proposal that runs A and F_run, which is what the headline claim needs. The B+PLD-first step bounds the risk in half a day.
**Weaknesses.** Lossless speculative decoding makes speed *orthogonal* to my main concern. A model that copies blindly drafts *better*, so the speedup can reward over-reliance; the author admits this. The trust fix (run head feeding the bonus gate) is a weaker version of agree/rel and may also give C ≈ C_notrust. At 140M batch 1 the model is launch-bound, so the speedup may not carry over to real scale. The engineering is heavy: GDN state rollback, an ICE undo log, and Python index overhead. The story shifts from "drop attention" to "decode faster", and PLD already exists without training.
**Hardest Q&A:** "PLD gets most of this on any model for free. What exactly did *training* buy you, and did follow-wrong go down or up?"
**Predicted score:** works → 7–7.5 (8 if F_run ≈ A on CodeRecall at less than half the bytes); fails → 5.5 (B+PLD numbers and the A frontier still help).

## Ranking
1. **P1.** It best answers the feedback I actually wrote, is the most falsifiable, and costs the least.
2. **P3.** It earns its place through the A/F arms and real measurement, not through spec-decode novelty.
3. **P2.** It is the most interesting research, but the wrong risk profile for a student deadline, and it erodes the attention-free claim.

## Recommendation: combine
A combination beats any single proposal:
- **Core method:** P1's ICE-V (V and V_nover; V+in only if step 0 says the input side matters).
- **Claim arms (from P3):** A at the full recipe, plus F with the ICE-V output.
- **Measurement (from P3):** the benchmark, without the spec decoder. Report measured bytes/token (including the table) and tokens/s at 4K/32K for A, B, C, V and F.
- **Diagnostics only (from P2):** the free CPU coverage analysis, to say honestly what fraction of override and conflict cases *no* exact method can reach. That fraction becomes the "where copying is the wrong behavior" slide.
- **Optional:** P3's run-length label as one more input to P1's sentinel (free label, a few lines).
- **Drop** ICE-Route training and full spec-decoding unless time is left over. The B+PLD half-day could be one slide of "future work, preliminary."

That is about 5–6 arm-runs, within budget. The story becomes: *exact lookup + independent verifier; same quality as A at X% of the memory; follows wrong lookups no more than B; here is where it still fails.* That is a 7–8.

## The one thing I want to see first
**The memory–quality frontier at 400M tokens: A vs B vs C (and F) on code loss, copyable loss and CodeRecall, against measured bytes/token and tokens/s.** If C/F does not close most of the B→A gap once B has finished learning to copy, no trust fix matters and the project's identity is in question. If it does, P1's free step-0 gate decides the next run the same day.
