# Critique, reviewer 3 (GStar faculty chair, 27 Oct 2026)

Verdict: the headline is real (two seeds, fair arms), but the four questions get one line each and two cost numbers don't trace to a G_VR measurement. Today: 6. With the fixes below: 7.

## 1. Do the slides answer the four points?

| Feedback point | Slide answer | Reviewer's read |
|---|---|---|
| Misleading lookups | "0.19 → 0 harm" | Half an answer: 0 is vs *no lookup*. Vs the hybrid it is still ~0.04 worse there, and copies a wrong lookup 34% vs 24%. |
| Over-reliance | "ICE off: 0.14 worse" | Honest but undersold: with ICE off, the ICE-trained backbone *beats* G and B on no-match tokens (code 2.511 vs 2.560 / 2.535; directions §0). Division of labour, not laziness (G: one seed). |
| Real cost | "12× less memory; 9% slower; on H100" | **Weakest.** (a) 12× is at 32K; quality is measured at 4K, where it is 3.7 vs 10.8 MiB = **2.9×**. (b) 65 B/token is *modelled* (`results_t200/README.md`: "estimates"). (c) The only decode benchmark, `results_3090/results/benchmark_140m.json`, is **RTX 3090, old arms A/B/C/F, not G_VR** (C 6–7% slower than B). Neither "9%" nor "H100" traces to a measurement. |
| Code use case | "CodeRecall + held-out code" | Asked for conflicts/corrections on a *realistic workload*. "No better than the hybrid" is soft: on override_diff and conflict_g2/g5, B is at chance and ICE *below* it (0.000–0.007), copying the stale value. |

**Training curve.** From `runs/*/log.jsonl`, G_VR − B on code is −0.117 at 131M tokens, −0.058 at 164M and −0.039 at 197M. Prose shrinks less (−0.096 → −0.065).
- G_VR at 164M tokens equals B at 197M, so ICE saves about 20% of the tokens.
- At 1.4 tokens/param, B repeats real text verbatim only ~60% (a trained hybrid: ~100%). My first question: "is your hybrid still learning to copy?" The deck answers it nowhere. Disclosure is free; hiding it costs credibility.
- Both curves include cosine annealing, so don't extrapolate. Say: "it wins at equal tokens, the lead is shrinking, and the 3× run settles it."

**Top-1 +0.4 pt.** Shown in the table, honestly. The script's "81% vs 76%" needs: "overall top-1 is a wash; the gain is on long copies." Without line-completion exact match, "loss ≠ behaviour" stands.

**Synthetic recall.** Keep it off. If asked: "the 0.00 was a rare-token artefact; on real text it's 60% vs 97% verbatim repeat."

### My 2-minute Q&A, and answer strength
1. "Is the hybrid undertrained, and does the gap close?" **Weak**: no data past 197M tokens.
2. "What's the memory saving at the length you measured quality?" **Weak** until the slide says 2.9×.
3. "Is 65 B/token and 9% measured on this model?" **Weak**: no.
4. "Does a user see it, given +0.4 pt top-1?" **Weak**: line-completion exact match is pending.
5. "On my stale-config case, what happens?" **Medium**: mechanism understood (s\* is longest/latest by construction; backbone alone picks right 65% on S), behaviour still worse.
6. "Is this just cheaper attention?" **Medium**: G_MQA pending; say so.

## 2. Next steps: what moves a GStar score

**Top 3**
1. **Measured memory and speed for G_VR** (evidence E2 + `bench` chain).
   - Mac: tracemalloc/RSS of the Python and a numpy compact index at 4K/16K/32K; Kaggle: `benchmark.py --runs B G G_VR`.
   - Literally point 3; ~2 h; removes two untraceable numbers.
2. **A real-code use case with a behaviour metric.**
   - Line-completion exact match, B/G/G_VR, 400 windows, repeat/reuse/novel with CI (Kaggle, built), plus the stale-version edit test (E4, Mac).
   - Answers point 4 and the +0.4 pt worry, even if negative.
3. **Modal after 1 Nov: B and G_VR at 600M tokens** (about $15), then **G_MQA at 200M** if money is left (about $3).
   - Only answer to Q1. Pre-register "gap ∈ [−0.03, −0.01]".
   - Free rider: move frozen-G + fitted scorer (run 3) up the Kaggle order ("ICE off = G exactly").

**Wasted effort for this audience**:
ICE-K/M2 refits and the V2 ladder; 8–32K eval (RoPE-confounded); speculative drafts; truncated state; Mamba2/Pythia bolt-on (a day of code); seed 3, MQAR, G_SWA; rarity-filtered decomposition.

ICE-K gains −0.006 to −0.011 nats and pre-registers no override_diff gain. Use its computed *insight*; spend no GPU on it.

## 3. Slide edits (≤ 3 per slide)

**Slide 1**
1. "A wrong candidate can be ignored." → "A wrong candidate can be **down-weighted**." (It hedges; it doesn't refuse.)
2. "Real cost? Memory and decode speed on H100" → "Real cost? **Total memory incl. the lookup table, at 4K and 32K**".
3. Footer: "140M parameters · 200M training tokens" → "140M parameters · 200M training tokens **(short: 1.4 tokens/param)**".

**Slide 2**
1. Title → "**With no attention layer, ICE beats the hybrid on held-out code and text, at ⅓ of its memory**". Memory tile → "**2.9× less memory at 4K** (3.7 vs 10.8 MiB, where quality is measured); 12× at 32K. Cache per token 65 vs 2,048 B (compact layout)." Speed: say "[measured T4] slower", or drop it until E2 runs.
2. Seeds tile → "**2 seeds**: −0.040 / −0.045 on code. **The lead shrinks with training**: −0.12 at 131M tokens → −0.04 at 197M; 600M run next." Drop "noise ≈ 0.004" and the script's "a tenth of the gap" (n = 2 is not a σ).
3. "Still weak" → "Conflicts and corrections: **both models are at chance, and ICE is worse** because it copies the stale value. Wrong lookups: copied 34% vs 24%. ICE off: 0.14 worse overall, **but better than both baselines on tokens with no match**."

Script: shorten "Next…" to "a 3× longer run, equal-memory attention, line completion" to stay at 2 minutes.

## 4. Closing insight

> "In a recurrent model, attention mostly buys exact copying: 88% of its gain is on tokens a 65-byte lookup already gets right. So a lookup can stand in for it. What it can't do is decide when copying is wrong, and that's the problem left."
