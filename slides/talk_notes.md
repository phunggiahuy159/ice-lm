# ICE: 2-minute talk

Slides: `slides/ice_gstar.html`. Open it in a browser and use ← → to switch slides. To export for the shared deck,
print to PDF; each slide prints as one 1920×1080 page. Wording follows `research/round3/decision3.md` §2.

## Script (about 2 minutes)

**Slide 1 (about 55 s)**

> Recurrent language models like Gated DeltaNet are cheap, because their memory has a fixed size. But they forget
> exact details, like a config value you saw 2,000 tokens ago. Hybrids fix this by adding attention back, and
> attention's cache grows with every token.
>
> ICE takes a different route. An exact lookup on the CPU asks: "Where did this phrase appear before, and what came
> next?" It proposes up to 34 candidates. The key change since your review is that the lookup only *proposes*. A small
> verifier, with 2,700 parameters, lets the model check each candidate against its own guess, so a wrong candidate
> gets down-weighted. The model has no attention layer at all.
>
> You asked four things: misleading lookups, over-reliance, real cost, and a code use case. I measured all four.

**Slide 2 (about 65 s)**

> Here is the result at 140 million parameters, with equal training. On code loss, the recurrent-only model scores
> 1.90, the attention hybrid 1.73, and recurrent plus ICE 1.69: **better than the hybrid, with no attention at all.**
> That's 4% lower perplexity on code and 6% on text. Where the answer is a copy from far back, it predicts the exact
> next token 81% of the time, against 76% for the hybrid. In fact, 88% of what attention adds is on tokens this
> lookup already gets right.
>
> It needs a third of the hybrid's memory at the 4K length we tested, and its memory grows 65 bytes per token instead
> of 2 kilobytes. Decoding is 9% slower. A second seed gives the same win. But the lead shrank during training, from
> 0.12 to 0.04, so a 3× longer run is next.
>
> On misleading lookups, the old version lost 0.19 nats where the lookup was wrong. The verifier brings that to zero.
>
> The weakness that remains is conflicts: when a variable was updated, it copies the outdated value. Next, I'll
> train 3× longer, compare against an attention layer with the same memory, and test on a real pretrained model.

## How to read the results

1. **The headline holds at equal training (200M tokens).**
   - G_VR beats the 7:1 hybrid by 0.040 / 0.045 on code in both seeds, and by 0.060 on prose. Data, token budget
     and parameter count are matched (148M vs 145M).
   - **The lead is shrinking.** In the 8-window training evals it went −0.117 → −0.058 → −0.039 over the last third
     of training. The recurrent-only gap to B stayed at +0.17.
     - G_VR at 164M tokens matches B at 197M, so B needs ≥ 17% more tokens to match.
     - The LR anneals (to 0.1×), so the curve cannot be extrapolated. A 3× longer run with a constant-then-decay
       schedule (WSD) settles it (decision3 M2).
2. **What attention adds is mostly exact copying.**
   - Split by what the lookup predicts (the same tokens for every model), 88% of the code-loss gain from G to B (90%
     on prose) falls on tokens the lookup already gets right. Ordinary training puts only 45% there.
   - On tokens that repeat something more than 512 back: G_VR 0.933, B 1.132, G 1.680.
   - Don't say "the whole prose gain is copying": 56% of B → G_VR on prose is elsewhere.
3. **Misleading lookups.**
   - Against no lookup, the harm went from 0.19 (old ICE) to 0. Against the hybrid, ICE is still 0.04 worse on
     those tokens.
   - It follows a wrong lookup 34% of the time vs 24%. It hedges rather than refuses.
4. **Over-reliance.**
   - With ICE off, the model is 0.14 worse than recurrent-only (0.14–0.17 over the two seeds; old ICE: 0.43).
   - On tokens with no match, the ICE-trained backbone with ICE off beats both G and B (code 2.511 vs 2.560 / 2.535).
     The likely cause is loss weighting (the auxiliary CE weights those tokens 1.5×), so it is Q&A only.
5. **Conflicts are the real limitation.**
   - On stale values and "which object", the hybrid guesses at chance. ICE does worse, because it confidently copies
     the wrong occurrence.
   - On real code, no fixed rule beats 0.07 where the lookup's top guess is wrong; the oracle is 0.34.
6. **Memory.**
   - The compact index layout, built as real arrays (`scripts/index_memory.py`, `research/round3/index_memory.json`):
     82 B/token at 4K and 90 at 32K, every lookup checked. The earlier count was 62–69, at an ideal load factor;
     power-of-two hash tables add the rest.
   - Per sequence: 3.8 vs 10.8 MiB at 4K (2.8×), where quality is measured; 6.3 vs 66.8 MiB at 32K (11×).
   - The Python index used here takes 520–640 B/token. That is an implementation artefact, and still 3× below the
     hybrid's KV cache.
   - Decode at 32K: 78.4 vs 86.4 tok/s (H100, `results_t200/benchmark_t200.json`).

## Likely questions
- **"Is the hybrid undertrained?"** "Possibly. It repeats real text verbatim only about 60% of the time. The 3×
  longer run answers that."
- **"Is 65 bytes measured?"** "A compact array index, built and checked, takes 82–90 bytes per token, against
  2,048 for the hybrid's KV cache. My Python prototype takes about 600; a C index would not."
- **"What happens with ICE off?"** "On tokens with no match it beats both baselines, probably because of loss
  weighting. On copyable tokens it is worse: it specialises."
- **"Synthetic recall?"** "With random rare tokens the hybrid scored 0, an artefact. On real text it is 60% vs 97%
  verbatim repeat."
- **"Is 140M too small?"** "Yes, it's a proof of concept. Next is a bolt-on to real pretrained models."
- **"Why is decoding slower?"** "The index is Python on the CPU, about 0.35 ms per step. A C++ or GPU index would
  remove most of that."
- **"Does it help a model that already has attention?"** "At 7:1 the hybrid learns to copy by itself, and ICE's gain
  shrinks to about 0.01. ICE matters most without attention."
