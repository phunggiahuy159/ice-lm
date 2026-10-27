# ICE: 2-minute talk

Slides: `slides/ice_gstar.html`. Open it in a browser and use ← → to switch slides. To export for the shared deck, print to PDF: each slide prints as one 1920×1080 page.

## Script (about 260 words, about 2 minutes)

**Slide 1 (about 55 s)**

> Recurrent language models like Gated DeltaNet are cheap, because their memory has a fixed size. But they forget
> exact details, like a config value you saw 2,000 tokens ago. Hybrids fix this by adding attention back, and
> attention's cache grows with every token.
>
> ICE takes a different route. An exact lookup on the CPU asks: "Where did this phrase appear before, and what came
> next?" It proposes up to 34 candidates. The key change since your review is that the lookup only *proposes*. A small
> verifier, with 2,700 parameters, lets the model check each candidate against its own guess, so a wrong candidate can
> be ignored. The model has no attention layer at all.
>
> You asked four things: misleading lookups, over-reliance, real cost, and a code use case. I measured all four.

**Slide 2 (about 65 s)**

> Here is the result at 140 million parameters. On code loss, the recurrent-only model scores 1.90, the attention
> hybrid 1.73, and recurrent plus ICE 1.69: **better than the hybrid, with no attention at all.** That's 4% lower
> perplexity on code and 6% on text. Where the answer is a copy from far back, it predicts the exact next token 81%
> of the time, against 76% for the hybrid.
>
> It uses 65 bytes per token instead of 2 kilobytes, which is 12 times less memory per sequence at 32K tokens. Decoding
> is 9% slower. I trained a second seed and got the same win. Seed noise is a tenth of the gap.
>
> On misleading lookups, the old version lost 0.19 nats where the lookup was wrong. The verifier brings that to zero.
>
> Two weaknesses remain. When the context has conflicting values, like a variable that was updated, it still picks
> the wrong one. And it still leans on the lookup. Next, I'll compare against an attention baseline with the same
> memory, build a verifier that compares candidates, and test on a real pretrained model.

## How to read the results

1. **The headline result is solid.**
   * Recurrent + ICE (G_VR) beats the 7:1 hybrid by 0.040 / 0.045 on code over two seeds, with a seed spread of about 0.004.
   * It also beats the hybrid on prose (−0.060).
   * The comparison is fair: same data, same token budget, almost the same parameter count (148M vs 145M).
2. **Where the gain comes from: copying from far back.**
   * On tokens that repeat something more than 512 tokens back, G_VR scores 0.933 vs the hybrid's 1.132 and
     recurrent-only's 1.680.
   * On tokens with no match, all three are about the same (2.51–2.56).
   * So ICE does what it was designed to do. It doesn't improve general language modelling.
3. **The verifier fixed robustness, the main criticism.**
   * Old ICE was 0.19 *worse* than no ICE where the lookup was wrong; the verifier brings that to 0.
   * Caveat: it still copies a wrong lookup 34% of the time (hybrid: 24%). It hedges with probability rather than
     refusing outright, which is why the loss doesn't get worse.
4. **Over-reliance is reduced but still present.**
   * With ICE switched off, the model is 0.14 worse than a recurrent model trained without ICE (old ICE: 0.43).
   * One fair reading: this is specialisation. The hybrid also breaks if you remove its attention.
5. **Conflicts are the real limitation.**
   * Examples: an updated value, or "which of two objects".
   * The right answer is often among the candidates: 33% of the time where the lookup's first guess is wrong on real
     code, and almost always on CodeRecall.
   * But the verifier scores each candidate on its own and can't compare them. Fixed rules don't help either:
     "longest match wins" is worse than the default on real code.
   * Next step: ICE-V2, which scores candidates relative to each other.
6. **The missing control.**
   * One attention layer with the same memory budget (256 B/token, 4× ICE's) hasn't been trained yet. Modal's
     monthly credit ran out.
   * If a reviewer asks "is this just cheaper attention?", the honest answer is "that run is next".

## Likely questions

* **"Is 140M too small?"** Yes, it's a proof of concept. Next is a bolt-on test with real pretrained models
  (Mamba2-370M vs Pythia-410M).
* **"Why is decoding slower?"** The index is in Python on the CPU, about 0.35 ms per step. A C++ or GPU index would
  remove most of that.
* **"Does it help a model that already has attention?"** At 7:1, the hybrid learns to copy by itself, and ICE's gain
  shrinks to about 0.01 with training. ICE matters most when there's no attention.
* **"What's CodeRecall?"** A synthetic code benchmark: copy a config value, pick the right object, use the latest
  value. ICE gains +0.15 to +0.23 where copying is enough, and is at or below the hybrid on conflicts.
