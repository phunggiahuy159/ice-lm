# ICE: 2-minute talk

Slides: `slides/ICE_GStar.pptx` (3 slides; the same script is in each slide's speaker notes).
`slides/ice_gstar.html` is an older version with older numbers; do not present it.

## Script

**Slide 1: title (about 5 s)**

> ICE, In-Context Exact Memory: exact memory for linear attention language models, without full attention.
> I'm Phung Gia Huy.

**Slide 2: problem and method (about 50 s)**

> Linear attention language models like Gated DeltaNet are cheap, because their memory has a fixed size. But they
> forget exact details, like a config value from 2,000 tokens ago. Hybrids fix this by adding full attention layers
> back, and full attention's cache grows with every token.
>
> ICE takes a different route. An exact lookup on the CPU asks: where did this phrase appear before, and what came
> next? It proposes up to 34 candidates. A small learned verifier, with 2,900 parameters, weighs each candidate
> against the model's own prediction, so an implausible candidate gets little weight. The model has no full
> attention layer at all.
>
> Why should this work? Prior work found that most of the quality gap to full attention is recall, and that
> fixed-state models cannot copy. So my expectation: hybrids today use one full attention layer per 3 or 7 linear
> attention layers; if the lookup does the recall, we can use fewer full attention layers and keep the same quality.

**Slide 3: results (about 65 s)**

> The result goes beyond the expectation: no full attention layers, yet better than the 7:1 hybrid.
>
> On held-out Python code and FineWeb-Edu text, linear attention plus ICE has the lowest perplexity: 5% lower than
> the hybrid on code and 6% on text. On the needle-in-a-haystack test, a random 4-digit code hidden in 4,000 tokens
> of text, it gets the whole code right 98% of the time; the hybrid gets 2%. And in our conflict test, where several
> servers share the same key, port, it picks the right server's value 82% of the time; both baselines are at
> chance, 9%.
>
> Three more results. First, ICE completes 64% of repeated code lines exactly, against 41% for the hybrid. Second,
> ICE also helps the hybrid itself: its code perplexity drops by 6%. Third, it is safe when the lookup is wrong. On
> 35% of code tokens the lookup suggests the wrong token, and the verifier learns to ignore it: perplexity on those
> tokens is 24.7, against 24.9 for the same model without ICE. When there is nothing to copy, the model simply runs
> as normal.
>
> Limitations: an exact lookup cannot help when a question is reworded, or when a value is later changed in a
> different form; then it copies the old value. Next: 3 times longer training, and pretrained models.

## Numbers behind each claim

Models: linear attention only = G (`t200_G`), hybrid 7:1 = B (`t200_B`), ours = G_VR backbone + V2 verifier
(`fit_GVRbb_v2`). All 140M parameters, 200M training tokens. Evals on held-out data at 4K context, the same items
for every model. Full analysis: `ice_results/README.md`; raw tables: `ice_results/all_eval_tables.txt`.

| Claim on the slide | Source |
|---|---|
| Perplexity 6.68 / 5.64 / 5.37 (code), 31.3 / 29.5 / 27.8 (text) | exp of held-out loss 1.899 / 1.730 / 1.680 and 3.444 / 3.384 / 3.324 nats, 500 × 4K windows per split |
| −5% code, −6% text vs the hybrid | 5.37 / 5.64 = 0.952; 27.8 / 29.5 = 0.942 |
| Cache 0 / 2,048 / 65–90 B per token | KV cache of the 2 full attention layers; ICE index measured as compact arrays, 82–90 B/token (`research/round3/results_cpu3.md` §4.3) |
| Needle 0% / 2% / 98% | `needle_templated@4096/em`, n = 50 |
| Conflicts 9% / 9% / 82% | CodeRecall `conflict_g2` and `conflict_g5`, deciding-digit accuracy, mean of the two; 100 items × 3 distances each |
| 64% of repeated code lines (hybrid 41%); all lines 15.7% vs 11.8% | Line completion, 1,200 lines from 400 held-out windows; ICE fitted on the frozen G backbone (`fit_Gbb_v2`). Difference +22.8 pts [+16.3, +29.7] on repeated lines, +3.8 [+2.2, +5.6] on all lines |
| −6% for the hybrid with ICE, 5.64 → 5.29 | B + V2 verifier (`fit_Bbb_v2`), code loss 1.730 → 1.665 |
| No harm: 35% of code tokens, 24.7 vs 24.9 | Tokens where the lookup's top suggestion is wrong (`code/lk_wrong`, 35.5% of tokens): loss 3.208 (ours) vs 3.215 (G) |

## Likely questions

- **"Are these standard benchmarks?"** "Perplexity is on held-out splits of two public datasets, FineWeb-Edu and
  CodeParrot. The needle test follows the needle-in-a-haystack setup (Kamradt, 2023), with FineWeb-Edu text and a
  random 4-digit code. The conflict test is my own, in the style of MQAR from the Zoology paper, but with keys that
  repeat on purpose. All use the same items for every model."
- **"What is the conflict test?"** "A list of servers that all have a `port=` field, then a question for one
  server's port. The most recent `port=` belongs to another server, so a plain lookup copies the wrong value. Each
  port starts with a different digit, so the first digit shows which server the model picked. Guessing gives about
  11%. The verifier was never trained on this test; my earlier verifier scored 0%."
- **"Why does the hybrid get only 2% on the needle?"** "It is small (140M) and briefly trained, with only 2 full
  attention layers. Per digit it gets 21% right, against 1.5% for linear attention only, so attention helps a
  little, but it rarely gets the whole code. At 1K tokens it gets 16%."
- **"What if the question is reworded?"** "An exact lookup can't help. With a paraphrased needle question every model
  is at 0% exact match, ICE included."
- **"What if the lookup is wrong?"** "No match: the model runs unchanged. Wrong suggestion, 35% of code tokens: the
  verifier learns to ignore it, 24.7 vs 24.9 perplexity on those tokens. On the conflict test, where the top
  suggestion is wrong by design, it scores 82%."
- **"How much memory does it save?"** "The per-token cache is about 25× smaller: 82 bytes against 2,048. Per
  sequence at 4K, not counting weights, that is 3.8 vs 10.8 MiB, and 11× at 32K. The index lives in CPU memory, not
  on the GPU."
- **"Is the 65–90 bytes real?"** "Yes: I built the compact index as arrays and checked every lookup against the
  working index, with 0 mismatches. My Python prototype is larger, about 520–640 bytes per token; that is an
  implementation detail."
- **"Is decoding slower?"** "About 9% at 32K on an H100 (78.4 vs 86.4 tokens/s), because the index is Python on the
  CPU, about 0.35 ms per step. A C++ or GPU index would remove most of that."
- **"How many parameters does ICE add?"** "2,873, about 0.002% of the model."
- **"Which model gives the 64%?"** "ICE added to the frozen linear attention model, training only the verifier. It
  still beats the hybrid on perplexity too: 5.53 vs 5.64 on code."
- **"Does the model depend on ICE?"** "The jointly trained model does: with ICE switched off it is worse than linear
  attention only. The frozen version doesn't: with ICE off it is exactly the original model."
- **"Any trade-offs?"** "When a value is reassigned with the same syntax, the newest value is right, and ICE picks it
  15% of the time against 33% for the hybrid, because the verifier prefers the longest match. Lines that only reuse
  earlier tokens lose about 1 point of exact match."
- **"Is it robust?"** "Refitting the verifier with another seed changes code loss by 0.0002. The lead over the hybrid
  narrows with training (−0.117 → −0.058 → −0.039 nats at 131M / 164M / 197M tokens), which is why 3× longer
  training is next."
- **"Is 140M too small?"** "Yes, it's a proof of concept. Next is adding ICE to real pretrained models."
