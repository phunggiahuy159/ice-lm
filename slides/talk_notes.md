# ICE: 2-minute talk

Slides: `slides/ice_gstar.html`. Open in a browser, use ← → to move between slides, and print to PDF for the shared
deck. Put your name in place of "[Your name]" on slide 1.

## Script

**Slide 1 (about 50 s)**

> Recurrent language models like Gated DeltaNet are cheap, because their memory has a fixed size. But they forget
> exact details, like a config value from 2,000 tokens ago. Hybrids fix this by adding attention layers back, and
> attention's cache grows with every token.
>
> ICE takes a different route. An exact lookup on the CPU asks: "Where did this phrase appear before, and what came
> next?" It proposes up to 34 candidates. A small learned verifier, with 2,700 parameters, weighs each candidate
> against the model's own prediction, so an implausible candidate gets little weight. The model has no attention
> layer at all.
>
> I compare three models of the same size, trained on the same data: recurrent only, a hybrid with attention, and
> recurrent plus ICE.

**Slide 2 (about 70 s)**

> Here are the results. On held-out Python code and FineWeb-Edu text, recurrent plus ICE has the lowest perplexity:
> 4% lower than the hybrid on code and 6% on text, with no attention at all.
>
> On two standard retrieval tests, a phone-book lookup and a needle in a haystack at 4,000 tokens, ICE gets the whole
> answer right 82% and 64% of the time. Both baselines get 0%.
>
> It uses 2.8 times less memory than the hybrid, measured, and decoding is 9% slower. And 88% of what attention adds
> to a recurrent model turns out to be exact copying, which is exactly what the lookup supplies. The result holds
> across two training seeds.
>
> Two limitations. When a value has changed, for example an old version of a file is in context, ICE copies the old
> name 65% of the time, against 13% for the hybrid. And the models are small and briefly trained; the lead narrows
> as training goes on. Next: 3 times longer training, and pretrained models.

## Numbers behind each claim

| Claim | Source |
|---|---|
| Perplexity 6.68 / 5.64 / 5.42 (code), 31.3 / 29.5 / 27.8 (text) | Held-out 500 × 4K windows; loss 1.899 / 1.730 / 1.690 and 3.444 / 3.384 / 3.324 nats (`results_t200/README.md`) |
| Phone-book 0 / 0 / 82%, needle 0 / 0 / 64% (exact match, 4K, n = 50) | `results_t200/tables.txt` (phonebook@4096/em, needle_templated@4096/em) |
| 2.8× less memory, 9% slower | 3.8 vs 10.8 MiB per sequence at 4K (measured index); 78.4 vs 86.4 tok/s at 32K, H100 (`results_t200/benchmark_t200.json`) |
| 88% | B − G code loss gap (0.169) split by lookup bucket: 0.149 falls on tokens the lookup predicts right (ordinary training: 45%) |
| 2 seeds | Code loss ICE − hybrid −0.040 / −0.045 |
| 65% vs 13% | Stale-file test, 100 held-out files (`research/round3/results_cpu3.md` §4.5) |
| Lead narrows | ICE − hybrid −0.117 → −0.058 → −0.039 at 131M / 164M / 197M tokens |

## Likely questions

- **"Why does the hybrid score 0% on phone-book and needle?"** "The models are small (140M) and briefly trained. The
  hybrid gets single digits right (11% phone-book, 21% needle, token accuracy) but rarely the whole answer. At 1K
  tokens it gets 14% on the needle; ICE gets 92%."
- **"Are these standard benchmarks?"** "Perplexity is on held-out splits of FineWeb-Edu and CodeParrot, two public
  datasets. The phone-book task follows Jelassi et al. 2024 (*Repeat After Me*), and the needle test is the usual
  needle-in-a-haystack setup. Both are my implementations, with the same items for every model."
- **"What if the question is worded differently from the context?"** "Exact lookup can't help there. With a
  paraphrased needle question, all models are near 0% exact match. ICE is for exact copies."
- **"Is the 65–90 bytes real?"** "Yes: I built the compact index as arrays and checked every lookup. My Python
  prototype is larger, about 600 bytes per token; that is an implementation detail."
- **"Why is decoding slower?"** "The index is Python on the CPU, about 0.35 ms per step. A C++ or GPU index would
  remove most of that."
- **"Does the model depend on ICE?"** "Partly. With ICE switched off, it is 0.14 nats worse than the recurrent-only
  model."
- **"Is 140M too small?"** "Yes, it's a proof of concept. Next is adding ICE to real pretrained models."
