# Round 3: directions beyond the current plan (advisor, 27 Oct 2026)

Web checks covered SSM speculative decoding, Zoology's AR-hit analysis and length extrapolation. Negative searches are not proof.

## 0. A free finding already in `results_t200.json`
The `lk_*` buckets come from the exact lookup alone. They are the same tokens for every arm (code: right 47.6%, wrong 35.5%, none 16.9%). Each gap therefore splits exactly as Σ frac × Δloss:

| gap (nats) | total | lookup right | lookup wrong | no lookup |
|---|---|---|---|---|
| code, G→B (what attention buys) | −0.169 | **−0.149 (88%)** | −0.015 | −0.004 |
| prose, G→B | −0.060 | **−0.054 (90%)** | −0.002 | −0.004 |
| code, B→G_VR | −0.040 | −0.050 | **+0.014** | −0.005 |
| prose, B→G_VR | −0.060 | −0.026 | −0.011 | **−0.022** |

- Almost all of attention's value to a GDN model is on tokens a 65 B/token lookup already predicts.
- ICE recovers 133% of that value (−0.199 vs −0.149).
- Attention wins only where a lookup must be *rejected*.
- **The ICE-trained backbone beats G and B on no-lookup tokens even with ICE off:**
  - code: G_VR_off 2.511 vs G 2.560 and B 2.535;
  - prose: 3.192 vs 3.255 and 3.245;
  - seed 2 repeats it against B (prose −0.059, code −0.027).
  - So "laziness" is half of a **division of labour**: the table does verbatim recall, and the backbone's freed capacity goes to parametric prediction. Engram (2601.07372) claims the same for static n-grams.
- Caveat: G has one seed.

## 1. The "copy-predictable gap" as the core insight (idea c), rank 1
**Novelty: moderate.** Zoology (2312.04927) put 82% of the gated-conv→attention gap on "AR hits": in-context bigrams that are rare in training, 6.4% of tokens. New here:
- a delta-rule RNN backbone;
- longest-match buckets, including a *lookup wrong* bucket;
- a constructive closure at 65 B/token;
- the division-of-labour effect.

**Experiment (CPU, local B/G_VR, ~1 h).** Per-token losses on 300 code and 300 prose windows. Sub-split lookup right/wrong by:
- distance (<64, 64–512, >512);
- match length;
- token class;
- a Zoology-style rarity filter (n-gram counts from a train slice, or a val proxy).

The G→B half needs one forward-only Kaggle cell for G (~20 min).

**Pre-registered:**
- ≥ 75% of code G→B sits on lookup-right tokens after the rarity filter;
- ≥ 50% of B→G_VR's lookup-right gain is at dist > 512;
- B's lookup-wrong edge concentrates at match length ≥ 4, where confident wrong lookups must be refused.

**Kill:** the share falls below 50% after filtering. Attention's value is then mostly easy local tokens; reframe.

**Value:** one figure explains why 65 B can stand in for 2 KB ("in a GDN hybrid, attention is mostly an exact-copy device") and turns weaknesses 2–3 into a story.

## 2. Copy-gap audit plus bolt-on on public models (ideas d + c), rank 2
**Novelty: moderate-high.** A frozen LM plus a cache or kNN mixer is old: kNN-LM 1911.00172, neural cache 1612.04426, Bakhtin 1804.07705, infini-gram interpolation 2401.17377. I found no exact copy bolt-on for frozen Mamba, RWKV or DeltaNet. A bucket audit across matched pairs is new as a measurement.

**Experiment.** Mamba-370m-hf (or mamba2-370m) vs Pythia-410m: same Pile data, same NeoX tokenizer, retokenized held-out code and prose.
- (i) Forward-only decomposition as in §0.
- (ii) **ICE-0 (training-free):** a count/recency candidate distribution, mixed with one λ per match-length bucket tuned on dev (Grave- or infini-gram-style), computed from cached top-K log-probs. It is mandatory: reviewers will ask what the 2K parameters add. Add it to decision2's run 3 (frozen G) as well.
- (iii) ICE-V `fit_scorer` on the frozen backbones, ~5M tokens.

**Cost.** (i) and (ii): CPU, 2–3 h (the Mamba slow path takes ~10–20 s per 2K window). (iii): Kaggle T4, 3–5 h (pin `mamba-ssm` wheels).

**Pre-registered:**
- |Mamba − Pythia| on code ≤ 0.05, but Mamba is ≥ 0.10 worse on lookup-right tokens and ≤ Pythia on no-lookup tokens;
- Mamba + ICE-V beats Pythia on code by ≥ 0.03;
- Pythia + ICE-V gains ≤ ⅓ of Mamba's gain;
- ICE-V beats ICE-0 by ≥ 0.01.

**Kill:**
- Lookup-right gap < 0.05: there is no natural-text copy deficit at 370M, so retreat to synthetic recall.
- ICE-V ≤ ICE-0 + 0.005: report ICE-0 as the method.

**Value:** the most *useful* result. A downloadable attention-free model gains hybrid-level recall at 65 B/token with no backbone training. It also addresses weakness 6.

## 3. ICE candidates as free speculative drafts (idea a), rank 3
**Estimate from existing teacher-forced numbers.** A draft exists at 83% of code positions. Share of those positions where the model's greedy token equals the lookup's s*:

| | code | prose |
|---|---|---|
| G_VR | 0.69 | 0.38 |
| B | 0.63 | |
| G | 0.58 | |

- ICE-trained models accept *more* drafts because they defer to the lookup. Speculative decoding preserves output, so the "follows wrong lookups" weakness becomes a free speed property.
- With i.i.d. acceptance, k = 4 gives ≈ 2.7 tokens per verify where a draft exists.

**Novelty: low.**
- Drafting from the context: Prompt Lookup Decoding, REST 2311.08252, SuffixDecoding 2411.04975, SAM-Decoding 2411.10666.
- Speculative decoding on SSMs: STree 2505.14969, Activation Replay (ENLSP 2024), SpecMamba 2509.19873.
- New only: one index serves both quality and speed, and the draft length is gated by the verifier's π_{s*}.

**Experiment.**
- CPU, ~1 h: log acceptance while the model decodes its *own* 256-token generations greedily (the `line_completion.py` loop), with k ∈ {2, 4, 8}.
- T4, ~1 h, after ~3 h of code (snapshot GDN and conv state, replay the accepted prefix): wall-clock speed.

**Pre-registered:** mean accepted length ≥ 1.6 tokens per forward on code and ≥ 1.15 on prose; T4 decode ≥ 1.25× faster than G (today it is 9% slower).

**Kill:** acceptance < 0.5 on own generations, or speedup < 1.1×.

**Value:** a strong slide line ("12× less memory *and* faster code decode"). It is not a paper contribution.

## 4. Truncated state, full table (new), rank 4
**Idea.** The verifier reads only the *current* state, slot features and the candidate log-prob, never the source position's hidden state. So candidates can come from text the backbone never processed. The table needs tokenization only: no prefill.

**Novelty: plausible, unverified.** Infini-gram builds forward-free datastores offline; kNN-LM and RETRO need a forward pass or an encoder. I found no in-context recurrent version.

**Experiment (CPU, local G_VR, ~2 h; ~40-line eval script, no model change).** On `long_context_eval`'s genuine docs > 8K: full prefill vs the backbone reset every 4K, with the table over the whole doc. Reference: B fed only the last 4K.

**Pre-registered:**
- copy > 4K tokens: truncated G_VR is within 0.05 of full G_VR;
- no-lookup tokens: ≥ 0.03 worse (the state carries gist, the table carries verbatim text).

**Kill:** truncation loses > 50% of the far-copy gain.

**Value:**
- a clean memory decomposition, matching the Titans/ATLAS learned-vs-symbolic framing (2501.00663, 2505.23735);
- a product angle: repo-scale memory at 65 B/token with zero prefill. A cross-document mode comes later; ICE resets at EOS today.

## 5. Length generalisation (idea b), rank 5: already built
`long_context_eval.py --clamp_dist` is run 9.

**Novelty: low.** Caches are position-free by construction, and CCQ (2606.01294) already reports recurrent recall collapsing past 4K.

**Pre-register (Kaggle cell, 1–2 h):**
- B's lookup-right loss in the 16–32K bucket is ≥ 0.10 above its 0–4K value;
- G_VR's is ≤ 0.03 above;
- CodeRecall at a 24K gap: G_VR copy_same ≥ B + 0.2.

**Kill:** B degrades by less than 0.03; then drop "length" from the headline.

**Value:** one poster panel.

## 6. Rejected or already covered
- **Engram 2×2:** needs pretraining, ~1 T4-day per arm, and Engram owns the static half.
- **Titans/ATLAS:** costly and a different regime; cite only.
- **MQAR re-runs:** done in round 3; §1's rarity filter explains the frequent-token weakness.
- **ICE in Jamba/Samba/Hymba-style hybrids:** answered by run 4.
- **Hidden-state/SimHash keys:** that is the ICE-K probe (run 6).

## Ranking

| # | idea | cost | novelty | value |
|---|---|---|---|---|
| 1 | copy-predictable gap + division of labour | CPU 1 h + 1 Kaggle cell | moderate | **core thesis** |
| 2 | public-model audit + ICE-0/ICE-V bolt-on | CPU 3 h + T4 4 h | moderate-high | **usefulness** |
| 3 | free drafts | CPU 1 h (+T4) | low | talk punchline |
| 4 | truncated state, full table | CPU 2 h | plausible | mechanism + product |
| 5 | length generalisation | T4 1–2 h | low | poster panel |

**Order:**
1. Today, on CPU: §1 and §4.
2. Add three Kaggle cells: the G forward, run 9 and ICE-0 on frozen G.
3. Then §2 and §3's acceptance logging.
